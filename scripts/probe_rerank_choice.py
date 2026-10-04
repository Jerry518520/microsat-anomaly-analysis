"""rerank 作为「统一打分器」的选型实验。

## 为什么必须用 rerank 而不是调融合权重

上一轮实测（scripts/probe_fusion_weight.py）暴露了一个结构性矛盾：

- 向量侧找到的合格 chunk：电源电压跌落排 #10、photodiode 排 #18
- BM25 侧能找到这些 chunk（且排在 #1）
- 但**纯 BM25 命中的 chunk 在向量侧没有分**（score=0.0），
  按 score 降序永远排在候选池末尾，进不了 top-5。

RRF（mode B）能给它们可比的分数，代价是**丢掉向量分数的语义序**，
实测 MRR 从 0.940 掉到 0.902。

rerank 是第三条路：它对「向量召回的」和「BM25 召回的」用**同一个**
cross-encoder 打分，两侧候选因此**天然可比**，不需要任何缩放系数。
本实验验证它能否在不牺牲 MRR 的前提下拿到那2 个召回。

注意：rerank 分数会取代向量相似度成为 ``score`` 字段。
这**不违反**RetrievalPolicy 的约定——该约定是
「priority 决定入选资格、score 决定最终顺序」，
rerank 只是换了一个更好的 score，仍然是 score 定序、priority 不参与排序。
"""
import os
import sys
import logging

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, '.env'), override=True)
logging.disable(logging.CRITICAL)
import transformers
transformers.logging.set_verbosity_error()

import eval_retrieval_quality as E
from src.rag.vectorstore import get_vectorstore
from src.rag.retrieval_policy import is_low_quality, select_candidates
from src.rag.bm25 import BM25Index

POOL = 40
TOPK = 5

store = get_vectorstore('configs/rag_config.yaml')
if store.index is None:
    store.load()
docs = store.documents
metas = store.document_metadata
N = len(docs)
print(f'N={N}')

print('building BM25...')
BM = BM25Index([d if isinstance(d, str) else d.page_content for d in docs])

from sentence_transformers import CrossEncoder
import torch
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
RR = CrossEncoder('models/bge-reranker-v2-m3', max_length=512, device=dev)
print(f'reranker on {dev}')


def _content(i):
    d = docs[i]
    return d if isinstance(d, str) else getattr(d, 'page_content', str(d))


def _ok(i):
    return not is_low_quality(_content(i), 80, 0.55)


def _mk(i, score):
    return {'score': score, 'content': _content(i), 'metadata': metas[i], '_pos': i}


def cand_vector(q):
    sims, ids = store.index.search(store.embedder.encode(q), POOL)
    return [_mk(int(i), float(s)) for s, i in zip(sims[0], ids[0]) if _ok(int(i))]


def cand_hybrid(q):
    pool = {}
    for c in cand_vector(q):
        pool[c['_pos']] = c
    for i, _s in BM.top_n(q, POOL):
        if i not in pool and _ok(i):
            pool[i] = _mk(i, 0.0)
    return list(pool.values())


def rerank(q, cands):
    if not cands:
        return cands
    rs = RR.predict([(q, c['content']) for c in cands],
                    batch_size=16, show_progress_bar=False)
    out = []
    for c, s in zip(cands, rs):
        c = dict(c)
        c['score'] = float(s)
        out.append(c)
    out.sort(key=lambda c: -c['score'])
    return out


PIPES = {
    'A 纯向量(现状)': lambda q: cand_vector(q),
    'E 混合(不加rerank)': lambda q: sorted(cand_hybrid(q), key=lambda c: -c['score']),
    'F 向量+rerank': lambda q: rerank(q, cand_vector(q)),
    'G 混合+rerank': lambda q: rerank(q, cand_hybrid(q)),
}


def run(fn):
    rows = []
    for spec in E.QUERIES:
        exp = set(spec['expect_sources'])
        mt = spec['must_terms']
        cands = fn(spec['q'])
        if len(cands) > TOPK:
            sel, _ = select_candidates(
                cands, top_k=TOPK, rel_delta=0.12, max_reserved=None,
                reserve_priority_max=2, priority_caps=store.priority_caps)
        else:
            sel = sorted(cands, key=lambda x: -x['score'])[:TOPK]
        first, kws = None, set()
        for i, c in enumerate(sel, 1):
            hits = E._contains_any(c['content'], mt)
            kws.update(hits)
            if first is None and c['metadata'].get('source_name') in exp and hits:
                first = i
        rows.append(dict(q=spec['q'], kind=spec['kind'], first=first,
                         usable=first is not None,
                         kws=len(kws) / max(1, len(mt)),
                         top1=sel[0]['score'] if sel else 0.0,
                         top1src=sel[0]['metadata'].get('source_name') if sel else '-',
                         topk_srcs=[c['metadata'].get('source_name') for c in sel]))
    return rows


print('\n' + '=' * 92)
print(f"{'pipeline':<24}{'usable@5':>10}{'MRR':>8}{'kw_recall':>11}")
print('=' * 92)
res = {}
for name, fn in PIPES.items():
    rows = run(fn)
    res[name] = rows
    us = np.mean([r['usable'] for r in rows])
    mrr = np.mean([1 / r['first'] for r in rows if r['first']])
    kw = np.mean([r['kws'] for r in rows])
    flag = '' if mrr >= 0.940 - 1e-9 else '  <-- MRR低于基线'
    print(f'{name:<24}{us:>10.3f}{mrr:>8.3f}{kw:>11.3f}{flag}')

base = res['A 纯向量(现状)']
print('\n' + '=' * 92)
print('相对 A 的逐查询 usable@5 变化')
print('=' * 92)
for name, rows in res.items():
    if name == 'A 纯向量(现状)':
        continue
    d = [(b['q'], b['usable'], r['usable'], b['first'], r['first'])
         for b, r in zip(base, rows) if b['usable'] != r['usable']]
    print(f'  {name}: {len(d)} 条变化')
    for q, bu, ru, bf, rf in d:
        print(f'      {"变好" if ru else "变差"}: {q[:44]:46s} first {bf}->{rf}')

print('\n' + '=' * 92)
print('MRR 位置退化明细（first 变大 = 首位命中变晚）')
print('=' * 92)
for name, rows in res.items():
    if name == 'A 纯向量(现状)':
        continue
    worse = [(b['q'], b['first'], r['first'])
             for b, r in zip(base, rows)
             if (r['first'] or 99) > (b['first'] or 99)]
    print(f'  {name}: {len(worse)} 条')
    for q, bf, rf in worse:
        print(f'      {q[:44]:46s} {bf} -> {rf}')
