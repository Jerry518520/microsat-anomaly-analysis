"""融合公式选型实验：确定「不降MRR」的前提下提升 usable@5 的融合方式。

候选公式（都保留向量分为��，以免 RRF 那种「完全抛弃语义序」的做法
把 MRR 拉低）：
  A. 纯向量（现状）
  B. RRF 融合（排名融合，score 字段换成 RRF 值）
  C. 归一化线性加权：score = (1-a)*cos_norm + a*bm25_norm
  D. 关键词加成：score = cos + a * bm25_norm  （cos 不归一化，保持原量纲）
  E. 关键词加权但只提升不惩罚（bm25 命中且排名靠前者获得加成）

指标：usable@5（召回）与 MRR（首位命中位置）都必须看。
硬要求：MRR 不得低于 A。
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


def _content(i):
    d = docs[i]
    return d if isinstance(d, str) else getattr(d, 'page_content', str(d))


def _cands_vector(q):
    sims, ids = store.index.search(store.embedder.encode(q), POOL)
    out = []
    for sc, i in zip(sims[0], ids[0]):
        i = int(i)
        if is_low_quality(_content(i), 80, 0.55):
            continue
        out.append({'score': float(sc), 'content': _content(i),
                    'metadata': metas[i], '_pos': i})
    return out


def _kw(q):
    return BM.top_n(q, POOL)


def build(q, mode, alpha=0.3):
    v = _cands_vector(q)
    kwt = _kw(q)
    kpos = [i for i, _ in kwt]
    kscore = {i: s for i, s in kwt}

    if mode == 'A':
        return v

    vpos = {c['_pos']: c for c in v}
    vrank = {c['_pos']: r for r, c in enumerate(v)}

    if mode == 'B':   # RRF
        pool = {}
        for r, c in enumerate(v):
            pool[c['_pos']] = {'rrf': 1.0 / (60 + r + 1), 'c': c}
        for r, i in enumerate(kpos):
            if i in pool:
                pool[i]['rrf'] += 1.0 / (60 + r + 1)
            else:
                cc = {'score': 0.0, 'content': _content(i), 'metadata': metas[i], '_pos': i}
                pool[i] = {'rrf': 1.0 / (60 + r + 1), 'c': cc}
        out = list(pool.values())
        out.sort(key=lambda x: -x['rrf'])
        return [{**x['c'], 'score': x['rrf']} for x in out]

    # 归一化 BM25 到 [0,1]
    if kscore:
        mx = max(kscore.values()) or 1.0
    else:
        mx = 1.0

    pool = {c['_pos']: dict(c) for c in v}
    for i in kpos:
        if i in pool:
            continue
        if is_low_quality(_content(i), 80, 0.55):
            continue
        pool[i] = {'score': 0.0, 'content': _content(i),
                   'metadata': metas[i], '_pos': i}

    if mode == 'D':
        for i in pool:
            pool[i]['score'] = pool[i]['score'] + alpha * (kscore.get(i, 0.0) / mx)
        return sorted(pool.values(), key=lambda c: -c['score'])

    if mode == 'E':   # 只提升：命中即按排名给递减加成
        for r, i in enumerate(kpos):
            if i in pool:
                pool[i]['score'] = pool[i]['score'] + alpha / (1.0 + r)
        return sorted(pool.values(), key=lambda c: -c['score'])

    if mode == 'C':   # 线性加权（cos 归一化到0-1）
        if pool:
            lo = min(c['score'] for c in pool.values())
            hi = max(c['score'] for c in pool.values())
            rng = (hi - lo) or 1.0
            for i in pool:
                cn = (pool[i]['score'] - lo) / rng
                kn = kscore.get(i, 0.0) / mx
                pool[i]['score'] = (1 - alpha) * cn + alpha * kn
        return sorted(pool.values(), key=lambda c: -c['score'])
    raise ValueError(mode)


def run(mode, alpha=0.3):
    rows = []
    for spec in E.QUERIES:
        exp = set(spec['expect_sources'])
        mt = spec['must_terms']
        cands = build(spec['q'], mode, alpha)
        #走与生产一致的 select_candidates
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
                         top1src=sel[0]['metadata'].get('source_name') if sel else '-'))
    return rows


print('\n' + '=' * 92)
print(f"{'mode':<28}{'usable@5':>10}{'MRR':>8}{'kw_recall':>11}{'top1mean':>10}")
print('=' * 92)
res = {}
for mode, alphas in [('A', [0]), ('B', [0]), ('C', [0.2, 0.3, 0.4, 0.5]),
                     ('D', [0.02, 0.05, 0.1, 0.2]), ('E', [0.005, 0.01, 0.02, 0.05])]:
    for a in alphas:
        rows = run(mode, a)
        name = mode if a == 0 else f'{mode}(a={a})'
        res[name] = rows
        us = np.mean([r['usable'] for r in rows])
        mrr = np.mean([1 / r['first'] for r in rows if r['first']])
        kw = np.mean([r['kws'] for r in rows])
        t1 = np.mean([r['top1'] for r in rows])
        flag = '' if mrr >= 0.940 - 1e-9 else '  <-- MRR下降'
        print(f'{name:<28}{us:>10.3f}{mrr:>8.3f}{kw:>11.3f}{t1:>10.4f}{flag}')

base = res['A']
print('\n' + '=' * 92)
print('相对 A 的 usable@5 变化（+=变好 -=变差）')
print('=' * 92)
for name, rows in res.items():
    if name == 'A':
        continue
    diffs = [(b['q'], b['usable'], r['usable'])
             for b, r in zip(base, rows) if b['usable'] != r['usable']]
    if diffs:
        print(f'  {name}:')
        for q, bu, ru in diffs:
            print(f'      {"变好" if ru else "变差"}: {q}')
