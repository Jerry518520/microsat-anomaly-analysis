"""融合策略终选：在 usable@5 与 MRR 之间找帕累托最优。

## 上一轮实测给出的矛盾

- G（混合召回 + rerank 打分）：**usable@5 = 1.000**（24/24 全部可用），
  但 **MRR 0.940 -> 0.807**。3 条查询的首位命中从 #1 掉到 #2/#4/#5。
- 反过来，任何保住向量序的做法（RRF、加权）都拿不到那 3 条召回。

根因：rerank 对**全部 40 个候选**重排，等于让一个cross-encoder
完全接管语义序；而向量序在「同源内细粒度相关性」上本就不错
（实测相邻两名差~0.01 但方向可靠），rerank 在这些地方反而更毛刺。

## 本轮要试的三类「约束 rerank」

H. **限定重排范围**：只让 rerank 重排向量 top-h（h=8/12/20），
   池子其余部分保持向量序。限制 rerank 的破坏范围。
I. **分数混合**：final = a*向量分归一化 + (1-a)*rerank分归一化。
   保留向量序作为主轴，rerank 只在分数接近时微调 + 把关键词命中提上来。
J. **rerank 只用于「提升候选」而非「重排」**：把 rerank 分数加到向量分上，
   但向量分为主项（等价于给关键词命中加权，不动语义序）。

判定标准（硬）：**MRR >= 0.940（不低于现状）且 usable@5 尽量高**。
达不到就如实报告取舍，不硬凑。
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


def pool_vector(q):
    sims, ids = store.index.search(store.embedder.encode(q), POOL)
    return [_mk(int(i), float(s)) for s, i in zip(sims[0], ids[0]) if _ok(int(i))]


def pool_hybrid(q):
    pool = {}
    for c in pool_vector(q):
        pool[c['_pos']] = c
    for i, _s in BM.top_n(q, POOL):
        if i not in pool and _ok(i):
            pool[i] = _mk(i, 0.0)
    return list(pool.values())


def _minmax(vals):
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return [(v - lo) / rng for v in vals]


def pipe_A(q):
    return pool_vector(q)


def pipe_H(q, h=12, base='hybrid'):
    """只让 rerank 重排前 h 名，其余保持向量序追加在后面。"""
    pool = pool_hybrid(q) if base == 'hybrid' else pool_vector(q)
    pool.sort(key=lambda c: -c['score'])
    head, tail = pool[:h], pool[h:]
    if head:
        rs = RR.predict([(q, c['content']) for c in head],
                        batch_size=16, show_progress_bar=False)
        for c, s in zip(head, rs):
            c['score'] = float(s)
        head.sort(key=lambda c: -c['score'])
    return head + tail


def pipe_I(q, a=0.7, base='hybrid'):
    """分数混合：a*向量归一化 + (1-a)*rerank归一化。"""
    pool = pool_hybrid(q) if base == 'hybrid' else pool_vector(q)
    if not pool:
        return pool
    vs = _minmax([c['score'] for c in pool])
    rs = RR.predict([(q, c['content']) for c in pool],
                    batch_size=16, show_progress_bar=False)
    rsn = _minmax([float(x) for x in rs])
    out = []
    for c, v, r in zip(pool, vs, rsn):
        c = dict(c)
        c['score'] = a * v + (1 - a) * r
        out.append(c)
    out.sort(key=lambda c: -c['score'])
    return out


def pipe_J(q, w=0.02, base='hybrid'):
    """rerank 分作为小幅加成加到向量分上，不动语义序主轴。

    向量分量纲是余弦(0~1)、rerank 是 logit(可正可负)，
    故先把 rerank 分压到[0,1] 再乘一个小权重。
    """
    pool = pool_hybrid(q) if base == 'hybrid' else pool_vector(q)
    if not pool:
        return pool
    rs = RR.predict([(q, c['content']) for c in pool],
                    batch_size=16, show_progress_bar=False)
    rsn = _minmax([float(x) for x in rs])
    out = []
    for c, r in zip(pool, rsn):
        c = dict(c)
        c['score'] = c['score'] + w * r
        out.append(c)
    out.sort(key=lambda c: -c['score'])
    return out


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
                         kws=len(kws) / max(1, len(mt))))
    return rows


CAND = [('A 纯向量(现状)', pipe_A)]
for h in (8, 12, 20, 40):
    CAND.append((f'H(h={h}) 限定重排', lambda q, h=h: pipe_H(q, h)))
for a in (0.3, 0.5, 0.7, 0.85):
    CAND.append((f'I(a={a}) 分数混合', lambda q, a=a: pipe_I(q, a)))
for w in (0.01, 0.02, 0.05, 0.1):
    CAND.append((f'J(w={w}) rerank加成', lambda q, w=w: pipe_J(q, w)))

print('\n' + '=' * 96)
print(f"{'pipeline':<26}{'usable@5':>10}{'MRR':>8}{'kw_recall':>11}   判定")
print('=' * 96)
res = {}
for name, fn in CAND:
    rows = run(fn)
    res[name] = rows
    us = np.mean([r['usable'] for r in rows])
    mrr = np.mean([1 / r['first'] for r in rows if r['first']])
    kw = np.mean([r['kws'] for r in rows])
    ok = mrr >= 0.940 - 1e-9
    verdict = 'PASS(MRR未降)' if ok else f'✗ MRR -{0.940-mrr:.3f}'
    print(f'{name:<26}{us:>10.3f}{mrr:>8.3f}{kw:>11.3f}   {verdict}')

print('\n' + '=' * 96)
print('帕累托前沿（MRR >= 0.940 的方案里usable@5 最高的）')
print('=' * 96)
best = None
for name, rows in res.items():
    mrr = np.mean([1 / r['first'] for r in rows if r['first']])
    us = np.mean([r['usable'] for r in rows])
    if mrr >= 0.940 - 1e-9 and (best is None or us > best[1]):
        best = (name, us, mrr)
if best:
    print(f'  {best[0]}: usable@5={best[1]:.3f} MRR={best[2]:.3f}')
else:
    print('  无')

print('\n全局最优（不管 MRR）:')
gname = max(res, key=lambda n: np.mean([r['usable'] for r in res[n]]))
grows = res[gname]
print(f'  {gname}: usable@5={np.mean([r["usable"] for r in grows]):.3f} '
      f'MRR={np.mean([1/r["first"] for r in grows if r["first"]]):.3f}')
