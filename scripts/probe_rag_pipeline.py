"""
探针：在**现有索引**上验证「rerank / BM25 / 融合」三类手段的真实增益。
只出数据，不改生产代码。用于决定哪些改动值得做。

    ./.venv/Scripts/python.exe scripts/probe_rag_pipeline.py
"""
import os
import re
import sys
import json
import logging
from collections import Counter

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
from src.rag.retrieval_policy import is_low_quality

TOPK = 5
POOL = 40

store = get_vectorstore('configs/rag_config.yaml')
if store.index is None:
    store.load()
docs = store.documents
metas = store.document_metadata
N = len(docs)
print(f'index N={N}')


# ---------------------------------------------------------------- 文本归一化
def normalize_ws(t: str) -> str:
    """把 PDF 抽取的排版碎片压回正常文本。

    MAXWELL 手册 100% 的 chunk 存在「词间3空格 + \\xa0」的两端对齐碎片
    （'Aaron\\xa0\\nAboaf'），直接损害精确短语匹配：
    'Safe   mode' 匹配不上 'safe mode'（实测语料里 safe mode
    原文只匹配 1 chunk，归一化后 4 chunk）。
    """
    t = t.replace('\xa0', ' ')
    t = re.sub(r'[ \t]+', ' ', t)
    return t.strip()


import jieba
jieba.setLogLevel(60)

TOKEN_RE = re.compile(r'[a-z][a-z0-9_\-]*|\d+(?:\.\d+)?')
CJK_RUN = re.compile(r'[\u4e00-\u9fff]+')


def tokenize(t: str):
    t = normalize_ws(t).lower()
    toks = TOKEN_RE.findall(t)
    for run in CJK_RUN.findall(t):
        toks += list(jieba.cut_for_search(run))
        toks += [run[i:i + 2] for i in range(len(run) - 1)]
    return toks


print('tokenizing corpus (normalized text)...')
CORPUS_TOKS = [tokenize(d) for d in docs]
print(f'avg tokens/chunk = {sum(len(t) for t in CORPUS_TOKS)/N:.1f}')

df = Counter()
for t in CORPUS_TOKS:
    df.update(set(t))
IDF = {w: np.log(1 + (N - c + 0.5) / (c + 0.5)) for w, c in df.items()}
AVGDL = sum(len(t) for t in CORPUS_TOKS) / N

INV = {}
for i, t in enumerate(CORPUS_TOKS):
    for w, f in Counter(t).items():
        INV.setdefault(w, []).append((i, f))
print(f'inverted index terms={len(INV)}')

K1, B = 1.5, 0.75


def bm25_scores(q_toks):
    sc = np.zeros(N, dtype=np.float32)
    for w in q_toks:
        postings = INV.get(w)
        if not postings:
            continue
        idw = IDF[w]
        for i, f in postings:
            norm = 1 - B + B * len(CORPUS_TOKS[i]) / AVGDL
            sc[i] += idw * f * (K1 + 1) / (f + K1 * norm)
    return sc


# ---------------------------------------------------------------- reranker
RR_LOCAL = 'models/bge-reranker-v2-m3'
rr = None
if os.path.isdir(RR_LOCAL):
    try:
        from sentence_transformers import CrossEncoder
        rr = CrossEncoder(RR_LOCAL, max_length=512)
        print(f'reranker loaded: {RR_LOCAL}')
    except Exception as e:
        print(f'reranker load FAILED: {type(e).__name__}: {e}')
else:
    print(f'reranker NOT present at {RR_LOCAL} -> rerank pipelines skipped')


# ---------------------------------------------------------------- pipelines
def _clean(rows):
    return [(sc, docs[i], metas[i].get('source_name', '?'), i)
            for sc, i in rows if not is_low_quality(docs[i], 80, 0.55)]


def baseline(q):
    hits = store.search(q, top_k=TOPK)
    return [(h['score'], h['content'], h['metadata'].get('source_name', '?'), -1)
            for h in hits]


def vector_only(q):
    sims, ids = store.index.search(store.embedder.encode(q), POOL)
    return _clean([(float(s), int(i)) for s, i in zip(sims[0], ids[0])])


def bm25_only(q):
    sc = bm25_scores(tokenize(q))
    order = np.argsort(-sc)[:POOL]
    return _clean([(float(sc[i]), int(i)) for i in order])


def hybrid_rrf(q, k=60):
    _, vi = store.index.search(store.embedder.encode(q), POOL)
    vr = {int(i): r for r, i in enumerate(vi[0])}
    bs = bm25_scores(tokenize(q))
    bo = np.argsort(-bs)[:POOL]
    br = {int(i): r for r, i in enumerate(bo)}
    fused = []
    for i in set(vr) | set(br):
        r = 0.0
        if i in vr:
            r += 1.0 / (k + vr[i] + 1)
        if i in br:
            r += 1.0 / (k + br[i] + 1)
        fused.append((r, i))
    fused.sort(key=lambda x: -x[0])
    return _clean(fused[:POOL])


def _rerank(q, base):
    if rr is None or not base:
        return base
    rs = rr.predict([(q, c[1]) for c in base], batch_size=16, show_progress_bar=False)
    scored = sorted(zip([float(x) for x in rs], base), key=lambda x: -x[0])
    return [(s, b[1], b[2], b[3]) for s, b in scored]


def vector_rerank(q):
    return _rerank(q, vector_only(q))


def hybrid_rerank(q):
    return _rerank(q, hybrid_rrf(q))


# ---------------------------------------------------------------- eval
def evaluate(fn):
    rows = []
    for spec in E.QUERIES:
        exp = set(spec['expect_sources'])
        mt = spec['must_terms']
        cands = fn(spec['q'])[:TOPK]          # 只截 top-5，与 search(top_k=5) 对齐
        if not cands:
            rows.append(dict(q=spec['q'], kind=spec['kind'], top1=0.0,
                             top1_src='<empty>', first=None, usable=False, kws=0.0))
            continue
        first, kws = None, set()
        for i, (_, content, src, _) in enumerate(cands, 1):
            hits = E._contains_any(content, mt)
            kws.update(hits)
            if first is None and src in exp and hits:
                first = i
        rows.append(dict(
            q=spec['q'], kind=spec['kind'], top1=float(cands[0][0]),
            top1_src=cands[0][2], first=first, usable=first is not None,
            kws=len(kws) / max(1, len(mt)),
        ))
    return rows


PIPES = [
    ('baseline (现状)', baseline),
    ('vector only', vector_only),
    ('BM25 only', bm25_only),
    ('hybrid RRF', hybrid_rrf),
]
if rr is not None:
    PIPES += [('vector -> rerank', vector_rerank),
              ('hybrid -> rerank', hybrid_rerank)]

print('\n' + '=' * 84)
print(f"{'pipeline':<26}{'usable@5':>10}{'MRR':>8}{'kw_recall':>11}{'meanScore':>11}")
print('=' * 84)
allrows = {}
for name, fn in PIPES:
    rows = evaluate(fn)
    allrows[name] = rows
    us = np.mean([r['usable'] for r in rows])
    mrr = np.mean([1 / r['first'] for r in rows if r['first']])
    kw = np.mean([r['kws'] for r in rows])
    ms = np.mean([r['top1'] for r in rows])
    print(f'{name:<26}{us:>10.3f}{mrr:>8.3f}{kw:>11.3f}{ms:>11.4f}')

base = allrows['baseline (现状)']
names = [n for n, _ in PIPES if n != 'baseline (现状)']
print('\n' + '=' * 84)
print('相对 baseline 的逐查询 usable@5 变化（+=变好  -=变差  ==不变）')
print('=' * 84)
print(f"{'query':<36}" + ''.join(f'{n[:13]:>15}' for n in names))
for i, spec in enumerate(E.QUERIES):
    b = base[i]['usable']
    cells = []
    for n in names:
        v = allrows[n][i]['usable']
        mark = '+' if (v and not b) else ('-' if (b and not v) else '=')
        cells.append(mark + ('Y' if v else 'n'))
    if all(c[0] == '=' for c in cells):
        continue
    print(f"{spec['q'][:34]:<36}" + ''.join(f'{c:>15}' for c in cells))

json.dump(allrows, open('data/results/rag/probe_pipeline.json', 'w'),
          ensure_ascii=False, indent=1, default=str)
print('\nsaved data/results/rag/probe_pipeline.json')
