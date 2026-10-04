"""端到端验证：真实 FAISSVectorStore，5 个查询修前 vs 修后 top5。"""
import os, sys, json, pickle
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, '.env'), override=True)

from src.rag.vectorstore import FAISSVectorStore

QUERIES = ['温度突变', '信号丢失', '姿控异常', '电源故障', '数据丢失']
CFG = os.path.join(ROOT, 'configs/rag_config.yaml')

vs = FAISSVectorStore(config_path=CFG)
assert vs.load(), "索引加载失败"
print(f'载入 chunk数: {len(vs.documents)}  top_k={vs.top_k} '
      f'threshold={vs.score_threshold} rel_delta={vs.rel_delta} '
      f'caps={vs.priority_caps} low_conf_th={vs.low_confidence_threshold}')

# ---- 修前逻辑（复刻改动前的 search：search_k=max(top_k*3,15)，无质量过滤，
#      源多样性按 source_name 均分，无priority）----
from collections import defaultdict
import faiss

def legacy_search(q, top_k=5, thresh=0.3):
    qe = vs.embedder.encode(q)
    scores, idxs = vs.index.search(qe, max(top_k * 3, 15))
    docs = vs.documents
    metas = vs.document_metadata
    res = []
    for sc, ix in zip(scores[0], idxs[0]):
        if ix == -1:
            continue
        sc = float(sc)
        if sc < thresh:
            continue
        c = docs[ix]
        content = c.page_content if hasattr(c, 'page_content') else (
            c if isinstance(c, str) else str(c))
        res.append({'content': content, 'score': sc, 'metadata': metas[ix]})
    if len(res) > top_k:
        by = defaultdict(list)
        for r in res:
            by[r['metadata'].get('source_name', 'unknown')].append(r)
        mx = max(2, (top_k // len(by)) + 1)
        div, cnt = [], defaultdict(int)
        for r in sorted(res, key=lambda x: -x['score']):
            s = r['metadata'].get('source_name', 'unknown')
            if cnt[s] < mx:
                div.append(r); cnt[s] += 1
            if len(div) >= top_k:
                break
        res = div
    return res


def show(res, title):
    print(f'\n{title}')
    for i, r in enumerate(res):
        m = r['metadata']
        lc = ' [低置信度]' if r.get('low_confidence') else ''
        print(f'  [{i+1}] {r["score"]:.4f} p={m.get("priority")} '
              f'{m.get("source_name")}{lc}')
        print(f'{m.get("filename","")} p{m.get("page","?")} | '
              f'{r["content"][:60].replace(chr(10)," ")}...')


out = {}
print('\n' + '=' * 78)
print('修前 vs 修后（真实 FAISSVectorStore 端到端）')
print('=' * 78)
for q in QUERIES:
    old = legacy_search(q)
    new = vs.search(q, top_k=5)
    show(old, f'【修前】Q: "{q}"')
    show(new, f'【修后】Q: "{q}"')
    out[q] = {
        'old': [{'rank': i+1, 'score': round(r['score'], 4),
                 'source': r['metadata'].get('source_name'),
                 'priority': r['metadata'].get('priority'),
                 'page': r['metadata'].get('page'),
                 'preview': r['content'][:70].replace('\n', ' ')}
                for i, r in enumerate(old)],
        'new': [{'rank': i+1, 'score': round(r['score'], 4),
                 'source': r['metadata'].get('source_name'),
                 'priority': r['metadata'].get('priority'),
                 'page': r['metadata'].get('page'),
                 'low_confidence': r.get('low_confidence'),
                 'preview': r['content'][:70].replace('\n', ' ')}
                for i, r in enumerate(new)],
    }

print('\n' + '=' * 78)
print('汇总')
print('=' * 78)
print(f'{"query":10s}{"修前top1":32s}{"修后top1":32s}{"低置信":>8s}{"来源数":>8s}')
for q in QUERIES:
    o = out[q]['old'][0]['source'] if out[q]['old'] else '-'
    n = out[q]['new'][0]['source'] if out[q]['new'] else '-'
    lc = any(c['low_confidence'] for c in out[q]['new'])
    ns = len({c['source'] for c in out[q]['new']})
    print(f'{q:10s}{o:32s}{n:32s}{str(lc):>8s}{ns:>8d}')

json.dump(out, open('data/verify_final.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)

# ---- 空结果防护验证：阈值不能调到检索不到东西 ----
print('\n' + '=' * 78)
print('阈值/参数稳健性')
print('=' * 78)
for th in [0.3, 0.45, 0.55, 0.6]:
    n = sum(len(vs.search(q, top_k=5, score_threshold=th)) for q in QUERIES)
    print(f'  score_threshold={th}: 5 个查询共返回 {n} 条')

print('\n=== search_with_context 低置信度提示 ===')
ctx, srcs = vs.search_with_context('姿控异常', top_k=3, max_tokens=800)
print(ctx[:300].replace('\n', ' '))

print('\n→ data/verify_final.json')
