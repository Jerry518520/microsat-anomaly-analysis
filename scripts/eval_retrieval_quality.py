"""
RAG 检索质量评测（可复现基线）
================================

用途：为「优化 RAG 检索质量」提供**可复现的量化对比**。同一个脚本、
同一份查询集与标注，跑在改动前/改动后两次，输出差异表。

设计原则（为何这样定指标）
--------------------------
1. **不只看相似度分数**。BGE-M3 归一化内积的绝对值窄（实测 top-15 落
   0.45~0.67），分数高低不等于答得好坏。真正要衡量的是
   「该被引用的资料有没有进 top-k」「进的是不是对的源」。
2. **来源命中（source_hit）**：查询标注了 ``expect_sources``，
   top-k 中出现任一即算命中。这是「召回是否覆盖到对口资料」。
3. **关键词覆盖（kw_recall）**：查询标注了 ``must_terms``，
   top-k 命中的关键词比例。专有名词（CADC0872 / EPS / thruster）
   靠向量检索容易糊，用它把「看似相关但答非所问」暴露出来。
4. **MRR**：第一个「来源对且含关键词」的结果排名的倒数。
   惩罚「对的资料排在第 4 条」——LLM 上下文里靠后的片段易被忽略。
5. **答案可用性（usable@k）**：来源对 **且** 至少含 1 个 must_term。
   这是最接近「LLM 能不能答对」的指标，也是优化的主目标。

用法
----
    ./.venv/Scripts/python.exe scripts/eval_retrieval_quality.py \
        --tag before            # 存档为 results/retrieval_eval_before.json

    ./.venv/Scripts/python.py scripts/eval_retrieval_quality.py --compare \
        --tag after --baseline results/retrieval_eval_before.json

加 ``--dump`` 打印每条查询的 top-k 明细（含来源、分数、片段预览）。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)


# --------------------------------------------------------------------------
# 查询集与标注
# --------------------------------------------------------------------------
# 每条查询：
#   q            查询文本
#   kind         查询类型，仅用于分组统计
#   expect_sources  期望命中的来源名（任一命中即算source_hit）
#   must_terms   期望在 top-k 片段中出现的关键词（子串匹配，忽略大小写）
#   note         诊断意图
#
# 标注依据：知识库实际内容 + 项目一手结论（experiment_knowledge.md /
# PAPER_TABLE.md）。凡expect_sources 标Experiment Knowledge Base 的，
# 指的是「只有项目自己的实验结论能回答」，通用手册回答不了。
QUERIES: List[Dict[str, Any]] = [
    # ---------- 用户实测的 4 条，复核用 ----------
    {
        "q": "立方体卫星姿态控制失稳的常见原因",
        "kind": "zh_general",
        "expect_sources": ["NASA SOA 2024", "ITU Small Satellite Handbook CN",
                           "MAXWELL Mission Handbook", "ITU Small Satellite Handbook EN"],
        "must_terms": ["attitude", "姿态", "control", "控制"],
        "note": "用户实测 top1=0.5803。中文问英文库，考察跨语言召回",
    },
    {
        "q": "电源系统电压跌落如何诊断",
        "kind": "zh_general",
        "expect_sources": ["NASA SOA 2024", "MAXWELL Mission Handbook",
                           "ITU Small Satellite Handbook CN", "ITU Small Satellite Handbook EN"],
        "must_terms": ["voltage", "电压", "power", "电源", "EPS"],
        "note": "用户实测 top1=0.5503",
    },
    {
        "q": "遥测数据缺失在地面站如何处理",
        "kind": "zh_general",
        "expect_sources": ["NASA SOA 2024", "ITU Small Satellite Handbook CN",
                           "ITU Small Satellite Handbook EN", "MAXWELL Mission Handbook"],
        "must_terms": ["telemetry", "遥测", "ground", "地面"],
        "note": "用户实测 top1=0.6131",
    },
    {
        "q": "磁力计读数突变说明什么",
        "kind": "zh_general",
        "expect_sources": ["NASA SOA 2024", "MAXWELL Mission Handbook",
                           "ITU Small Satellite Handbook CN", "ITU Small Satellite Handbook EN"],
        "must_terms": ["magnetometer", "磁力计", "magnetic", "磁"],
        "note": "用户实测 top1=0.5568",
    },

    # ---------- 项目一手知识：只有 experiment_knowledge.md 能答 ----------
    {
        "q": "CADC0874通道对应什么物理量",
        "kind": "project_fact",
        "expect_sources": ["Experiment Knowledge Base"],
        "must_terms": ["CADC0874", "磁力计", "I_B_FB_MM_2"],
        "note": "CADC 编号是精确编号，向量检索极易糊；通用手册不可能有",
    },
    {
        "q": "本项目最终方案的F1是多少",
        "kind": "project_fact",
        "expect_sources": ["Experiment Knowledge Base"],
        "must_terms": ["0.6281", "gate_perchannel", "F1"],
        "note": "项目最终性能，只有实验知识库有",
    },
    {
        "q": "磁力计和光电二极管两类通道的异常强度差异",
        "kind": "project_fact",
        "expect_sources": ["Experiment Knowledge Base"],
        "must_terms": ["磁力计", "光电二极管", "F1"],
        "note": "项目实测结论：强通道/弱通道分化",
    },
    {
        "q": "投票阈值 vote_threshold 0.25 是怎么来的",
        "kind": "project_fact",
        "expect_sources": ["Experiment Knowledge Base"],
        "must_terms": ["vote_threshold", "0.25", "网格搜索"],
        "note": "项目参数溯源，含「无网格搜索」这一诚实声明",
    },
    {
        "q": "段级异常率和点级异常率有什么区别",
        "kind": "project_fact",
        "expect_sources": ["Experiment Knowledge Base"],
        "must_terms": ["20.4", "33.0", "段级", "点级"],
        "note": "口径混用是本项目高频踩坑点",
    },
    {
        "q": "OPS-SAT数据集的异常标注由谁完成",
        "kind": "project_fact",
        "expect_sources": ["Experiment Knowledge Base", "Ruszczak OPS-SAT AD 2025"],
        "must_terms": ["ESA", "工程师", "annotat", "标注"],
        "note": "论文原文亦可答，故双来源",
    },

    # ---------- 专有名词：混合检索的试金石 ----------
    {
        "q": "EPS电源模块的故障模式",
        "kind": "acronym",
        "expect_sources": ["NASA SOA 2024", "MAXWELL Mission Handbook",
                           "ITU Small Satellite Handbook CN", "ITU Small Satellite Handbook EN"],
        "must_terms": ["EPS", "electrical power", "电源"],
        "note": "EPS 是缩写，纯向量检索常糊成energy performance score",
    },
    {
        "q": "thruster 推力器故障如何排查",
        "kind": "acronym",
        "expect_sources": ["NASA SOA 2024", "MAXWELL Mission Handbook",
                           "ITU Small Satellite Handbook EN", "ITU Small Satellite Handbook CN"],
        "must_terms": ["thruster", "推力器", "thruster firing", "propulsion"],
        "note": "专有英文术语",
    },
    {
        "q": "reaction wheel 反作用飞轮卡死",
        "kind": "acronym",
        "expect_sources": ["NASA SOA 2024", "MAXWELL Mission Handbook",
                           "ITU Small Satellite Handbook EN", "ITU Small Satellite Handbook CN"],
        "must_terms": ["reaction wheel", "飞轮", "wheel", "momentum wheel"],
        "note": "ADCS 核心执行器",
    },
    {
        "q": "safe mode 进入条件与恢复流程",
        "kind": "acronym",
        "expect_sources": ["NASA SOA 2024", "MAXWELL Mission Handbook",
                           "ITU Small Satellite Handbook EN", "ITU Small Satellite Handbook CN"],
        "must_terms": ["safe mode", "safe-mode", "安全模式", "safemode"],
        "note": "带连字符的复合术语，分词/关键词检索敏感",
    },

    # ---------- 英文查询：验证非中文场景没被牺牲 ----------
    {
        "q": "magnetometer anomaly detection",
        "kind": "en_general",
        "expect_sources": ["NASA SOA 2024", "MAXWELL Mission Handbook",
                           "Ruszczak OPS-SAT AD 2025", "ITU Small Satellite Handbook EN"],
        "must_terms": ["magnetometer", "magnetic", "anomal"],
        "note": "英文基线，防止中文优化反噬英文",
    },
    {
        "q": "ADCS safe mode reaction wheel",
        "kind": "en_general",
        "expect_sources": ["NASA SOA 2024", "MAXWELL Mission Handbook",
                           "ITU Small Satellite Handbook EN"],
        "must_terms": ["ADCS", "reaction wheel", "safe mode", "attitude"],
        "note": "build_index.py 自带测试查询",
    },
    {
        "q": "photodiode zero value gap anomaly",
        "kind": "en_general",
        # Experiment Knowledge Base 为**首要**合法来源：该chunk 原文为
        # 「光电二极管异常含义…### 异常类型：1. Unusual shapes 2. Peaks
        # 3. **Zero values**（零值）4. **Gaps**（数据间隙）」，
        # 同时命中 photodiode(光电二极管) / zero / gap 三个关键词，
        # 是对本查询最直接的答案。
        # 原标注只认英文手册，导致扩充后（该chunk 升入 top1）被误判为
        #「来源变差」；实测原top1 是 NASA 的**卫星总线参数表**
        #（'Space Dynamics Laboratory USA 400 ... LEO/GEO/GTO'），
        # 与photodiode 异常无关。属标注错误，已修正。
        "expect_sources": ["Experiment Knowledge Base", "NASA SOA 2024",
                           "ITU Small Satellite Handbook EN",
                           "Ruszczak OPS-SAT AD 2025", "MAXWELL Mission Handbook"],
        # 「光电二极管」是 photodiode 的中文表述，须并入关键词表，
        # 否则中文 chunk 虽是正确的答案却判为未命中。
        "must_terms": ["photodiode", "photo diode", "光电二极管",
                       "zero", "gap", "anomal"],
        "note": "对应实验里的 Zero values / Gaps 异常类型",
    },
    {
        "q": "satellite attitude control magnetic torquer",
        "kind": "en_general",
        "expect_sources": ["NASA SOA 2024", "MAXWELL Mission Handbook",
                           "ITU Small Satellite Handbook EN"],
        "must_terms": ["attitude control", "magnetic torquer", "torque", "magnetorqu"],
        "note": "磁力矩器，ADCS 粗控执行器",
    },
    # ---------- 中文 ITU 应答：验证中文库是否真能被检索到 ----------
    {
        "q": "小卫星的对地通信频段选择",
        "kind": "zh_itu",
        "expect_sources": ["ITU Small Satellite Handbook CN", "ITU Small Satellite Handbook EN",
                           "NASA SOA 2024", "MAXWELL Mission Handbook"],
        "must_terms": ["频段", "frequency", "band", "通信", "transmission"],
        "note": "ITU 手册核心内容（频谱管理）",
    },
    {
        "q": "空间碎片减缓要求",
        "kind": "zh_itu",
        "expect_sources": ["ITU Small Satellite Handbook CN", "ITU Small Satellite Handbook EN",
                           "NASA SOA 2024", "MAXWELL Mission Handbook"],
        "must_terms": ["碎片", "debris", "mitigation", "减缓"],
        "note": "ITU/监管类内容",
    },
    {
        "q": "立方体卫星标准化限制",
        "kind": "zh_itu",
        "expect_sources": ["ITU Small Satellite Handbook CN", "ITU Small Satellite Handbook EN",
                           "NASA SOA 2024", "MAXWELL Mission Handbook"],
        "must_terms": ["CubeSat", "立方体", "standard", "标准", "nanosatellite"],
        "note": "ITU 手册有专门章节",
    },
    # ---------- 故障诊断类：NASA SOA 主场 ----------
    {
        "q": "遥测零值与传感器故障如何区分",
        "kind": "diagnosis",
        # Experiment Knowledge Base 亦为合法来源：其条目1「异常类型」节
        # 逐条定义了 Zero values（零值）=「传感器读数归零，可能指示传感器
        # 故障或数据丢失」，正面回答了本查询。实测该chunk 位于
        # experiment_knowledge.md 第3 个切分单元。原标注漏掉此源，
        # 导致 rerank把它排到 #1 时被误判为「退化」，属标注错误，已修正。
        "expect_sources": ["NASA SOA 2024", "Ruszczak OPS-SAT AD 2025",
                           "MAXWELL Mission Handbook", "ITU Small Satellite Handbook EN",
                           "Experiment Knowledge Base"],
        "must_terms": ["zero", "sensor", "故障", "fault", "零值"],
        "note": "项目最核心的解释类问题",
    },
    {
        "q": "太阳电池阵输出功率下降的原因",
        "kind": "diagnosis",
        "expect_sources": ["NASA SOA 2024", "MAXWELL Mission Handbook",
                           "ITU Small Satellite Handbook EN", "ITU Small Satellite Handbook CN"],
        #语料实测 NASA SOA 用 solar cells（53 chunk）与 solar array（61 chunk）
        # 两种写法，「太阳电池阵」对应二者；只标 solar array 会漏掉
        # 真正在讲「电池退化因素」的 NASA 段落。
        "must_terms": ["solar array", "solar cell", "power", "photovoltaic",
                       "太阳", "电池"],
        "note": "电源类故障",
    },
    {
        "q": "陀螺仪漂移与温度的关系",
        "kind": "diagnosis",
        # ⚠ 语料实测：gyro AND (drift|bias|temperature) 仅 7 个 chunk，
        # 且全是 NASA 的「陀螺型号对照表」/「其他陀螺类型」等，
        # **无一条讲漂移-温度关系**（gyro drift / drift rate / bias
        # instability 全部 0 命中）。本查询属**语料覆盖缺口**，
        # 任何检索策略都不可能召回正确内容。保留此查询以如实记录
        # 该缺口，不因难召回而删除。
        "expect_sources": ["NASA SOA 2024", "MAXWELL Mission Handbook",
                           "ITU Small Satellite Handbook EN"],
        "must_terms": ["gyro", "drift", "陀螺", "temperature", "温度"],
        "note": "COVERAGE GAP：语料无「漂移-温度」内容，用于记录缺口",
    },
]

EXPECTED_SOURCE_SET = {q["q"] for q in QUERIES}


# --------------------------------------------------------------------------
# 评测
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    return (s or "").lower()


def _contains_any(text: str, terms: Sequence[str]) -> List[str]:
    low = _norm(text)
    return [t for t in terms if _norm(t) in low]


@dataclass
class QueryResult:
    q: str
    kind: str
    top1_score: float
    top1_source: str
    topk_sources: List[str] = field(default_factory=list)
    topk_scores: List[float] = field(default_factory=list)
    source_hit_at_1: bool = False
    source_hit_at_k: bool = False
    kw_recall: float = 0.0
    hit_terms: List[str] = field(default_factory=list)
    rank_first_usable: Optional[int] = None
    usable_at_k: bool = False
    low_confidence: bool = False


def evaluate(store, top_k: int, queries: List[Dict[str, Any]],
             dump: bool = False) -> Dict[str, Any]:
    results: List[QueryResult] = []
    for spec in queries:
        hits = store.search(spec["q"], top_k=top_k)
        if not hits:
            results.append(QueryResult(
                q=spec["q"], kind=spec["kind"], top1_score=0.0,
                top1_source="<empty>", source_hit_at_1=False, source_hit_at_k=False,
            ))
            continue

        sources = [h["metadata"].get("source_name", "unknown") for h in hits]
        scores = [float(h["score"]) for h in hits]
        contents = [h["content"] for h in hits]
        expect = set(spec["expect_sources"])

        # 关键词覆盖：按 must_term 是否在任何一条 top-k 片段中出现
        found: set = set()
        for c in contents:
            found.update(_contains_any(c, spec["must_terms"]))
        kw_recall = len(found) / max(1, len(spec["must_terms"]))

        first_usable = None
        for i, (s, c) in enumerate(zip(sources, contents), 1):
            if s in expect and _contains_any(c, spec["must_terms"]):
                first_usable = i
                break

        r = QueryResult(
            q=spec["q"],
            kind=spec["kind"],
            top1_score=scores[0],
            top1_source=sources[0],
            topk_sources=sources,
            topk_scores=[round(s, 4) for s in scores],
            source_hit_at_1=sources[0] in expect,
            source_hit_at_k=any(s in expect for s in sources),
            kw_recall=round(kw_recall, 4),
            hit_terms=sorted(found),
            rank_first_usable=first_usable,
            usable_at_k=any(
                s in expect and _contains_any(c, spec["must_terms"])
                for s, c in zip(sources, contents)
            ),
            low_confidence=bool(hits[0].get("low_confidence", False)),
        )
        results.append(r)

        if dump:
            print(f"\n{'='*78}\nQ: {r.q}   [{r.kind}]")
            print(f"   top1={r.top1_score:.4f} src={r.top1_source}"
                  f" | source_hit@1={r.source_hit_at_1} source_hit@k={r.source_hit_at_k}"
                  f" | kw_recall={r.kw_recall:.2f} first_usable={r.rank_first_usable}"
                  f" | lowconf={r.low_confidence}")
            for i, (s, sc) in enumerate(zip(sources, scores), 1):
                mark = "OK " if s in expect else "   "
                prev = contents[i - 1][:70].replace("\n", " ")
                print(f"   {mark}[{i}] {sc:.4f} {s:38s} | {prev}...")

    # ---- 汇总指标 ----
    def _avg(vals: Sequence[float]) -> float:
        return float(statistics.mean(vals)) if vals else 0.0

    def _agg(rows: List[QueryResult]) -> Dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {}
        rr = [1.0 / r.rank_first_usable for r in rows if r.rank_first_usable]
        return {
            "n": n,
            "source_hit@1": round(sum(r.source_hit_at_1 for r in rows) / n, 4),
            "source_hit@k": round(sum(r.source_hit_at_k for r in rows) / n, 4),
            "usable@k": round(sum(r.usable_at_k for r in rows) / n, 4),
            "kw_recall": round(_avg([r.kw_recall for r in rows]), 4),
            "mrr": round(_avg(rr), 4),
            "top1_score_mean": round(_avg([r.top1_score for r in rows]), 4),
            "top1_score_min": round(min(r.top1_score for r in rows), 4),
            "low_conf_ratio": round(sum(r.low_confidence for r in rows) / n, 4),
        }

    by_kind: Dict[str, Any] = {}
    for kind in sorted({r.kind for r in results}):
        by_kind[kind] = _agg([r for r in results if r.kind == kind])

    # 来源分布：看哪个源在吃 top_k 名额
    src_dist: Dict[str, int] = {}
    for r in results:
        for s in r.topk_sources:
            src_dist[s] = src_dist.get(s, 0) + 1

    return {
        "top_k": top_k,
        "n_queries": len(results),
        "overall": _agg(results),
        "by_kind": by_kind,
        "source_slot_share": dict(sorted(src_dist.items(), key=lambda x: -x[1])),
        "results": [asdict(r) for r in results],
    }


# --------------------------------------------------------------------------
def _fmt_delta(a: float, b: float, pct: bool = False) -> str:
    d = b - a
    if pct:
        rel = (d / a * 100) if a else 0.0
        return f"{a:.4f} -> {b:.4f}  ({d:+.4f}, {rel:+.1f}%)"
    return f"{a:.4f} -> {b:.4f}  ({d:+.4f})"


def compare(base: Dict[str, Any], new: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("优化前 vs 优化后")
    print("=" * 78)
    bo, no = base["overall"], new["overall"]
    print(f"{'指标':<20}{'优化前':>12}{'优化后':>12}   变化")
    print("-" * 78)
    for k in ("source_hit@1", "source_hit@k", "usable@k", "kw_recall",
              "mrr", "top1_score_mean", "low_conf_ratio"):
        if k in bo and k in no:
            better = "↑" if no[k] > bo[k] else ("↓" if no[k] < bo[k] else "=")
            print(f"{k:<20}{bo[k]:>12.4f}{no[k]:>12.4f}   "
                  f"{no[k]-bo[k]:+.4f} {better}")

    # 逐查询
    print("\n" + "-" * 78)
    print("逐查询变化（仅列出有变化的）")
    print("-" * 78)
    bmap = {r["q"]: r for r in base["results"]}
    nmap = {r["q"]: r for r in new["results"]}
    n_changed = 0
    for q in [x["q"] for x in QUERIES]:
        b, n = bmap.get(q), nmap.get(q)
        if not b or not n:
            continue
        changed = (b["topk_sources"] != n["topk_sources"]
                   or abs(b["top1_score"] - n["top1_score"]) > 1e-6
                   or b["rank_first_usable"] != n["rank_first_usable"]
                   or b["usable_at_k"] != n["usable_at_k"])
        if not changed:
            continue
        n_changed += 1
        print(f"\nQ: {q}")
        print(f"  top1  {_fmt_delta(b['top1_score'], n['top1_score'])}"
              f"   {b['top1_source']} -> {n['top1_source']}")
        print(f"  first_usable {b['rank_first_usable']} -> {n['rank_first_usable']}"
              f" | usable@k {b['usable_at_k']} -> {n['usable_at_k']}"
              f" | kw_recall {b['kw_recall']:.2f} -> {n['kw_recall']:.2f}")
        print(f"  before: {b['topk_sources']}")
        print(f"  after : {n['topk_sources']}")
    if n_changed == 0:
        print("（无任何查询结果发生变化）")

    # 回退检测
    print("\n" + "-" * 78)
    print("回退检测（这些查询变差了，必须如实报告）")
    print("-" * 78)
    regress = []
    for q in [x["q"] for x in QUERIES]:
        b, n = bmap.get(q), nmap.get(q)
        if not b or not n:
            continue
        if (n["usable_at_k"] < b["usable_at_k"]
                or n["kw_recall"] < b["kw_recall"] - 1e-9
                or (b["rank_first_usable"] or 99) > (n["rank_first_usable"] or 99)):
            regress.append((q, b, n))
    if not regress:
        print("无回退。")
    for q, b, n in regress:
        print(f"  [!] {q}")
        print(f"      usable@k {b['usable_at_k']} -> {n['usable_at_k']}"
              f" | kw_recall {b['kw_recall']:.2f} -> {n['kw_recall']:.2f}"
              f" | first_usable {b['rank_first_usable']} -> {n['rank_first_usable']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run", help="本次结果的标签，如 before/after")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--baseline", help="对比基线 JSON 路径")
    ap.add_argument("--compare", action="store_true", help="与 --baseline 对比")
    ap.add_argument("--dump", action="store_true", help="打印每条查询 top-k 明细")
    ap.add_argument("--out-dir", default="data/results/rag")
    ap.add_argument("--config", default="configs/rag_config.yaml")
    args = ap.parse_args()

    from src.rag.vectorstore import get_vectorstore

    store = get_vectorstore(args.config)
    if store.index is None:
        if not store.load():
            print("[FATAL] 索引加载失败，先跑 scripts/build_index.py")
            return 1
    n_vec = store.index.ntotal
    if n_vec != len(store.documents):
        print(f"[FATAL] 索引/元数据不一致：index={n_vec} docs={len(store.documents)}")
        return 1
    print(f"索引已加载：{n_vec} 向量 / {len(store.documents)} 文档块")

    report = evaluate(store, args.top_k, QUERIES, dump=args.dump)
    report["tag"] = args.tag
    report["index_ntotal"] = n_vec
    report["n_documents"] = len(store.documents)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"retrieval_eval_{args.tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存：{out_path}")

    o = report["overall"]
    print("\n" + "=" * 78)
    print(f"汇总（n={report['n_queries']}, top_k={args.top_k}）")
    print("=" * 78)
    for k, v in o.items():
        print(f"  {k:<20} {v}")

    print("\n分类型：")
    for kind, agg in report["by_kind"].items():
        print(f"  {kind:<14} usable@k={agg['usable@k']:.2f} "
              f"src@1={agg['source_hit@1']:.2f} kw={agg['kw_recall']:.2f} "
              f"mrr={agg['mrr']:.2f} top1={agg['top1_score_mean']:.4f}")

    print("\ntop_k 名额来源分布：")
    for s, c in report["source_slot_share"].items():
        print(f"  {c:4d}  {s}")

    if args.compare and args.baseline:
        with open(args.baseline, "r", encoding="utf-8") as f:
            base = json.load(f)
        compare(base, report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
