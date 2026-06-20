"""Explanation API — 深度诊断详情接口"""
import os
import re
import sys
import time
from typing import Optional
from fastapi import APIRouter, Query

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.constants import CHANNEL_MAP
from src.utils.results_loader import load_json
_KB_DIRS = [
    os.path.join(PROJECT_ROOT, "docs", "knowledge_base"),
    os.path.join(PROJECT_ROOT, "docs", "papers"),
    os.path.join(PROJECT_ROOT, "docs"),
]

router = APIRouter()


def _resolve_kb_path(filename: str) -> Optional[str]:
    for d in _KB_DIRS:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    for d in _KB_DIRS:
        for root, _, files in os.walk(d):
            if filename in files:
                return os.path.join(root, filename)
    return None


def _parse_explanation(explanation: str) -> dict:
    if not explanation:
        return {}
    try:
        data = json.loads(explanation)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    sections = {}
    current_key = None
    current_lines = []

    KEYWORDS = {
        "通道定位": ["通道定位", "传感器定位", "通道说明", "异常通道分析", "异常通道定位", "异常通道物理含义分析", "通道标识"],
        "异常类型": ["异常类型", "异常分类", "异常类型判断", "异常类型分析", "异常类型与可能原因"],
        "可能原因": ["可能原因", "原因分析", "异常原因", "潜在原因", "异常原因推断", "异常原因分析", "基于知识的可能原因"],
        "影响评估": ["影响评估", "影响分析", "潜在影响", "影响范围", "补充说明", "局限性说明", "知识局限性", "知识充分性", "知识依据", "知识缺口", "信息充足"],
        "结论": ["结论", "诊断结论", "总结"],
        "建议措施": ["建议措施", "排查建议", "处理建议", "应对措施", "建议后续操作", "诊断总结与建议", "建议"],
        "紧急程度": ["紧急程度", "严重程度", "优先级"],
        "来源": ["来源", "知识来源", "参考来源"],
    }

    def _match_key(line: str) -> Optional[str]:
        ls = line.strip()
        clean = ls.replace('**', '')
        is_header = (
            ('【' in ls and '】' in ls) or
            re.match(r'\*\*\d+\.\s', ls) is not None or
            (re.match(r'^\*\*[^*]+\*\*\s*[:：]?', ls) is not None
             and not ls.startswith('- ') and not ls.startswith('* '))
        )
        if not is_header:
            return None
        for canonical, aliases in KEYWORDS.items():
            for alias in aliases:
                if alias in clean:
                    return canonical
        return None

    for line in explanation.split("\n"):
        matched = _match_key(line)
        if matched:
            if current_key and current_lines:
                filtered = [l for l in current_lines if l.strip() and not l.strip().startswith("#")]
                if filtered:
                    sections[current_key] = "\n".join(filtered).strip()
            current_key = matched
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)

    if current_key and current_lines:
        filtered = [l for l in current_lines if l.strip() and not l.strip().startswith("#")]
        if filtered:
            sections[current_key] = "\n".join(filtered).strip()

    return sections


def _is_garbled(text: str) -> bool:
    if not text:
        return False
    _errors = ["RAG解释生成失败", "检索失败", "生成失败", "RAG 查询失败", "查询过程中出现错误", "RAG引擎初始化失败"]
    for p in _errors:
        if p in text.strip():
            return True
    # 检测 Unicode 替换字符
    count = text.count('�') + text.count('�')
    if (count / max(len(text), 1)) > 0.05:
        return True
    # 检测 GBK 编码损坏的特征（连续的高位字符）
    garbled_chars = sum(1 for c in text if ord(c) > 0x7F and not ('一' <= c <= '鿿'))
    if len(text) > 10 and garbled_chars / len(text) > 0.3:
        return True
    return False


@router.get("/detail")
def get_detail(segment: str = Query(...), channel: str = Query(...)):
    """单个异常的完整诊断详情"""
    rag_data = load_json("anomaly_rag_results.json")
    if not rag_data:
        return {"error": "No RAG results available"}

    results = rag_data if isinstance(rag_data, list) else rag_data.get("results", [])

    # 查找匹配的异常条目
    item = None
    for r in results:
        if str(r.get("segment")) == segment and str(r.get("channel")) == channel:
            item = r
            break

    if not item:
        return {"error": f"Segment {segment} / Channel {channel} not found"}

    explanation = item.get("rag_explanation") or item.get("answer", "")
    garbled = _is_garbled(explanation)
    parsed = {} if garbled else _parse_explanation(explanation)

    ch_label = CHANNEL_MAP.get(channel, channel)
    score = item.get("anomaly_score", 0)

    urgency = parsed.get("紧急程度", "中")
    urgency_map = {"低": "low", "中": "medium", "高": "high", "紧急": "critical"}
    urgency_en = urgency_map.get(urgency, "medium")

    sections = {}
    if not garbled:
        section_order = ["可能原因", "影响评估", "结论", "建议措施", "来源"]
        for key in section_order:
            if key in parsed:
                sections[key] = parsed[key]

    sources = item.get("rag_sources") or item.get("sources", [])
    resolved_sources = []
    for src in sources:
        if isinstance(src, dict):
            meta = src.get("metadata", src)
            doc_name = meta.get("filename") or meta.get("source_name") or "未知文献"
            page = meta.get("page_label") or meta.get("page", "?")
            score_val = src.get("score", 0)
            file_path = _resolve_kb_path(doc_name)
            resolved_sources.append({
                "filename": doc_name,
                "page": page,
                "score": round(score_val, 4) if isinstance(score_val, float) else score_val,
                "local_path": file_path,
            })

    return {
        "segment": segment,
        "channel": channel,
        "channel_label": ch_label,
        "anomaly_type": item.get("anomaly_type", "未知"),
        "anomaly_score": round(score, 4),
        "urgency": urgency,
        "urgency_level": urgency_en,
        "sections": sections,
        "garbled": garbled,
        "feature_summary": item.get("feature_summary"),
        "sources": resolved_sources,
        "raw_explanation": explanation if not garbled else "",
    }


@router.get("/waveform")
def get_waveform(
    channel: str = Query(...),
    segment: Optional[str] = Query(None),
    context_size: int = Query(200),
):
    """遥测波形数据 — 异常段上下文"""
    import pandas as pd

    data_path = os.path.join(PROJECT_ROOT, "data", "raw", "segments.csv")
    if not os.path.exists(data_path):
        return {"error": "segments.csv not found"}

    df = pd.read_csv(data_path)
    ch_df = df[df["channel"] == channel].sort_values("timestamp").reset_index(drop=True)

    seg_indices = []
    if segment:
        try:
            seg_id = int(segment)
            if "segment" in ch_df.columns:
                seg_mask = ch_df["segment"] == seg_id
                seg_indices = ch_df.index[seg_mask].tolist()
        except (ValueError, TypeError):
            pass

    if not seg_indices:
        seg_indices = ch_df.index[ch_df["anomaly"] == 1].tolist()

    if seg_indices:
        center = seg_indices[len(seg_indices) // 2]
        start = max(0, center - context_size)
        end = min(len(ch_df), center + context_size)
        window_df = ch_df.iloc[start:end]
    else:
        window_df = ch_df.tail(500)

    mean_val = float(ch_df["value"].mean())
    std_val = float(ch_df["value"].std())

    timestamps = window_df["timestamp"].tolist()
    values = window_df["value"].tolist()
    anomaly_mask = window_df["anomaly"].tolist()

    return {
        "timestamps": timestamps,
        "values": values,
        "anomaly_mask": anomaly_mask,
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "upper_3sigma": round(mean_val + 3 * std_val, 4),
        "lower_3sigma": round(mean_val - 3 * std_val, 4),
        "channel": channel,
        "channel_label": CHANNEL_MAP.get(channel, channel),
    }


@router.post("/query")
def rag_query(body: dict):
    """RAG 自由问答"""
    query = body.get("query", "")
    channel = body.get("channel", "")
    segment = body.get("segment", "")
    context = body.get("context", {})

    if not query:
        return {"error": "Empty query"}

    try:
        from src.rag.pipeline import get_rag_pipeline
        pipeline = get_rag_pipeline()
        if hasattr(pipeline, 'vectorstore') and pipeline.vectorstore.index is None:
            pipeline.load_existing_knowledge_base()

        enriched = f"【当前异常上下文】\n通道: {channel}\n异常段: {segment}\n"
        if context:
            for k, v in context.items():
                enriched += f"{k}: {v}\n"
        enriched += f"\n【用户问题】\n{query}"

        result = pipeline.query(query=enriched, query_type="general", top_k=3)
        return {
            "answer": result.get("answer", "无法给出确切解答。"),
            "sources": [
                {"filename": s.get("metadata", s).get("filename", "未知"), "score": s.get("score", 0)}
                for s in result.get("sources", [])
            ],
        }
    except Exception as e:
        return {"error": str(e), "answer": "RAG 引擎暂不可用。"}
