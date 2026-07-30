"""Dashboard API — 实时告警中心数据接口"""
import json
import os
import sys
import pandas as pd
from fastapi import APIRouter
from starlette.responses import Response

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.constants import CHANNEL_MAP, NO_ANOMALY_CHANNELS, FAULT_THRESHOLD
from src.utils.results_loader import load_json

router = APIRouter()


def _load_segments():
    path = os.path.join(PROJECT_ROOT, "data", "raw", "segments.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _severity(score: float, channel: str = "") -> str:
    if channel in NO_ANOMALY_CHANNELS:
        return "nominal"
    if score > 0.4:
        return "critical"
    if score > 0.15:
        return "warning"
    if score > FAULT_THRESHOLD:
        return "caution"
    return "nominal"


@router.get("/metrics")
def get_core_metrics():
    """核心指标: 异常率、告警数、F1、吞吐量"""
    rag_data = load_json("anomaly_rag_results.json")
    ss_data = load_json("subsampling_sweep_1.9_results.json")
    df_seg = _load_segments()

    best_f1 = ss_data.get("best_seg_f1", 0.425) if ss_data else 0.425

    total_anomalies = 0
    fault_count = 0
    if isinstance(rag_data, list):
        total_anomalies = len(rag_data)
        fault_count = sum(1 for item in rag_data if item.get("anomaly_score", 0) > FAULT_THRESHOLD)
    elif isinstance(rag_data, dict):
        total_anomalies = rag_data.get("total_anomalies_detected", 0)

    total_rows = 0
    anomaly_rows = 0
    anomaly_rate = 0.0
    if df_seg is not None and "anomaly" in df_seg.columns:
        total_rows = len(df_seg)
        anomaly_rows = int(df_seg["anomaly"].sum())
        anomaly_rate = anomaly_rows / total_rows if total_rows > 0 else 0

    return {
        "anomaly_rate": round(anomaly_rate, 4),
        "anomaly_rate_str": f"{anomaly_rate * 100:.1f}%",
        "anomaly_rows": anomaly_rows,
        "total_rows": total_rows,
        "fault_count": fault_count,
        "total_anomalies": total_anomalies,
        "best_f1": round(best_f1, 4),
        "throughput": total_rows,
        "channel_count": 9,
    }


@router.get("/channels")
def get_channels():
    """9 个遥测通道的 sparkline 数据 + 状态"""
    df_seg = _load_segments()
    if df_seg is None:
        return {"channels": [], "error": "segments.csv not found"}

    channels = []
    for ch, label in CHANNEL_MAP.items():
        ch_df = df_seg[df_seg["channel"] == ch].sort_values("timestamp").tail(500)
        if ch_df.empty:
            channels.append({
                "channel": ch, "label": label, "anomaly_rate": 0,
                "status": "offline", "sparkline_values": [], "anomaly_indices": [],
            })
            continue

        anomaly_count = int(ch_df["anomaly"].sum())
        total_count = len(ch_df)
        rate = anomaly_count / total_count if total_count > 0 else 0

        if ch in NO_ANOMALY_CHANNELS:
            status = "nominal"
        elif rate > 0.4:
            status = "critical"
        elif rate > 0.15:
            status = "warning"
        elif rate > 0:
            status = "caution"
        else:
            status = "nominal"

        # 转换为 Python 原生类型，避免 numpy 类型导致 JSON 序列化失败
        values = [float(v) for v in ch_df["value"].tolist()]
        anom_idx = ch_df.index[ch_df["anomaly"] == 1].tolist()
        # 转换为相对于窗口的索引
        start_idx = int(ch_df.index[0])
        anom_idx = [int(i - start_idx) for i in anom_idx]

        channels.append({
            "channel": ch, "label": label,
            "anomaly_rate": round(float(rate), 4),
            "status": status,
            "sparkline_values": values,
            "anomaly_indices": anom_idx,
        })

    # 直接用 starlette Response 返回，完全绕过 FastAPI 编码器
    return Response(
        content=json.dumps({"channels": channels}),
        media_type="application/json"
    )


@router.get("/alerts")
def get_alerts():
    """故障告警队列 — 异常分 > 0.05 的条目"""
    rag_data = load_json("anomaly_rag_results.json")
    if not rag_data:
        return {"alerts": [], "total": 0}

    results = rag_data if isinstance(rag_data, list) else rag_data.get("results", [])
    faults = []
    for item in results:
        score = item.get("anomaly_score", 0)
        if score <= FAULT_THRESHOLD:
            continue
        ch = item.get("channel", "Unknown")
        explanation = item.get("rag_explanation") or item.get("answer", "AI 诊断正在生成中...")
        explanation = str(explanation) if not isinstance(explanation, str) else explanation
        summary = explanation[:120] + "..." if len(explanation) > 120 else explanation

        faults.append({
            "segment": str(item.get("segment", "")),
            "channel": ch,
            "channel_label": CHANNEL_MAP.get(ch, ch),
            "anomaly_score": round(score, 4),
            "severity": _severity(score, ch),
            "summary": summary,
            "anomaly_type": item.get("anomaly_type", "未知"),
        })

    faults.sort(key=lambda x: x["anomaly_score"], reverse=True)
    return {"alerts": faults, "total": len(faults)}


@router.get("/status")
def get_system_status():
    """系统健康状态"""
    segments_path = os.path.join(PROJECT_ROOT, "data", "raw", "segments.csv")
    faiss_path = os.path.join(PROJECT_ROOT, "data", "faiss_index", "index.faiss")
    rag_error = None
    rag_available = os.path.exists(faiss_path)

    return {
        "segments_available": os.path.exists(segments_path),
        "rag_available": rag_available,
        "rag_error": rag_error,
    }
