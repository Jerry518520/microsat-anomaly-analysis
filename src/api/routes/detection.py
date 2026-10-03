"""Detection API — 算法配置 + 实验数据接口"""
import os
import sys
from fastapi import APIRouter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.results_loader import load_json
# P0-A6：索引路径收敛到 src/utils/paths.py，避免各处手写漂移
from src.utils.paths import read_faiss_index_status

router = APIRouter()


@router.get("/rag-config")
def get_rag_config():
    """RAG 知识库配置"""
    doc_dirs = [
        os.path.join(PROJECT_ROOT, "docs", "knowledge_base"),
        os.path.join(PROJECT_ROOT, "docs", "papers"),
    ]
    doc_count = {"pdf": 0, "md": 0, "html": 0}
    for d in doc_dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            ext = f.lower().rsplit(".", 1)[-1] if "." in f else ""
            if ext in doc_count:
                doc_count[ext] += 1

    # P0-A6：改用 paths.read_faiss_index_status()，与 UI 侧同一来源；
    # 同时把状态与原因一并返回，不再"读取失败就当 chunk 数为 None"静默吞掉。
    index_status, chunk_count, index_reason = read_faiss_index_status()

    embedding_model = "BGE-M3"
    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            embedding_model = cfg.get("rag", {}).get("embedding_model", embedding_model)
        except Exception:
            pass

    return {
        "status": "online" if index_status == "ok" else "offline",
        # P0-A6：新增字段，接口消费方（如前端）可据此显示显式错误，
        # 不必再靠 chunk_count is None 反推"是否降级"
        "index_status": index_status,
        "index_reason": index_reason,
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "embedding_model": embedding_model,
    }


@router.get("/experiments")
def get_experiments():
    """实验演进路线 F1 Benchmark"""
    pipeline_data = load_json("pipeline_segment_18d_output.json")

    stage0_f1 = None
    stages = {}
    if pipeline_data:
        stages = pipeline_data.get("stages_performance", {})
        stage0_f1 = stages.get("stage0_global_if", {}).get("metrics", {}).get("f1")

    ordered = [
        ("Stage 0", "stage0_global_if", "全局 IForest (c=0.2)"),
        ("Stage 1", "stage1_per_channel", "分通道独立 IForest"),
        ("Stage 2", "stage2_fusion", "IF + 规则融合"),
    ]

    experiments = []
    for version, key, strategy in ordered:
        f1 = stages.get(key, {}).get("metrics", {}).get("f1")
        if f1 is None:
            continue
        if stage0_f1 and key != "stage0_global_if":
            delta = (f1 - stage0_f1) / stage0_f1 * 100
            improvement = f"{delta:+.1f}%"
        else:
            improvement = "—"
        experiments.append({
            "version": version,
            "strategy": strategy,
            "f1": round(float(f1), 4),
            "improvement": improvement,
        })

    if not experiments:
        experiments = [
            {"version": "Stage 0", "strategy": "全局 IForest (c=0.2)", "f1": 0.2996, "improvement": "—"},
            {"version": "Stage 1", "strategy": "分通道独立 IForest", "f1": 0.5381, "improvement": "+79.6%"},
            {"version": "Stage 2", "strategy": "IF + 规则融合", "f1": 0.5683, "improvement": "+89.7%"},
        ]

    return {"experiments": experiments}


@router.get("/channel-f1")
def get_channel_f1():
    """分通道灵敏度测试 F1/Precision/Recall"""
    data = load_json("v2_segments_per_channel_f1.json")
    if not data:
        return {"channels": [], "available": False}

    details = data.get("per_channel_detail", {})
    if details:
        rows = []
        for ch, info in details.items():
            rows.append({
                "channel": ch,
                "f1": round(info.get("f1", 0), 4),
                "precision": round(info.get("precision", 0), 4),
                "recall": round(info.get("recall", 0), 4),
            })
        rows.sort(key=lambda x: x["f1"], reverse=True)
        return {"channels": rows, "available": True}

    per_channel = data.get("per_channel_f1", {})
    if per_channel:
        rows = [{"channel": ch, "f1": round(f1, 4), "precision": None, "recall": None}
                for ch, f1 in per_channel.items()]
        rows.sort(key=lambda x: x["f1"], reverse=True)
        return {"channels": rows, "available": True}

    # 原始格式:顶层即 {channel: {f1, n_test, n_anomaly, strategy}, ...}
    if all(isinstance(v, dict) and "f1" in v for v in data.values()) and data:
        rows = [{"channel": ch, "f1": round(info.get("f1", 0), 4),
                 "precision": info.get("precision"), "recall": info.get("recall")}
                for ch, info in data.items()]
        rows.sort(key=lambda x: x["f1"], reverse=True)
        return {"channels": rows, "available": True}

    return {"channels": [], "available": False}


@router.get("/system-params")
def get_system_params():
    """算法系统参数"""
    return {
        "algorithm": "Isolation Forest",
        "dimensions": 18,
        "contamination": 0.2,
    }
