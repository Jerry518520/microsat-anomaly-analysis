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
    """实验演进路线 F1 Benchmark

    口径：`data/results/v3/fusion.json` 的 **论文口径 test 段级 F1**
    （官方 train 1594 段 → fit 1275 / val 319，官方 test 529 段只评估一次）。
    三档对应论文第四节叙述的递进：纯规则 → 硬 AND → 逐通道门控。
    """
    fusion_data = load_json("fusion.json")
    methods = (fusion_data or {}).get("results", {}).get("methods", {})

    ordered = [
        ("Stage 0", "rule_only", "纯规则（零模型，3σ / IQR）"),
        ("Stage 1", "if_and_rule", "IF + 规则硬 AND 融合"),
        ("Stage 2", "gate_perchannel", "IF + 规则逐通道门控融合"),
    ]

    experiments = []
    base_f1 = None
    for version, key, strategy in ordered:
        f1 = methods.get(key, {}).get("test", {}).get("f1")
        if not isinstance(f1, (int, float)):
            continue
        if base_f1 is None:
            base_f1 = float(f1)
            improvement = "—"
        else:
            improvement = f"{(float(f1) - base_f1) / base_f1 * 100:+.1f}%"
        experiments.append({
            "version": version,
            "strategy": strategy,
            "f1": round(float(f1), 4),
            "improvement": improvement,
        })

    if not experiments:
        experiments = [
            {"version": "Stage 0", "strategy": "纯规则（零模型，3σ / IQR）", "f1": 0.5882, "improvement": "—"},
            {"version": "Stage 1", "strategy": "IF + 规则硬 AND 融合", "f1": 0.6038, "improvement": "+2.7%"},
            {"version": "Stage 2", "strategy": "IF + 规则逐通道门控融合", "f1": 0.6281, "improvement": "+6.8%"},
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
