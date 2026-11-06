"""Detection API — 算法配置 + 实验数据接口"""
import os
import sys
from fastapi import APIRouter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.results_loader import load_json
from src.api.result_data import confirmed_results
from src.detection.frozen import DetectorUnavailable, load_bundle, load_recipe
from fastapi import HTTPException
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
    """Uniform comparisons and the separate engineering reference."""
    metrics, source = confirmed_results()
    names = {"global_if": "全局 IF", "channel_if": "分通道 IF", "rule": "纯规则",
             "AND": "AND", "OR": "OR", "gate": "逐通道门控", "soft": "软融合",
             "engineering_gate": "当前工程冻结方案"}
    base = metrics["overall"]["rule"]["f1"]
    rows = []
    for method, label in names.items():
        value = metrics["overall"][method]["f1"]
        rows.append({"version": method, "strategy": label, "f1": round(value, 4),
                     "improvement": f"{(value - base) / base * 100:+.1f}%" if base else "—",
                     "comparison_group": "engineering_reference" if method == "engineering_gate" else "uniform"})
    return {"experiments": rows, "source": source}


@router.get("/channel-f1")
def get_channel_f1():
    """All nine channels of the same engineering result package."""
    metrics, source = confirmed_results()
    rows = [{"channel": channel, "f1": round(item["f1"], 4),
             "precision": round(item["precision"], 4), "recall": round(item["recall"], 4),
             "n_test": item["n"], "positives": item["positives"], "small_sample": item["small_sample"],
             "confusion": item["confusion"]}
            for channel, item in metrics["per_channel"]["engineering_gate"].items()]
    return {"channels": sorted(rows, key=lambda row: row["f1"], reverse=True), "available": True, "source": source}


@router.get("/system-params")
def get_system_params():
    """Expose the per-channel engineering recipe without a false global c."""
    _, source = confirmed_results()
    try:
        judges = load_bundle()
        if next(iter(judges.values())).bundle_identity.manifest_sha256 != source["bundle_id"]:
            raise DetectorUnavailable("current detector and result package differ")
        recipe = load_recipe()
    except DetectorUnavailable as exc:
        raise HTTPException(status_code=503, detail={"error": "detector_result_mismatch", "message": str(exc)}) from exc
    return {"algorithm": "统计规则与 Isolation Forest 门控", "dimensions": 18,
            "contamination": None, "contamination_summary": "按通道固定参数",
            "contamination_by_channel": recipe["contamination"], "gate_choice": recipe["gate_choice"],
            "rule_k": recipe["best_k"], "isolation_forest": recipe["isolation_forest"], "source": source}
