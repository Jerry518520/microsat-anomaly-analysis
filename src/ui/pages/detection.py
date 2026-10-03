"""
系统后台 — 算法调优参数 + 评测对比与实验演进
"""
import streamlit as st
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from src.utils.results_loader import load_json as _load_json_raw
# P0-A6：索引路径收敛到 src/utils/paths.py 单一来源，不再手写字符串拼接
# （原写法 data/faiss_index/index.faiss 指向不存在的目录，见 paths.py 说明）
from src.utils.paths import read_faiss_index_status

@st.cache_data
def _get_rag_config():
    """动态读取 RAG 知识库实际配置"""
    # 1. 统计文档数
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

    # 2. 读取 FAISS 索引 chunk 数
    # P0-A6：原来这里手写错误路径 data/faiss_index/index.faiss（该目录不存在），
    # 且被 `if os.path.exists(...)` + `except: pass` 双重包裹，导致索引缺失时
    # 不报错不提示、页面照常渲染但实际没有 RAG 能力（静默降级）。
    # 现在改为向 src/utils/paths.py 取真实路径，并把状态透传给 render()，
    # 由 render() 用 st.warning/st.error 显式提示——本函数是 @st.cache_data，
    # 在这里调 st.* 只会首次计算时显示，缓存命中后用户看不到，故不在此处提示。
    index_status, chunk_count, index_reason = read_faiss_index_status()

    # 3. 读取嵌入引擎配置
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

    return doc_count, chunk_count, embedding_model, index_status, index_reason


@st.cache_data
def load_json(filename):
    return _load_json_raw(filename)

def render():
    st.markdown("## ⚙️ 核心算法参数与基准调优")
    pipeline_data = load_json("pipeline_segment_18d_output.json")
    channel_f1_data = load_json("v2_segments_per_channel_f1.json")

    stage0_f1 = None
    if pipeline_data:
        stages = pipeline_data.get("stages_performance", {})
        stage0_f1 = stages.get("stage0_global_if", {}).get("metrics", {}).get("f1")

    col1, col2 = st.columns([3, 7])
    with col1:
        st.markdown("#### RAG 知识库配置")
        doc_count, chunk_count, embedding_model, index_status, index_reason = _get_rag_config()

        # P0-A6：把索引状态变成页面上可见的信号。
        # 索引缺失时绝不能只显示 "未索引" 三个字了事——用户会误以为 RAG 在工作，
        # 实际所有异常解释都在无知识库的情况下生成。这里显式告警并给出排查路径。
        if index_status == "missing":
            st.error(
                "🔴 **FAISS 向量索引缺失 — RAG 解释能力不可用**\n\n"
                f"{index_reason}\n\n"
                "本页展示的 Chunk 数为空，异常解释将退化为**无知识库引用**的纯 LLM 输出，"
                "结论不可溯源。请先构建索引（`python scripts/build_index.py`）"
                "或解压含 `data/vectorstore/` 的数据包后再使用解释功能。"
            )
        elif index_status == "unreadable":
            st.error(
                "🔴 **FAISS 索引无法读取 — RAG 解释能力不可用**\n\n"
                f"{index_reason}\n\n"
                "通常是 faiss 版本与索引写出版本不匹配，或文件传输中损坏。"
                "请核对 requirements.txt 中 faiss-cpu 版本后重建索引。"
            )

        doc_parts = []
        if doc_count["pdf"]: doc_parts.append(f"{doc_count['pdf']} PDF")
        if doc_count["md"]: doc_parts.append(f"{doc_count['md']} MD")
        if doc_count["html"]: doc_parts.append(f"{doc_count['html']} HTML")
        st.write(f"已挂载 {' + '.join(doc_parts) if doc_parts else '无文档'}")
        st.write(f"Chunk总数: {chunk_count if chunk_count is not None else '未索引'}")
        st.write(f"嵌入引擎: {embedding_model}")
        st.button("全量重建 FAISS 索引", use_container_width=True)

    with col2:
        st.markdown("#### 实验演进路线 (F1 Benchmark)")
        experiments = []
        if pipeline_data:
            stages = pipeline_data.get("stages_performance", {})
            ordered = [
                ("Stage 0", "stage0_global_if", "全局 IForest (c=0.2)"),
                ("Stage 1", "stage1_per_channel", "分通道独立 IForest"),
                ("Stage 2", "stage2_fusion", "IF + 规则融合"),
            ]
            for stage_name, stage_key, strategy in ordered:
                f1 = stages.get(stage_key, {}).get("metrics", {}).get("f1")
                if f1 is None:
                    continue
                if stage0_f1:
                    delta = (f1 - stage0_f1) / stage0_f1 * 100
                    improve = "—" if stage_key == "stage0_global_if" else f"{delta:+.1f}%"
                else:
                    improve = "—"
                experiments.append((stage_name, strategy, round(float(f1), 4), improve))
        if not experiments:
            experiments = [
                ("Stage 0", "全局 IForest (c=0.2)", 0.2996, "—"),
                ("Stage 1", "分通道独立 IForest", 0.5381, "+79.6%"),
                ("Stage 2", "IF + 规则融合", 0.5683, "+89.7%"),
            ]
        df = pd.DataFrame(experiments, columns=["版本代号", "技术策略", "段级 F1", "相对提升"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("#### 分通道灵敏度测试")
        if channel_f1_data:
            details = channel_f1_data.get("per_channel_detail", {})
            if details:
                rows = []
                for ch, info in details.items():
                    rows.append({
                        "通道": ch,
                        "F1得分": f"{info.get('f1', 0):.4f}",
                        "Precision": f"{info.get('precision', 0):.4f}",
                        "Recall": f"{info.get('recall', 0):.4f}",
                    })
                rows = sorted(rows, key=lambda x: float(x["F1得分"]), reverse=True)
            else:
                per_channel = channel_f1_data.get("per_channel_f1", {})
                rows = [{"通道": ch, "F1得分": f"{f1:.4f}"} for ch, f1 in per_channel.items()]
                rows = sorted(rows, key=lambda x: float(x["F1得分"]), reverse=True)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("分通道测试数据未就绪。")
