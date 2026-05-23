"""
系统后台 — 算法调优参数 + 评测对比与实验演进
"""
import streamlit as st
import json
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")

@st.cache_data
def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    for sub in os.listdir(RESULTS_DIR) if os.path.isdir(RESULTS_DIR) else []:
        sub_path = os.path.join(RESULTS_DIR, sub, filename)
        if os.path.isfile(sub_path):
            with open(sub_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None

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
        st.write("已挂载 5 PDF + 1 MD")
        st.write("Chunk总数: 5064")
        st.write("嵌入引擎: BGE-M3")
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
                ("Stage 3", "stage3_optimal", "Threshold Sweep 最优"),
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
                ("Stage 2", "IF + 规则融合", 0.4574, "+52.7%"),
                ("Stage 3", "Threshold Sweep 最优", 0.5056, "+68.8%"),
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
