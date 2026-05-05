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
    
    col1, col2 = st.columns([3, 7])
    with col1:
        st.markdown("#### 控制台热更新指令")
        ws = st.slider("时间窗口 (Window Size)", 5, 100, 20, step=5)
        threshold = st.slider("段级投票阈值", 0.1, 0.9, 0.25, step=0.05)
        psi = st.selectbox("IForest 子采样 (Psi)", [64, 128, 256, 512], index=1)
        st.button("重载管道参数 (Hot Reload)", type="primary", use_container_width=True)
        
        st.divider()
        st.markdown("#### RAG 知识库配置")
        st.write("已挂载 5 PDF + 1 MD")
        st.write("Chunk总数: 5064")
        st.write("嵌入引擎: BGE-M3")
        st.button("全量重建 FAISS 索引", use_container_width=True)

    with col2:
        st.markdown("#### 实验演进路线 (F1 Benchmark)")
        experiments = [
            ("Baseline", "隔离森林全局检测", 0.262, "—"),
            ("调参优化", "Contamination=0.5", 0.268, "+2%"),
            ("兜底熔断", "强 IF + 规则 OR", 0.325, "+24%"),
            ("动态加权", "方案 G", 0.342, "+31%"),
            ("阈值截断", "Threshold=0.25", 0.418, "+59%"),
            ("拓扑降噪", "Psi=128", 0.425, "+62%"),
        ]
        df = pd.DataFrame(experiments, columns=["版本代号", "技术策略", "段级 F1", "相对提升"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("#### 分通道灵敏度测试")
        cs_data = load_json("contamination_search_ws20.json")
        if cs_data:
            pcb = cs_data.get("per_channel_best", {})
            rows = [{"通道": ch, "F1得分": f"{info.get('best_f1', 0):.3f}", "最优 Contamination": info.get('best_contamination', 'N/A')} for ch, info in pcb.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("分通道测试数据未就绪。")