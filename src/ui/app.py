import streamlit as st
import json
import os
import sys

# 加载 .env 到 os.environ（确保 RAG Pipeline 的 LLM API Key 可用）
from dotenv import load_dotenv
_ENV_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), ".env")
load_dotenv(_ENV_PATH, override=True)

st.set_page_config(
    page_title="OPS-SAT 遥测异常诊断系统",
    page_icon="🛰️",
    layout="wide",
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")

# ── 全局 CSS ──
st.markdown("""
<style>
.diagnosis-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
}
.section-divider {
    border: none;
    border-top: 1px solid #334155;
    margin: 0.8rem 0;
    opacity: 0.5;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem;
    color: #94a3b8;
}
[data-testid="stMetricValue"] {
    font-size: 1.05rem;
}
.source-item {
    background: #1e293b;
    border-left: 3px solid #3b82f6;
    border-radius: 4px;
    padding: 8px 12px;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.source-item .doc-name {
    font-size: 13px;
    color: #e2e8f0;
    font-weight: 500;
}
.source-item .doc-meta {
    font-size: 11px;
    color: #94a3b8;
    text-align: right;
    white-space: nowrap;
}
a .source-item:hover {
    background: #253348 !important;
    border-left-color: #60a5fa !important;
}
a .source-item {
    transition: background 0.2s, border-color 0.2s;
}
</style>
""", unsafe_allow_html=True)

# ── 1. 状态初始化 ──
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'list'
if 'target_anomaly_id' not in st.session_state:
    st.session_state.target_anomaly_id = None
if 'target_channel' not in st.session_state:
    st.session_state.target_channel = None
if 'target_rag_data' not in st.session_state:
    st.session_state.target_rag_data = None

# ── 2. 侧边栏 ──
with st.sidebar:
    st.markdown("### 🛰️ OPS-SAT 遥测异常诊断系统")
    if st.button("🚨 实时告警中心", use_container_width=True):
        st.session_state.view_mode = 'list'
        st.rerun()
    if st.button("⚙️ 算法与配置", use_container_width=True):
        st.session_state.view_mode = 'settings'
        st.rerun()

    st.divider()
    # 健康检查：遥测数据
    _seg_path = os.path.join(PROJECT_ROOT, "data", "raw", "segments.csv")
    if os.path.exists(_seg_path):
        st.success("🟢 遥测数据接口正常")
    else:
        st.error("🔴 遥测数据缺失")

    # 健康检查：FAISS 索引
    _faiss_path = os.path.join(PROJECT_ROOT, "data", "vectorstore", "faiss_index.bin")
    _rag_available = False
    try:
        from src.rag.pipeline import get_rag_pipeline  # noqa: F401
        _rag_available = True
    except Exception:
        pass

    if _rag_available and os.path.exists(_faiss_path):
        st.info("🔵 RAG诊断引擎在线")
    else:
        st.info("ℹ️ RAG 诊断引擎暂未部署（仅展示模式）")

# ── 3. 主页面路由逻辑 ──

# A. 设置页
if st.session_state.view_mode == 'settings':
    import pages.detection as detection
    if hasattr(detection, 'render_settings'):
        detection.render_settings()
    else:
        detection.render()

# B. 告警列表模式 (首页)
elif st.session_state.view_mode == 'list':
    import pages.dashboard as dashboard
    dashboard.render()

# C. 深度诊断模式 (详情页)
elif st.session_state.view_mode == 'detail':
    if st.session_state.target_rag_data is None and st.session_state.target_anomaly_id is not None:
        try:
            rag_path = os.path.join(RESULTS_DIR, "anomaly_rag_results.json")
            if os.path.exists(rag_path):
                with open(rag_path, "r", encoding="utf-8") as f:
                    rag_data = json.load(f)
                results = rag_data if isinstance(rag_data, list) else rag_data.get("results", [])
                target_id = str(st.session_state.target_anomaly_id)
                for item in results:
                    if str(item.get("segment")) == target_id:
                        st.session_state.target_rag_data = item
                        st.session_state.target_channel = item.get("channel", "Unknown")
                        break
        except Exception:
            pass

    from pages.explanation import render_detail
    render_detail()
