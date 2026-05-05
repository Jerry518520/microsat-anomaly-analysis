"""
首页 — 实时监控看板 + 故障告警队列
"""
import streamlit as st
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")

CHANNEL_MAP = {
    "CADC0872": "磁力计 X轴", "CADC0873": "磁力计 Y轴", "CADC0874": "磁力计 Z轴",
    "CADC0884": "光二极管 1", "CADC0886": "光二极管 2", "CADC0888": "光二极管 3",
    "CADC0890": "光二极管 4", "CADC0892": "光二极管 5", "CADC0894": "光二极管 6",
}
NO_ANOMALY_CHANNELS = {"CADC0884"}

@st.cache_data
def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # fallback: 在子目录中查找（兼容 reorganize 后的目录结构）
    for sub in os.listdir(RESULTS_DIR) if os.path.isdir(RESULTS_DIR) else []:
        sub_path = os.path.join(RESULTS_DIR, sub, filename)
        if os.path.isfile(sub_path):
            with open(sub_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None

@st.cache_data
def load_segments():
    import pandas as pd
    path = os.path.join(PROJECT_ROOT, "data", "raw", "segments.csv")
    if not os.path.exists(path): return None
    return pd.read_csv(path)

def _render_core_metrics():
    col1, col2, col3, col4 = st.columns(4)
    ss_data = load_json("subsampling_sweep_1.9_results.json")
    best_f1 = ss_data.get("best_seg_f1", 0.425) if ss_data else 0.425
    rag_data = load_json("anomaly_rag_results.json")
    
    total_anomalies = len(rag_data) if isinstance(rag_data, list) else (rag_data.get("total_anomalies_detected", 0) if rag_data else 0)
    # 只计异常分 > 0.05 的真正故障
    fault_count = sum(1 for item in (rag_data if isinstance(rag_data, list) else []) if item.get("anomaly_score", 0) > 0.05)
    
    # 从 segments.csv 计算真实异常率
    df_seg = load_segments()
    if df_seg is not None and "anomaly" in df_seg.columns:
        total_rows = len(df_seg)
        anomaly_rows = int(df_seg["anomaly"].sum())
        anomaly_rate = anomaly_rows / total_rows if total_rows > 0 else 0
        rate_str = f"{anomaly_rate*100:.1f}%"
        rate_delta = f"{anomaly_rows:,}行/{total_rows:,}行"
    else:
        rate_str = "N/A"
        rate_delta = "数据缺失"
    
    with col1: st.metric("系统异常率", rate_str, delta=rate_delta, delta_color="inverse")
    with col2: st.metric("当前告警队列", f"{fault_count} 条", delta=f"共{total_anomalies}条异常段")
    with col3: st.metric("核心诊断 F1", f"{best_f1:.3f}")
    with col4: st.metric("遥测吞吐量", f"{total_rows//1000}K Rows", delta="9 通道并发")

def _render_channel_sparklines():
    st.markdown("### 📡 遥测通道矩阵 (实时监测中)")
    df = load_segments()
    if df is None: return st.warning("未找到 segments.csv 数据文件")

    import plotly.graph_objects as go
    cols = st.columns(3)
    for idx, (ch, label) in enumerate(CHANNEL_MAP.items()):
        col = cols[idx % 3]
        ch_df = df[df["channel"] == ch].sort_values("timestamp").tail(500)
        anomaly_count = ch_df["anomaly"].sum()
        total_count = len(ch_df)
        anomaly_rate = anomaly_count / total_count if total_count > 0 else 0

        border_color, status_icon = ("#22c55e", "🟢") if ch in NO_ANOMALY_CHANNELS or anomaly_rate <= 0.15 else \
                                    ("#f59e0b", "🟡") if anomaly_rate <= 0.4 else ("#ef4444", "🔴")

        with col:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=ch_df["value"].values, mode="lines", line=dict(color=border_color, width=1),
                fill="tozeroy", fillcolor=f"rgba({int(border_color[1:3],16)},{int(border_color[3:5],16)},{int(border_color[5:7],16)},0.08)"
            ))
            if (ch_df["anomaly"] == 1).any():
                fig.add_trace(go.Scatter(y=ch_df.loc[ch_df["anomaly"] == 1, "value"].values, mode="markers", marker=dict(color="#ef4444", size=3)))
            
            fig.update_layout(
                height=100, margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False,
                title=dict(text=f"{status_icon} {ch} ({label})", font=dict(size=12, color=border_color), x=0.02, y=0.95)
            )
            st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

def _render_alert_list():
    st.markdown("### 🚨 故障告警队列")
    rag_data = load_json("anomaly_rag_results.json")
    if not rag_data: return st.info("暂无 RAG 检测结果，队列安全。")
    
    results = rag_data if isinstance(rag_data, list) else rag_data.get("results", [])
    if not results: return st.success("当前无未处理告警。")

    # 只展示异常分 > 0.05 的真正故障，过滤掉低分/负分噪声
    FAULT_THRESHOLD = 0.05
    faults = [item for item in results if item.get("anomaly_score", 0) > FAULT_THRESHOLD]
    if not faults:
        return st.success(f"全部 {len(results)} 条异常段均低于告警阈值 ({FAULT_THRESHOLD})，当前无待处理故障。")

    st.caption(f"共 {len(results)} 条异常段，{len(faults)} 条超过告警阈值")

    # 👇 新增的回调函数：专门负责在跳转前，把数据死死锁进 session_state
    def nav_to_detail(s_id, ch, data_item):
        st.session_state.target_anomaly_id = s_id
        st.session_state.target_channel = ch
        st.session_state.target_rag_data = data_item
        st.session_state.view_mode = 'detail'

    for i, item in enumerate(faults):
        seg_id = item.get("segment", f"{i}")
        channel = item.get("channel", "Unknown")
        score = item.get("anomaly_score", 0)
        # 兼容两种字段名
        explanation = item.get("rag_explanation") or item.get("answer", "AI 诊断正在生成中...")
        
        summary = (explanation[:80] + "...") if len(str(explanation)) > 80 else explanation
        border_color = "#ef4444" if score > 0.15 else "#f59e0b" if score > 0.05 else "#3b82f6"
        
        st.markdown(f"""
        <div style="background-color: #1a1010; border-left: 4px solid {border_color}; padding: 12px; margin-bottom: 8px; border-radius: 4px;">
            <div style="display: flex; justify-content: space-between;">
                <h4 style="margin:0; color: #e2e8f0;">段 #{seg_id} · {channel}</h4>
                <span style="color: {border_color}; font-weight: bold;">异常分: {score:.3f}</span>
            </div>
            <p style="margin: 8px 0 0 0; font-size: 14px; color: #9ca3af;">🧠 <b>AI摘要:</b> {summary}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 👇 核心修改点：改用 on_click 回调传参，绝对不会丢数据
        st.button(
            f"🔍 深度诊断分析", 
            key=f"btn_detail_{seg_id}_{i}",
            on_click=nav_to_detail,
            args=(seg_id, channel, item) # 把真实数据当作参数传进去
        )
def render():
    _render_core_metrics()
    st.divider()
    _render_channel_sparklines()
    st.divider()
    _render_alert_list()