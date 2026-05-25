"""
详情页 — 单个异常的深度诊断中心
功能：波形放大可视化 + RAG 结构化报告 + 知识来源溯源 + 自由问答交互
"""
import streamlit as st
import json
import os
import sys
import re
import time
import pandas as pd
import plotly.graph_objects as go

# ── 路径与配置 ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")

# ── 文件路径映射（知识库文件 → 实际路径）──
_KB_DIRS = [
    os.path.join(PROJECT_ROOT, "docs", "knowledge_base"),
    os.path.join(PROJECT_ROOT, "docs", "papers"),
    os.path.join(PROJECT_ROOT, "docs"),
]


def _resolve_kb_path(filename: str) -> str | None:
    """根据文件名找到知识库文件的实际路径"""
    for d in _KB_DIRS:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    # 递归搜索
    for d in _KB_DIRS:
        for root, _, files in os.walk(d):
            if filename in files:
                return os.path.join(root, filename)
    return None


CHANNEL_MAP = {
    "CADC0872": "磁力计 X轴", "CADC0873": "磁力计 Y轴", "CADC0874": "磁力计 Z轴",
    "CADC0884": "光电二极管 1", "CADC0886": "光电二极管 2", "CADC0888": "光电二极管 3",
    "CADC0890": "光电二极管 4", "CADC0892": "光电二极管 5", "CADC0894": "光电二极管 6",
}
NO_ANOMALY_CHANNELS = {"CADC0884"}

# ── RAG Pipeline 延迟初始化 ──
_rag_pipeline = None
_rag_init_error = None

def _get_rag_pipeline():
    global _rag_pipeline, _rag_init_error
    if _rag_pipeline is not None:
        return _rag_pipeline
    if _rag_init_error is not None:
        return None
    try:
        from src.rag.pipeline import get_rag_pipeline
        _rag_pipeline = get_rag_pipeline()
        if hasattr(_rag_pipeline, 'vectorstore') and _rag_pipeline.vectorstore.index is None:
            _rag_pipeline.load_existing_knowledge_base()
        return _rag_pipeline
    except Exception as e:
        _rag_init_error = str(e)
        return None


@st.cache_data(ttl=60)
def _get_live_anomaly_score(segment_id: int, channel: str, progress_callback=None):
    """
    实时重算指定 segment 的异常分（基于当前数据与当前算法）。
    返回: (score, mode, timestamp)
    """
    from src.integration.anomaly_rag_pipeline import AnomalyRAGPipeline

    pipeline = AnomalyRAGPipeline()
    anomalies = pipeline.detect(progress_callback=progress_callback)
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    for a in anomalies:
        if int(a.segment) == int(segment_id) and str(a.channel) == str(channel):
            return float(a.anomaly_score), "recomputed", now_str

    # 若本次检测未判为异常，返回 0（表示该段当前未越过阈值）
    return 0.0, "recomputed", now_str


def _refresh_detection_results(max_explanations: int = 20, progress_callback=None):
    """
    一键重跑检测+RAG解释，并落盘覆盖 anomaly_rag_results.json。
    progress_callback(current, total, message) 用于更新进度条。
    """
    from dataclasses import asdict
    from src.integration.anomaly_rag_pipeline import AnomalyRAGPipeline

    pipeline = AnomalyRAGPipeline()
    results = pipeline.detect_and_explain(
        max_explanations=max_explanations,
        progress_callback=progress_callback
    )

    output_path = os.path.join(RESULTS_DIR, "anomaly_rag_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    # 刷新缓存，确保页面读取到新结果
    load_json.clear()
    _get_live_anomaly_score.clear()
    return output_path, len(results)


def _is_garbled_text(text: str) -> bool:
    """
    检测文本是否为乱码或无效错误信息。
    检查两类情况：
    1. UTF-8 内容被错误编码后的常见特征（替换字符、高位孤立字节）
    2. RAG 管道错误信息（不应作为 AI 诊断展示给用户）
    """
    if not text:
        return False

    # ---- 新增：检测 RAG 管道错误信息模式 ----
    _error_patterns = [
        "RAG解释生成失败",
        "检索失败",
        "生成失败",
        "RAG 查询失败",
        "查询过程中出现错误",
        "RAG引擎初始化失败",
        "抱歉，查询过程中出现错误",
    ]
    text_stripped = text.strip()
    for pattern in _error_patterns:
        if pattern in text_stripped:
            return True

    # ---- 原有逻辑：统计替换字符和常见乱码模式 ----
    replacement_count = text.count('\ufffd')
    # 统计连续的高位字节（乱码常见特征）
    garbled_patterns = text.count('��') + text.count('�')
    total_chars = len(text)
    if total_chars == 0:
        return False
    # 如果替换字符或乱码模式占比超过 5%，认为是乱码
    garbled_ratio = (replacement_count + garbled_patterns) / total_chars
    return garbled_ratio > 0.05


def _parse_explanation(explanation: str) -> dict:
    """
    解析 RAG explanation 为结构化字段。
    支持两种格式：
    1. JSON 格式：{"通道定位": "...", "异常类型判断": "...", ...}
    2. 文本格式：按固定标题分段
    """
    if not explanation:
        return {}

    # 尝试 JSON 解析
    try:
        data = json.loads(explanation)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # 文本分段解析
    sections = {}
    current_key = None
    current_lines = []

    # 常见标题映射（支持中英文、带/不带序号、【】包裹）
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

    def _match_key(line: str) -> str | None:
        line_stripped = line.strip()
        # 去掉 markdown 粗体标记再匹配
        clean_line = line_stripped.replace('**', '')
        # 只匹配 section 标题行（【xxx】、**N. xxx** 或独立 **xxx** 格式），跳过 bullet 子项
        is_section_header = (
            '【' in line_stripped and '】' in line_stripped
        ) or (
            re.match(r'\*\*\d+\.\s', line_stripped) is not None
        ) or (
            # 独立粗体行：**xxx** 后面跟 ：或 : 或换行，且不在 bullet 内
            re.match(r'^\*\*[^*]+\*\*\s*[:：]?', line_stripped) is not None
            and not line_stripped.startswith('- ')
            and not line_stripped.startswith('* ')
        )
        if not is_section_header:
            return None
        for canonical, aliases in KEYWORDS.items():
            for alias in aliases:
                if alias in clean_line:
                    return canonical
        return None

    for line in explanation.split("\n"):
        matched = _match_key(line)
        if matched:
            if current_key and current_lines:
                # 过滤掉只有空行或只有 markdown 语法的内容
                filtered = [l for l in current_lines if l.strip() and not l.strip().startswith("#")]
                if filtered:
                    sections[current_key] = "\n".join(filtered).strip()
            current_key = matched
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)

    # 处理末尾未闭合的 section
    if current_key and current_lines:
        filtered = [l for l in current_lines if l.strip() and not l.strip().startswith("#")]
        if filtered:
            sections[current_key] = "\n".join(filtered).strip()

    return sections


def _split_sections_raw(explanation: str) -> dict:
    """
    按 **N. xxx** 或 **xxx** 标题拆分原始 markdown，保留完整格式。
    返回 {"_header": "开头无标题内容", "通道定位": "### **1. 异常通道分析**\n...", ...}
    """
    if not explanation:
        return {}

    lines = explanation.split('\n')
    section_starts = []  # (line_index, canonical_key)

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

    for i, line in enumerate(lines):
        ls = line.strip()
        clean = ls.replace('**', '')
        is_header = (
            ('【' in ls and '】' in ls) or
            re.match(r'\*\*\d+\.\s', ls) is not None or
            (re.match(r'^\*\*[^*]+\*\*\s*[:：]?', ls) is not None
             and not ls.startswith('- ') and not ls.startswith('* '))
        )
        if not is_header:
            continue
        for canonical, aliases in KEYWORDS.items():
            for alias in aliases:
                if alias in clean:
                    section_starts.append((i, canonical))
                    break

    result = {}
    if section_starts and section_starts[0][0] > 0:
        header_content = '\n'.join(lines[:section_starts[0][0]]).strip()
        if header_content:
            result['_header'] = header_content

    for idx, (start, canonical) in enumerate(section_starts):
        end = section_starts[idx + 1][0] if idx + 1 < len(section_starts) else len(lines)
        result[canonical] = '\n'.join(lines[start:end]).strip()

    return result


def _extract_metric_text(raw_section: str) -> str:
    """从 section 原始 markdown 中提取关键指标文本（去掉标题行）"""
    if not raw_section:
        return ""
    lines = raw_section.split('\n')
    content_lines = []
    for line in lines:
        ls = line.strip()
        if not ls:
            continue
        # 跳过标题行（**N. xxx** 或 **xxx**）
        if re.match(r'^\*\*\d*\.?\s*[^*]+\*\*\s*[:：]?$', ls):
            continue
        content_lines.append(ls)
    return '\n'.join(content_lines)


def _render_structured_diagnosis(item: dict, channel: str):
    """渲染结构化 AI 诊断卡片 —— 信息层级清晰、md 原生渲染"""
    explanation = item.get("rag_explanation") or item.get("answer", "")

    # ── 乱码检测 ──
    if _is_garbled_text(explanation):
        st.warning("⚠️ 诊断结论编码异常（历史实验结果损坏），显示异常特征摘要作为替代。请点击页面底部「刷新检测结果」重新生成。")
        _render_feature_fallback(item, channel)
        return

    # ── 降级：无结构化内容时直接渲染原始 markdown ──
    if not explanation:
        st.caption("暂无 AI 诊断结论")
        _render_feature_fallback(item, channel)
        return

    raw_sections = _split_sections_raw(explanation)
    parsed = _parse_explanation(explanation)

    # ── 卡片容器 ──
    st.markdown('<div class="diagnosis-card">', unsafe_allow_html=True)

    # ── 标题行 ──
    ch_label = CHANNEL_MAP.get(channel, channel)
    anomaly_type = item.get("anomaly_type", "未知")
    st.markdown(f"**🔬 异常诊断** · {ch_label} · {anomaly_type}")

    # ── 关键指标行 ──
    score = item.get("anomaly_score", 0)
    score_color = "#ef4444" if score > 0.6 else "#f59e0b" if score > 0.3 else "#22c55e"
    urgency = parsed.get("紧急程度", "中")
    urgency_color = {"低": "#22c55e", "中": "#f59e0b", "高": "#ef4444", "紧急": "#dc2626"}.get(urgency, "#f59e0b")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("通道定位", parsed.get("通道定位", ch_label))
    m2.metric("异常类型", anomaly_type)
    m3.metric("异常分", f"{score:.3f}")
    m4.metric("紧急程度", urgency)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── 诊断内容：按 section 渲染，保留原始 markdown 格式 ──
    # 定义渲染顺序
    SECTION_ORDER = [
        ("可能原因", "🔴 可能原因"),
        ("影响评估", "⚡ 影响评估"),
        ("结论", "📋 结论"),
        ("建议措施", "✅ 建议措施"),
        ("来源", "📚 知识来源"),
    ]

    rendered_any = False
    for key, label in SECTION_ORDER:
        if key not in raw_sections:
            continue
        rendered_any = True
        section_md = raw_sections[key]
        st.markdown(f"**{label}**")
        content = _extract_metric_text(section_md)
        if content:
            st.markdown(content)
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # 如果没有匹配到任何 section，渲染整个 explanation
    if not rendered_any:
        st.markdown(explanation)

    # ── 统计特征（折叠）──
    feature = item.get("feature_summary", {})
    if feature:
        with st.expander("📊 统计特征细节"):
            f1, f2, f3 = st.columns(3)
            f1.metric("均值", f"{feature.get('mean', 0):.2e}")
            f2.metric("标准差", f"{feature.get('std', 0):.2e}")
            f3.metric("峰值因子", f"{feature.get('crest_factor', 0):.2f}")

    st.markdown('</div>', unsafe_allow_html=True)


def _render_feature_fallback(item: dict, channel: str):
    """当 RAG explanation 乱码时，显示异常特征摘要作为降级"""
    st.markdown('<div class="diagnosis-card">', unsafe_allow_html=True)

    ch_label = CHANNEL_MAP.get(channel, channel)
    atype = item.get("anomaly_type", "未知")
    score = item.get("anomaly_score", 0)

    st.markdown(f"**🔬 异常特征** · {ch_label} · {atype}")

    m1, m2, m3 = st.columns(3)
    m1.metric("通道定位", ch_label)
    m2.metric("异常类型", atype)
    m3.metric("异常分", f"{score:.3f}")

    feature = item.get("feature_summary", {})
    if feature:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("**📊 统计特征**")
        c1, c2, c3 = st.columns(3)
        c1.metric("均值", f"{feature.get('mean', 0):.2e}")
        c2.metric("标准差", f"{feature.get('std', 0):.2e}")
        c3.metric("峰值因子", f"{feature.get('crest_factor', 0):.2f}")
    else:
        st.info("暂无异常特征数据")

    st.markdown('</div>', unsafe_allow_html=True)


def render_detail():
    """渲染深度诊断详情页"""
    item = st.session_state.get('target_rag_data', {})
    seg_id = st.session_state.get('target_anomaly_id', 'Unknown')
    channel = st.session_state.get('target_channel', 'Unknown')

    # 顶部导航
    col_nav, col_title = st.columns([1, 5])
    with col_nav:
        if st.button("← 返回看板"):
            st.session_state.view_mode = 'list'
            st.rerun()
    with col_title:
        ch_label = CHANNEL_MAP.get(channel, channel)
        st.markdown(f"### 🔎 深度诊断 | 段 #{seg_id} · {ch_label}")

    # 实时重算 / 全量刷新控制区
    pipeline = _get_rag_pipeline()
    rag_online = pipeline is not None

    if not rag_online:
        if _rag_init_error:
            st.warning(f"⚠️ RAG 引擎未就绪：{_rag_init_error}")
        else:
            st.info("ℹ️ RAG 诊断引擎暂未部署，实时功能不可用。")

    c1, c2 = st.columns([1, 1])
    with c1:
        if rag_online:
            if st.button("🔄 实时重算异常分", use_container_width=True):
                try:
                    progress_bar = st.progress(0, text="正在初始化检测...")

                    def _on_progress(current, total, message):
                        progress_bar.progress(current / total, text=message)

                    seg_id_int = int(seg_id)
                    live_score, mode, ts = _get_live_anomaly_score(
                        seg_id_int, channel, progress_callback=_on_progress
                    )
                    progress_bar.progress(100, text=f"完成！异常分: {live_score:.3f}")
                    item["anomaly_score"] = float(live_score)
                    st.session_state.target_rag_data = item
                    st.success(f"已实时重算 ({mode})：{live_score:.3f} @ {ts}")
                    st.rerun()
                except Exception as e:
                    st.error(f"实时重算失败: {e}")
        else:
            st.button("🔄 实时重算异常分", use_container_width=True, disabled=True,
                       help="RAG 诊断引擎暂未部署")
    with c2:
        if rag_online:
            if st.button("🚀 刷新检测结果", use_container_width=True):
                try:
                    progress_bar = st.progress(0, text="正在初始化...")
                    status_text = st.empty()

                    def _on_progress(current, total, message):
                        progress_bar.progress(current / total, text=message)

                    out, n = _refresh_detection_results(
                        max_explanations=20,
                        progress_callback=_on_progress
                    )
                    progress_bar.progress(100, text=f"完成！共 {n} 条结果")
                    st.success(f"刷新完成：{n} 条结果，已写入 {out}")
                    st.rerun()
                except Exception as e:
                    st.error(f"刷新失败: {e}")
        else:
            st.button("🚀 刷新检测结果", use_container_width=True, disabled=True,
                       help="RAG 诊断引擎暂未部署")

    st.divider()

    # ── 1. 异常波形（聚焦异常段上下文）──
    st.markdown("#### 📡 遥测波形（异常段上下文）")
    data_path = os.path.join(PROJECT_ROOT, "data", "raw", "segments.csv")

    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            ch_df = df[df["channel"] == channel].sort_values("timestamp").reset_index(drop=True)

            # 定位异常段：用 segment 列匹配
            seg_id_int = None
            try:
                seg_id_int = int(seg_id)
            except (ValueError, TypeError):
                pass

            if seg_id_int is not None and "segment" in ch_df.columns:
                seg_mask = ch_df["segment"] == seg_id_int
                seg_indices = ch_df.index[seg_mask].tolist()
            else:
                # 降级：用 anomaly 列找异常点
                seg_indices = ch_df.index[ch_df["anomaly"] == 1].tolist()

            if seg_indices:
                # 取异常段前后各 200 个采样点作为上下文
                center = seg_indices[len(seg_indices) // 2]
                start = max(0, center - 200)
                end = min(len(ch_df), center + 200)
                window_df = ch_df.iloc[start:end]
            else:
                # 无异常标记，显示最近 500 点
                window_df = ch_df.tail(500)

            fig = go.Figure()

            # 正常数据
            fig.add_trace(go.Scatter(
                x=window_df["timestamp"], y=window_df["value"],
                mode="lines", name="遥测值",
                line=dict(color="#3b82f6", width=1.2)
            ))

            # 异常标记
            anom_df = window_df[window_df["anomaly"] == 1]
            if not anom_df.empty:
                fig.add_trace(go.Scatter(
                    x=anom_df["timestamp"], y=anom_df["value"],
                    mode="markers", name="异常点",
                    marker=dict(color="#ef4444", size=6, symbol="x")
                ))

            # 3σ 阈值
            mean_val = ch_df["value"].mean()
            std_val = ch_df["value"].std()
            fig.add_hline(y=mean_val + 3*std_val, line_dash="dash", line_color="#f59e0b", annotation_text="+3σ")
            fig.add_hline(y=mean_val - 3*std_val, line_dash="dash", line_color="#f59e0b", annotation_text="-3σ")

            fig.update_layout(
                height=320, margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="时间戳", yaxis_title="遥测值"
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        except Exception as e:
            st.warning(f"波形加载失败: {e}")
    else:
        st.error("segments.csv 未找到")

    st.markdown("<hr style='border:none;border-top:1px solid #334155;margin:1rem 0;'>", unsafe_allow_html=True)

    # ── 2. AI 诊断与知识溯源（左右分栏）──
    col_diag, col_source = st.columns([6, 4])

    with col_diag:
        st.markdown("#### 🤖 AI 智能诊断")
        _render_structured_diagnosis(item, channel)

    with col_source:
        st.markdown("#### 📚 知识溯源")
        sources = item.get("rag_sources") or item.get("sources", [])
        if sources:
            for idx, src in enumerate(sources):
                if isinstance(src, dict):
                    meta = src.get("metadata", src)
                    doc_name = meta.get("filename") or meta.get("source_name") or "未知文献"
                    page = meta.get("page_label") or meta.get("page", "?")
                    score = src.get("score", 0)
                    
                    # 尝试解析本地文件路径
                    file_path = _resolve_kb_path(doc_name)
                    
                    if file_path:
                        # 本地文件：生成 file:// 链接，PDF 带页码
                        file_url = "file:///" + file_path.replace("\\", "/")
                        if doc_name.lower().endswith(".pdf") and isinstance(page, int) and page > 0:
                            file_url += f"#page={page}"
                        st.markdown(f"""
                        <a href="{file_url}" target="_blank" style="text-decoration:none;">
                        <div class="source-item" style="cursor:pointer;transition:background 0.2s;">
                            <div>
                                <span class="doc-name">📄 {doc_name}</span>
                            </div>
                            <div class="doc-meta">p.{page} · {score:.3f} 🔗</div>
                        </div>
                        </a>
                        """, unsafe_allow_html=True)
                    else:
                        # 无法定位文件：纯展示
                        st.markdown(f"""
                        <div class="source-item">
                            <div>
                                <span class="doc-name">📄 {doc_name}</span>
                            </div>
                            <div class="doc-meta">p.{page} · {score:.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.caption("暂无关联文献")

    st.markdown("<hr style='border:none;border-top:1px solid #334155;margin:1rem 0;'>", unsafe_allow_html=True)

    # ── 3. 自由追问（接入 RAG Pipeline，精简 prompt）──
    st.markdown("#### 💬 异常追踪问答")

    if rag_online:
        user_query = st.chat_input("追问示例：对姿态控制的影响？")
        if user_query:
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.spinner("检索中..."):
                try:
                    enriched_query = f"""【当前异常上下文】
通道: {channel} ({CHANNEL_MAP.get(channel, channel)})
异常段: {seg_id}
异常类型: {item.get('anomaly_type', '未知')}
异常分: {item.get('anomaly_score', 0):.3f}

【用户问题】
{user_query}"""

                    result = pipeline.query(
                        query=enriched_query,
                        query_type="general",
                        top_k=3
                    )

                    with st.chat_message("assistant"):
                        answer = result.get("answer", "无法给出确切解答。")
                        st.markdown(answer)

                        if result.get("sources"):
                            with st.expander("参考来源"):
                                for s in result["sources"]:
                                    meta = s.get("metadata", s)
                                    fname = meta.get("filename", "未知文档")
                                    st.caption(f"· {fname}")
                except Exception as e:
                    st.error(f"查询出错: {e}")
    else:
        st.caption("🔒 RAG 诊断引擎暂未部署，问答功能暂不可用。部署后即可启用智能追问。")


@st.cache_data
def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # fallback: 子目录查找
    for sub in os.listdir(RESULTS_DIR) if os.path.isdir(RESULTS_DIR) else []:
        sub_path = os.path.join(RESULTS_DIR, sub, filename)
        if os.path.isfile(sub_path):
            with open(sub_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None
