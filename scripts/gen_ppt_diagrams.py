#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成开题PPT三张图: 架构图、技术路线图、甘特图"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Font config
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = r'F:\微小卫星项目\microsat-anomaly-analysis\data\results'
DPI = 200

# Color palette
C = {
    'data': '#3498DB',
    'feat': '#2ECC71',
    'detect': '#9B59B6',
    'rag': '#E67E22',
    'show': '#E74C3C',
    'arrow': '#34495E',
    'text': '#2C3E50',
    'gantt': '#1ABC9C',
    'paper': '#34495E',
}


def draw_box(ax, x, y, w, h, color, title, details, title_size=12, detail_size=9):
    """Rounded box with colored header + detail text"""
    body = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                           facecolor=color, alpha=0.08, edgecolor=color, linewidth=1.8)
    ax.add_patch(body)
    hdr_h = 0.35
    hdr = FancyBboxPatch((x, y + h - hdr_h), w, hdr_h, boxstyle="round,pad=0.05",
                           facecolor=color, alpha=0.85, edgecolor=color, linewidth=1)
    ax.add_patch(hdr)
    ax.text(x + w/2, y + h - hdr_h/2, title, ha='center', va='center',
            fontsize=title_size, fontweight='bold', color='white')
    ax.text(x + w/2, y + (h - hdr_h)/2, details, ha='center', va='center',
            fontsize=detail_size, color=C['text'], linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color=None, style='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color or C['arrow'], lw=2.2, linestyle=style))


# ===================== 1. 架构图 =====================
def gen_architecture():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(7, 9.6, '系统架构图', ha='center', fontsize=20, fontweight='bold', color=C['text'])

    layers = [
        (8.2, C['data'],   '① 数据层',
         'ESA OPS-SAT 遥测数据'),
        (6.4, C['feat'],   '② 特征工程层',
         '段级18维特征提取  ·  分通道独立处理  ·  参数精细搜索'),
        (4.6, C['detect'], '③ 异常检测层',
         'Isolation Forest 分通道建模  ·  多策略融合'),
        (2.8, C['rag'],    '④ RAG智能解释层',
         '向量检索 + 多源知识库  ·  LLM 生成结构化解释报告'),
        (1.0, C['show'],   '⑤ 展示层',
         'Streamlit 交互界面  ·  异常标注 + 智能解释'),
    ]

    bw, bh = 12, 1.35
    bx = (14 - bw) / 2

    for y, color, title, details in layers:
        draw_box(ax, bx, y - bh/2, bw, bh, color, title, details, detail_size=9.5)

    for i in range(len(layers) - 1):
        y_from = layers[i][0] - bh/2
        y_to = layers[i+1][0] + bh/2
        arrow(ax, 7, y_from, 7, y_to)

    plt.tight_layout()
    plt.savefig(f'{OUT}/架构图_系统架构.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print('[OK] Architecture diagram saved')


# ===================== 2. 技术路线图 =====================
def gen_tech_route():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(8, 9.6, '技术路线图', ha='center', fontsize=20, fontweight='bold', color=C['text'])

    # Row 1: Data input
    draw_box(ax, 4, 8.0, 8, 1.0, C['data'], '原始遥测数据输入',
             'ESA OPS-SAT')

    # Row 2: Two parallel tracks
    draw_box(ax, 1.5, 5.8, 5, 1.2, C['feat'], '数据预处理与特征工程',
             '分通道切割 + 异常标注\n段级18维特征提取\n参数精细搜索')
    draw_box(ax, 9, 5.8, 5.5, 1.2, C['rag'], '知识库构建',
             '多源文档解析\n多语言向量化嵌入\nFAISS向量索引\n多源去重检索')

    # Row 3: Core algorithms
    draw_box(ax, 1.5, 3.2, 5, 1.5, C['detect'], '异常检测算法',
             'Isolation Forest 分通道建模\n多策略融合\n逐通道门控选算子\n强/弱通道差异化处理')
    draw_box(ax, 9, 3.2, 5.5, 1.5, C['rag'], 'RAG智能解释生成',
             '异常段 → 检索词具体化\n向量检索 + 多源去重\nLLM 结构化生成\n输出: 原因分析 + 知识来源\n全链路自动解释')

    # Row 4: Integration
    draw_box(ax, 4, 0.8, 8, 1.0, C['show'], '系统集成与展示',
             'Streamlit交互界面  |  总览仪表盘 · 异常检测分析 · RAG智能解释  |  异常标注 + 解释报告')

    # Arrows Row1 → Row2
    arrow(ax, 6, 8.0, 4, 7.0)
    arrow(ax, 10, 8.0, 11.75, 7.0)

    # Arrows Row2 → Row3
    arrow(ax, 4, 5.8, 4, 4.7)
    arrow(ax, 11.75, 5.8, 11.75, 4.7)

    # Arrows Row3 → Row4
    arrow(ax, 4, 3.2, 6, 1.8)
    arrow(ax, 11.75, 3.2, 10, 1.8)

    # Cross-link: detect → RAG
    arrow(ax, 6.5, 3.95, 9, 3.95, color='#E67E22', style='--')
    ax.text(7.75, 4.2, '异常段→检索', ha='center', fontsize=9, color='#E67E22', style='italic')



    plt.tight_layout()
    plt.savefig(f'{OUT}/技术路线图_流程.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print('[OK] Tech route diagram saved')


# ===================== 3. 甘特图 =====================
def gen_gantt():
    fig, ax = plt.subplots(figsize=(18, 7))

    # 项目周期: 2026.4 — 2027.12 (21个月)
    # 月0 = 2026.4, 月20 = 2027.12
    phases = [
        ('需求分析与方案设计',       0,  2, C['data']),    # 4-5月
        ('数据预处理与特征工程',     1,  3, C['feat']),    # 5-7月
        ('异常检测算法开发与优化',   3,  4, C['detect']),  # 7-10月
        ('RAG解释系统开发',         5,  4, C['rag']),     # 9-12月
        ('系统集成与测试',          8,  3, C['gantt']),   # 12-2月
        ('论文与报告撰写',          10, 5, C['paper']),   # 2-6月
        ('答辩准备与结题',          16, 5, C['show']),    # 8-12月
    ]

    # 里程碑 (按实际截止日期)
    milestones = [
        (1,   '开题答辩\n5.15截止'),
        (11,  '中期检查\n3.5开放'),
        (20,  '结题\n12.31截止'),
    ]

    months = ['4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月',
              '1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']

    for i, (name, start, dur, color) in enumerate(phases):
        ax.barh(i, dur, left=start, height=0.6, color=color, alpha=0.8,
                edgecolor='white', linewidth=1.5, zorder=2)
        text_x = start + dur / 2
        ax.text(text_x, i, name, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color='white', zorder=3)

    # Current progress marker (2026.5 = month 1)
    ax.axvline(x=1, color=C['show'], linestyle='-', linewidth=2.5, alpha=0.7, zorder=1)
    ax.text(1, -0.7, '当前', ha='center', fontsize=10,
            color=C['show'], fontweight='bold')

    # Milestones — 放在条形图下方
    for m_x, m_label in milestones:
        ax.plot(m_x, len(phases) + 0.5, marker='^', color=C['show'], markersize=12, zorder=5)
        ax.text(m_x, len(phases) + 1.0, m_label, ha='center', va='top', fontsize=8.5,
                color=C['show'], fontweight='bold')

    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels([p[0] for p in phases], fontsize=11)
    ax.set_xticks(np.arange(0, 21))
    ax.set_xticklabels(months, fontsize=8.5)
    ax.set_xlim(-0.5, 21)
    ax.set_ylim(-0.5, len(phases) + 3.0)
    ax.invert_yaxis()
    ax.set_title('项目进度甘特图 (2026.4 — 2027.12)', fontsize=18, fontweight='bold',
                 color=C['text'], pad=25)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Year dividers
    ax.axvline(x=9, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=21, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(4.5, len(phases) + 2.7, '2026年', ha='center', fontsize=10, color='gray')
    ax.text(15, len(phases) + 2.7, '2027年', ha='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(f'{OUT}/甘特图_项目进度.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print('[OK] Gantt chart saved')


if __name__ == '__main__':
    gen_architecture()
    gen_tech_route()
    gen_gantt()
    print('\n[DONE] All 3 diagrams generated!')
