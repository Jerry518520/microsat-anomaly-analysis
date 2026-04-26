"""
可视化工具模块
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_anomaly_distribution(features_df, save_path=None):
    """绘制异常分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 按通道统计异常比例
    channel_stats = features_df.groupby("channel")["anomaly"].agg(["sum", "count"])
    channel_stats["ratio"] = channel_stats["sum"] / channel_stats["count"]
    channel_stats = channel_stats.sort_values("ratio", ascending=True)

    axes[0].barh(channel_stats.index, channel_stats["ratio"], color="#C00000")
    axes[0].set_xlabel("Anomaly Ratio")
    axes[0].set_title("Anomaly Ratio by Channel")
    for i, (idx, row) in enumerate(channel_stats.iterrows()):
        axes[0].text(row["ratio"] + 0.01, i, f"{row['ratio']:.1%}", va="center", fontsize=8)

    # 总体异常/正常分布
    counts = features_df["anomaly"].value_counts()
    colors = ["#2F5496", "#C00000"]
    labels = ["Normal", "Anomaly"]
    axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 12})
    axes[1].set_title("Overall Anomaly Distribution")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_segment_time_series(segments_df, segment_id, save_path=None):
    """绘制单个 segment 的时序图，标注异常"""
    seg_data = segments_df[segments_df["segment"] == segment_id]
    is_anomaly = seg_data["anomaly"].iloc[0]
    channel = seg_data["channel"].iloc[0]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(seg_data["timestamp"], seg_data["value"],
            color="#C00000" if is_anomaly else "#2F5496", linewidth=0.8)
    ax.set_title(f"Segment {segment_id} — Channel: {channel} — "
                 f"{'ANOMALY' if is_anomaly else 'NORMAL'}",
                 fontsize=13, color="red" if is_anomaly else "blue")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sliding_window_visual(segments_df, segment_id, window_size=50,
                                step_size=10, save_path=None):
    """可视化滑动窗口（用于 PPT 展示）"""
    seg_data = segments_df[segments_df["segment"] == segment_id]
    values = seg_data["value"].values
    is_anomaly = seg_data["anomaly"].iloc[0]

    fig, ax = plt.subplots(figsize=(14, 5))

    # 原始信号
    ax.plot(values, color="#2F5496", linewidth=1, label="Raw Signal", alpha=0.8)

    # 标注滑动窗口
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(range(0, len(values) - window_size + 1, step_size))))
    for i, start in enumerate(range(0, len(values) - window_size + 1, step_size)):
        ax.axvspan(start, start + window_size, alpha=0.15, color=colors[i])

    ax.set_title(f"Sliding Window Visualization — Segment {segment_id} — "
                 f"{'ANOMALY' if is_anomaly else 'NORMAL'} (window={window_size}, step={step_size})",
                 fontsize=13)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
