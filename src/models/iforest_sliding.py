"""
IForest + 滑动窗口特征模型
核心实验：滑动窗口特征 vs 段级特征对比
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score, classification_report
)
from src.utils.data_loader import load_config, get_project_root
from src.features.sliding_window import run_sliding_window_pipeline


def get_sw_train_test(features_df):
    """从滑动窗口特征中划分训练/测试集"""
    meta_cols = ["channel", "segment", "anomaly", "train", "sampling"]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    train_mask = features_df["train"] == 1
    X_train = features_df.loc[train_mask, feature_cols]
    y_train = features_df.loc[train_mask, "anomaly"]
    X_test = features_df.loc[~train_mask, feature_cols]
    y_test = features_df.loc[~train_mask, "anomaly"]

    print(f"[IForestSliding] Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"[IForestSliding] Features: {len(feature_cols)}")
    return X_train, X_test, y_train, y_test, feature_cols


def train_and_evaluate(X_train, X_test, y_train, y_test, contamination=0.2):
    """训练并评估滑动窗口 IForest"""
    model = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    y_pred_raw = model.predict(X_test)
    y_pred = (y_pred_raw == -1).astype(int)
    y_scores = -model.decision_function(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_scores),
    }

    print(f"  F1={metrics['f1']:.3f}, AUC={metrics['auc_roc']:.3f}, "
          f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
    return metrics, model, y_pred, y_scores


def run_per_channel_eval(features_df, contamination=0.2):
    """分通道评估"""
    channels = sorted(features_df["channel"].unique())
    results = {}

    for ch in channels:
        ch_data = features_df[features_df["channel"] == ch]
        if ch_data["anomaly"].nunique() < 2:
            print(f"  Channel {ch}: skipped (only 1 class)")
            continue

        meta_cols = ["channel", "segment", "anomaly", "train", "sampling"]
        feature_cols = [c for c in ch_data.columns if c not in meta_cols]

        train_mask = ch_data["train"] == 1
        X_train = ch_data.loc[train_mask, feature_cols]
        y_train = ch_data.loc[train_mask, "anomaly"]
        X_test = ch_data.loc[~train_mask, feature_cols]
        y_test = ch_data.loc[~train_mask, "anomaly"]

        if len(X_train) < 10 or len(X_test) < 5:
            print(f"  Channel {ch}: skipped (too few samples)")
            continue

        print(f"\n  Channel: {ch} (train={len(X_train)}, test={len(X_test)})")
        metrics, _, _, _ = train_and_evaluate(
            X_train, X_test, y_train, y_test, contamination
        )
        results[ch] = metrics

    return results


def plot_per_channel_f1(results, save_path=None):
    """绘制分通道 F1 对比图"""
    channels = list(results.keys())
    f1_scores = [results[ch]["f1"] for ch in channels]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(channels)), f1_scores, color="#2F5496")
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("IForest + Sliding Window: Per-Channel F1 Score", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=np.mean(f1_scores), color="red", linestyle="--",
               label=f"Mean F1={np.mean(f1_scores):.3f}")
    ax.legend()

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {save_path}")
    plt.close()


def run_sliding_pipeline(config=None):
    """完整的滑动窗口 IForest 流程"""
    if config is None:
        config = load_config()

    # 1. 提取滑动窗口特征（如果还没有）
    root = get_project_root()
    sw_path = os.path.join(root, config["data"]["processed_dir"], "sliding_window_features.csv")
    if os.path.exists(sw_path):
        print(f"[SlidingPipeline] Loading existing features: {sw_path}")
        features_df = pd.read_csv(sw_path, encoding="utf-8")
    else:
        features_df = run_sliding_window_pipeline(config)

    # 2. 划分数据
    X_train, X_test, y_train, y_test, feature_cols = get_sw_train_test(features_df)

    # 3. 训练评估
    results_dir = os.path.join(root, config["data"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    # 不同 contamination 对比
    all_results = {}
    for c in [0.1, 0.15, 0.2, 0.25]:
        print(f"\n{'='*50}")
        print(f"Sliding Window IForest — contamination={c}")
        print(f"{'='*50}")
        metrics, model, y_pred, y_scores = train_and_evaluate(
            X_train, X_test, y_train, y_test, contamination=c
        )
        all_results[f"sw_contam_{c}"] = metrics

    # 4. 分通道评估
    print(f"\n{'='*50}")
    print("Per-Channel Evaluation (contamination=0.2)")
    print(f"{'='*50}")
    per_channel = run_per_channel_eval(features_df, contamination=0.2)
    all_results["per_channel"] = per_channel

    # 5. 保存结果
    out_path = os.path.join(results_dir, "iforest_sliding_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Results] Saved: {out_path}")

    # 6. 画分通道 F1 图
    if per_channel:
        plot_path = os.path.join(results_dir, "iforest_sliding_per_channel_f1.png")
        plot_per_channel_f1(per_channel, save_path=plot_path)

    return all_results


if __name__ == "__main__":
    results = run_sliding_pipeline()
