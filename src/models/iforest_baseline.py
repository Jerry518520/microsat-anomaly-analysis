"""
IForest 基准模型
使用段级特征 + IForest 进行无监督异常检测
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
from src.features.segment_features import run_segment_baseline


def train_iforest(X_train, contamination=0.2, random_state=42, n_estimators=100):
    """训练 IForest 模型"""
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    print(f"[IForest] Trained with contamination={contamination}, n_estimators={n_estimators}")
    return model


def evaluate_iforest(model, X_test, y_test):
    """评估 IForest 模型"""
    # IForest 预测: 1=正常, -1=异常; 我们需要转换为 0=正常, 1=异常
    y_pred_raw = model.predict(X_test)
    y_pred = (y_pred_raw == -1).astype(int)

    # 异常分数 (越高越异常)
    y_scores = -model.decision_function(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_scores),
    }

    print(f"[IForest] Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  AUC_ROC:   {metrics['auc_roc']:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'])}")

    return metrics, y_pred, y_scores


def run_iforest_variants(X_train, X_test, y_train, y_test):
    """运行不同 contamination 的 IForest 对比"""
    results = {}
    contaminations = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, "auto"]

    for c in contaminations:
        print(f"\n{'='*50}")
        print(f"contamination = {c}")
        print(f"{'='*50}")
        contam = 0.2 if c == "auto" else c
        model = train_iforest(X_train, contamination=contam)
        metrics, y_pred, y_scores = evaluate_iforest(model, X_test, y_test)
        results[str(c)] = metrics

    return results


def plot_contamination_comparison(results, save_path=None):
    """绘制不同 contamination 的指标对比图"""
    contaminations = list(results.keys())
    f1_scores = [results[c]["f1"] for c in contaminations]
    auc_scores = [results[c]["auc_roc"] for c in contaminations]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(contaminations))
    width = 0.35

    bars1 = ax.bar(x - width/2, f1_scores, width, label="F1", color="#2F5496")
    bars2 = ax.bar(x + width/2, auc_scores, width, label="AUC_ROC", color="#C00000")

    ax.set_xlabel("Contamination", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("IForest: Contamination vs Metrics (Segment Features)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(contaminations)
    ax.legend()
    ax.set_ylim(0, 1.05)

    # 在柱子上标注数值
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {save_path}")
    plt.close()


def save_results(results, filename="iforest_segment_baseline_results.json", config=None):
    """保存实验结果"""
    if config is None:
        config = load_config()
    root = get_project_root()
    out_dir = os.path.join(root, config["data"]["results_dir"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Results] Saved: {out_path}")


def run_baseline_pipeline(config=None):
    """完整的段级 IForest baseline 流程"""
    if config is None:
        config = load_config()

    # 1. 加载段级特征
    X_train, X_test, y_train, y_test, feature_names = run_segment_baseline(config)

    # 2. 跑不同 contamination 对比
    results = run_iforest_variants(X_train, X_test, y_train, y_test)

    # 3. 保存结果
    save_results(results, "iforest_segment_baseline_results.json", config)

    # 4. 画图
    root = get_project_root()
    results_dir = os.path.join(root, config["data"]["results_dir"])
    plot_path = os.path.join(results_dir, "iforest_contamination_comparison.png")
    plot_contamination_comparison(results, save_path=plot_path)

    return results


if __name__ == "__main__":
    results = run_baseline_pipeline()
