"""
段级 vs 滑动窗口 对比实验
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score

from src.utils.data_loader import load_config, get_project_root
from src.features.segment_features import run_segment_baseline
from src.features.sliding_window import run_sliding_window_pipeline
from src.models.iforest_sliding import get_sw_train_test


def run_comparison(config=None):
    """
    段级特征 vs 滑动窗口特征 全面对比
    返回对比结果 dict
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    results_dir = os.path.join(root, config["data"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    # ---- 段级特征 ----
    print("=" * 60)
    print("SEGMENT FEATURES BASELINE")
    print("=" * 60)
    X_train_seg, X_test_seg, y_train_seg, y_test_seg, seg_cols = run_segment_baseline(config)

    seg_results = {}
    for c in [0.1, 0.2, 0.25]:
        model = IsolationForest(n_estimators=100, contamination=c,
                                random_state=42, n_jobs=-1)
        model.fit(X_train_seg)
        y_pred = (model.predict(X_test_seg) == -1).astype(int)
        y_scores = -model.decision_function(X_test_seg)
        seg_results[f"seg_c{c}"] = {
            "f1": f1_score(y_test_seg, y_pred),
            "auc_roc": roc_auc_score(y_test_seg, y_scores),
        }
        print(f"  Segment c={c}: F1={seg_results[f'seg_c{c}']['f1']:.3f}, "
              f"AUC={seg_results[f'seg_c{c}']['auc_roc']:.3f}")

    # ---- 滑动窗口特征 ----
    print("\n" + "=" * 60)
    print("SLIDING WINDOW FEATURES")
    print("=" * 60)
    sw_path = os.path.join(root, config["data"]["processed_dir"], "sliding_window_features.csv")
    if os.path.exists(sw_path):
        sw_df = pd.read_csv(sw_path, encoding="utf-8")
    else:
        sw_df = run_sliding_window_pipeline(config)

    X_train_sw, X_test_sw, y_train_sw, y_test_sw, sw_cols = get_sw_train_test(sw_df)

    sw_results = {}
    for c in [0.1, 0.2, 0.25]:
        model = IsolationForest(n_estimators=100, contamination=c,
                                random_state=42, n_jobs=-1)
        model.fit(X_train_sw)
        y_pred = (model.predict(X_test_sw) == -1).astype(int)
        y_scores = -model.decision_function(X_test_sw)
        sw_results[f"sw_c{c}"] = {
            "f1": f1_score(y_test_sw, y_pred),
            "auc_roc": roc_auc_score(y_test_sw, y_scores),
        }
        print(f"  SlidingWindow c={c}: F1={sw_results[f'sw_c{c}']['f1']:.3f}, "
              f"AUC={sw_results[f'sw_c{c}']['auc_roc']:.3f}")

    # ---- 汇总 ----
    comparison = {"segment": seg_results, "sliding_window": sw_results}

    # 保存
    out_path = os.path.join(results_dir, "segment_vs_sliding_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n[Comparison] Saved: {out_path}")

    # 画对比图
    _plot_comparison(seg_results, sw_results,
                     save_path=os.path.join(results_dir, "segment_vs_sliding_comparison.png"))

    return comparison


def _plot_comparison(seg_results, sw_results, save_path=None):
    """绘制段级 vs 滑动窗口 F1/AUC 对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    seg_keys = [f"seg_c{c}" for c in [0.1, 0.2, 0.25]]
    sw_keys = [f"sw_c{c}" for c in [0.1, 0.2, 0.25]]
    x = np.arange(len(seg_keys))
    width = 0.35

    # F1 对比
    seg_f1 = [seg_results[k]["f1"] for k in seg_keys]
    sw_f1 = [sw_results[k]["f1"] for k in sw_keys]
    axes[0].bar(x - width/2, seg_f1, width, label="Segment", color="#8B0000")
    axes[0].bar(x + width/2, sw_f1, width, label="Sliding Window", color="#2F5496")
    axes[0].set_title("F1 Score Comparison", fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["c=0.1", "c=0.2", "c=0.25"])
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    # AUC 对比
    seg_auc = [seg_results[k]["auc_roc"] for k in seg_keys]
    sw_auc = [sw_results[k]["auc_roc"] for k in sw_keys]
    axes[1].bar(x - width/2, seg_auc, width, label="Segment", color="#8B0000")
    axes[1].bar(x + width/2, sw_auc, width, label="Sliding Window", color="#2F5496")
    axes[1].set_title("AUC_ROC Comparison", fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["c=0.1", "c=0.2", "c=0.25"])
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.suptitle("Segment Features vs Sliding Window Features", fontsize=15, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    run_comparison()
