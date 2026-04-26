"""
特征贡献度分析模块
基于 IForest 的特征重要性 + Permutation Importance
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from src.utils.data_loader import load_config, get_project_root
from src.features.segment_features import run_segment_baseline


def compute_iforest_importance(model, feature_names, top_k=10):
    """
    IForest 特征重要性 (基于树分裂频率统计)
    sklearn IsolationForest 没有内置 feature_importances_，
    因此手动统计每棵树中各特征被用作分裂节点的次数。
    """
    # 统计每棵树中各特征的分裂次数
    n_features = len(feature_names)
    split_counts = np.zeros(n_features)

    for tree in model.estimators_:
        tree_obj = tree.tree_
        feature_indices = tree_obj.feature
        # -2 表示叶子节点
        valid = feature_indices >= 0
        for fi in feature_indices[valid]:
            split_counts[fi] += 1

    # 归一化
    importances = split_counts / split_counts.sum() if split_counts.sum() > 0 else split_counts
    indices = np.argsort(importances)[::-1]

    result = []
    for i in range(min(top_k, len(feature_names))):
        idx = indices[i]
        result.append({
            "feature": feature_names[idx],
            "importance": float(importances[idx]),
            "rank": i + 1,
        })

    print(f"[FeatureImportance] Top {len(result)} features:")
    for r in result:
        print(f"  #{r['rank']}: {r['feature']} = {r['importance']:.4f}")

    return result


def compute_permutation_importance(model, X_test, y_test, feature_names,
                                    top_k=10, n_repeats=10):
    """
    Permutation Importance (基于 F1 下降)
    更可靠的特征重要性评估
    """
    # 自定义 scoring 函数，处理 IForest 的 1/-1 输出
    from sklearn.metrics import make_scorer
    def iforest_f1(y_true, y_pred_raw):
        y_pred = (y_pred_raw == -1).astype(int)
        return f1_score(y_true, y_pred, zero_division=0)

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring=make_scorer(iforest_f1),
    )

    indices = np.argsort(result.importances_mean)[::-1]

    perm_result = []
    for i in range(min(top_k, len(feature_names))):
        idx = indices[i]
        perm_result.append({
            "feature": feature_names[idx],
            "importance_mean": float(result.importances_mean[idx]),
            "importance_std": float(result.importances_std[idx]),
            "rank": i + 1,
        })

    print(f"\n[PermutationImportance] Top {len(perm_result)} features:")
    for r in perm_result:
        print(f"  #{r['rank']}: {r['feature']} = {r['importance_mean']:.4f} ± {r['importance_std']:.4f}")

    return perm_result


def plot_feature_importance(iforest_imp, perm_imp, save_path=None):
    """绘制双视图特征贡献度图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # IForest 内置重要性
    names = [r["feature"] for r in iforest_imp][::-1]
    values = [r["importance"] for r in iforest_imp][::-1]
    axes[0].barh(range(len(names)), values, color="#2F5496")
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].set_xlabel("Importance")
    axes[0].set_title("IForest Built-in Feature Importance", fontsize=13)

    # Permutation Importance
    names2 = [r["feature"] for r in perm_imp][::-1]
    means = [r["importance_mean"] for r in perm_imp][::-1]
    stds = [r["importance_std"] for r in perm_imp][::-1]
    axes[1].barh(range(len(names2)), means, xerr=stds, color="#C00000", alpha=0.8)
    axes[1].set_yticks(range(len(names2)))
    axes[1].set_yticklabels(names2, fontsize=9)
    axes[1].set_xlabel("Importance (F1 drop)")
    axes[1].set_title("Permutation Importance", fontsize=13)

    plt.suptitle("Feature Contribution Analysis", fontsize=15, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {save_path}")
    plt.close()


def run_feature_importance_pipeline(config=None):
    """完整的特征贡献度分析流程"""
    if config is None:
        config = load_config()

    # 1. 加载数据 + 训练模型
    X_train, X_test, y_train, y_test, feature_names = run_segment_baseline(config)
    model = IsolationForest(
        n_estimators=100, contamination=0.2,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train)

    # 2. IForest 内置特征重要性
    iforest_imp = compute_iforest_importance(model, feature_names, top_k=10)

    # 3. Permutation Importance
    perm_imp = compute_permutation_importance(
        model, X_test, y_test, feature_names, top_k=10
    )

    # 4. 保存结果
    root = get_project_root()
    results_dir = os.path.join(root, config["data"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    results = {
        "iforest_importance": iforest_imp,
        "permutation_importance": perm_imp,
    }
    out_path = os.path.join(results_dir, "feature_importance.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Results] Saved: {out_path}")

    # 5. 画图
    plot_path = os.path.join(results_dir, "feature_importance.png")
    plot_feature_importance(iforest_imp, perm_imp, save_path=plot_path)

    return results


if __name__ == "__main__":
    run_feature_importance_pipeline()
