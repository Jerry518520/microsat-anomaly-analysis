"""
Contamination 精细搜索 + 分通道独立调参
基于 window_size=20（段级F1最优窗口）
1. 全局 contamination 搜索: [0.2, 0.25, 0.3, 0.35, 0.37, 0.4, 0.45, 0.5]
2. 分通道独立搜索每个通道的最佳 contamination
3. 输出最佳参数组合供后续规则兜底使用
"""

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from src.utils.data_loader import load_segments, load_config, get_project_root
from src.features.sliding_window import extract_all_sliding_features, META_COLS


def eval_iforest(X_train, y_train, X_test, y_test, contamination):
    """训练IForest并返回指标"""
    model = IsolationForest(n_estimators=100, max_samples="auto",
                            contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X_train)
    y_pred = (model.predict(X_test) == -1).astype(int)
    y_scores = -model.decision_function(X_test)

    metrics = {
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_test, y_scores))
    except ValueError:
        metrics["auc"] = None
    return metrics, y_pred, y_scores


def segment_level_eval(features_df, y_pred_window):
    """窗口级 → 段级聚合评估（多数投票）"""
    test_mask = features_df["train"] == 0
    test_df = features_df[test_mask].copy()
    test_df["pred"] = y_pred_window

    seg_pred = test_df.groupby("segment").agg(
        anomaly=("anomaly", "first"),
        pred_rate=("pred", "mean"),
        channel=("channel", "first"),
    ).reset_index()
    seg_pred["pred_label"] = (seg_pred["pred_rate"] >= 0.5).astype(int)

    f1_seg = f1_score(seg_pred["anomaly"], seg_pred["pred_label"])
    try:
        auc_seg = float(roc_auc_score(seg_pred["anomaly"], seg_pred["pred_rate"]))
    except ValueError:
        auc_seg = None

    return {"f1": float(f1_seg), "auc": auc_seg}, seg_pred


def main():
    config = load_config()
    root = get_project_root()
    segments_df = load_segments(config)
    results_dir = os.path.join(root, config["data"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    # ===== Step 1: 用 ws=20 提取特征 =====
    print("=" * 60)
    print("Step 1: Extracting features (window_size=20)")
    print("=" * 60)
    features_df = extract_all_sliding_features(segments_df, window_size=20, step_size=10)
    feature_cols = [c for c in features_df.columns if c not in META_COLS]

    train_mask = features_df["train"] == 1
    X_train_all = np.nan_to_num(features_df.loc[train_mask, feature_cols].values)
    y_train_all = features_df.loc[train_mask, "anomaly"].values
    X_test_all = np.nan_to_num(features_df.loc[~train_mask, feature_cols].values)
    y_test_all = features_df.loc[~train_mask, "anomaly"].values

    # ===== Step 2: 全局 contamination 搜索 =====
    print("\n" + "=" * 60)
    print("Step 2: Global contamination sweep")
    print("=" * 60)

    contam_range = [0.2, 0.25, 0.3, 0.35, 0.37, 0.4, 0.45, 0.5]
    global_results = {}

    for c in contam_range:
        wl_metrics, y_pred, y_scores = eval_iforest(
            X_train_all, y_train_all, X_test_all, y_test_all, contamination=c
        )
        sl_metrics, seg_pred = segment_level_eval(features_df, y_pred)

        global_results[f"contam_{c}"] = {
            "window_level": wl_metrics,
            "segment_level": sl_metrics,
        }

        print(f"  c={c:.2f} | WinF1={wl_metrics['f1']:.3f} | "
              f"SegF1={sl_metrics['f1']:.3f} | P={wl_metrics['precision']:.3f} | R={wl_metrics['recall']:.3f}")

    # 找全局最佳
    best_global_key = max(
        [k for k in global_results if k.startswith("contam_")],
        key=lambda k: global_results[k]["segment_level"]["f1"]
    )
    best_global_c = float(best_global_key.replace("contam_", ""))
    print(f"\n  >>> Best global contamination: {best_global_c} (SegF1={global_results[best_global_key]['segment_level']['f1']:.3f})")

    # ===== Step 3: 分通道独立搜索 =====
    print("\n" + "=" * 60)
    print("Step 3: Per-channel independent contamination search")
    print("=" * 60)

    per_channel_best = {}
    channels = sorted(features_df["channel"].unique())

    for ch in channels:
        ch_data = features_df[features_df["channel"] == ch]
        if ch_data["anomaly"].nunique() < 2:
            print(f"  {ch}: skipped (single class)")
            continue

        ch_train = ch_data[ch_data["train"] == 1]
        ch_test = ch_data[ch_data["train"] == 0]
        X_tr = np.nan_to_num(ch_train[feature_cols].values)
        y_tr = ch_train["anomaly"].values
        X_te = np.nan_to_num(ch_test[feature_cols].values)
        y_te = ch_test["anomaly"].values

        if len(X_tr) < 10 or len(X_te) < 5:
            print(f"  {ch}: skipped (too few samples)")
            continue

        best_ch_c = None
        best_ch_f1 = -1
        ch_all = {}

        for c in contam_range:
            try:
                metrics, _, _ = eval_iforest(X_tr, y_tr, X_te, y_te, contamination=c)
                ch_all[f"c={c}"] = metrics
                if metrics["f1"] > best_ch_f1:
                    best_ch_f1 = metrics["f1"]
                    best_ch_c = c
            except Exception as e:
                ch_all[f"c={c}"] = {"error": str(e)}

        per_channel_best[ch] = {
            "best_contamination": best_ch_c,
            "best_f1": float(best_ch_f1),
            "all_contaminations": ch_all,
        }
        print(f"  {ch}: best_c={best_ch_c}, F1={best_ch_f1:.3f}")

    # ===== Step 4: 分通道最优c组合评估 =====
    print("\n" + "=" * 60)
    print("Step 4: Per-channel optimized combination evaluation")
    print("=" * 60)

    # 用每个通道的最佳c重新训练，收集窗口级预测
    all_preds = None
    ch_eval_results = {}

    for ch, ch_info in per_channel_best.items():
        ch_c = ch_info["best_contamination"]
        ch_data = features_df[features_df["channel"] == ch]
        ch_train = ch_data[ch_data["train"] == 1]
        ch_test = ch_data[ch_data["train"] == 0]
        ch_test_idx = ch_test.index
        X_tr = np.nan_to_num(ch_train[feature_cols].values)
        y_tr = ch_train["anomaly"].values
        X_te = np.nan_to_num(ch_test[feature_cols].values)
        y_te = ch_test["anomaly"].values

        if len(X_tr) < 10 or len(X_te) < 5:
            continue

        model = IsolationForest(n_estimators=100, max_samples="auto",
                                contamination=ch_c, random_state=42, n_jobs=-1)
        model.fit(X_tr)
        y_pred = (model.predict(X_te) == -1).astype(int)

        if all_preds is None:
            all_preds = pd.Series(-1, dtype=int, index=features_df[~train_mask].index)
        all_preds.loc[ch_test_idx] = y_pred

        ch_metrics = {
            "f1": float(f1_score(y_te, y_pred)),
            "precision": float(precision_score(y_te, y_pred, zero_division=0)),
            "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        }
        ch_eval_results[ch] = ch_metrics

    # 段级聚合
    if all_preds is not None and (all_preds >= 0).all():
        combined_seg_metrics, combined_seg_pred = segment_level_eval(features_df, all_preds.values)
    else:
        combined_seg_metrics = {"f1": 0.0, "auc": None}
        combined_seg_pred = None
    auc_str = f"{combined_seg_metrics['auc']:.3f}" if combined_seg_metrics['auc'] is not None else "N/A"
    print(f"  Combined SegF1={combined_seg_metrics['f1']:.3f}, SegAUC={auc_str}")

    # ===== 保存结果 =====
    all_output = {
        "window_size": 20,
        "step_size": 10,
        "global_contamination_sweep": global_results,
        "best_global_contamination": best_global_c,
        "per_channel_best": per_channel_best,
        "combined_per_channel_eval": {
            "segment_level": combined_seg_metrics,
            "per_channel_metrics": ch_eval_results,
        },
    }

    out_path = os.path.join(results_dir, "contamination_search_ws20.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[Results] Saved: {out_path}")

    # 摘要表
    print(f"\n{'='*60}")
    print("CONTAMINATION SEARCH SUMMARY (ws=20)")
    print(f"{'='*60}")
    print(f"\n--- Global Sweep ---")
    print(f"{'c':>6} | {'WinF1':>6} | {'SegF1':>6} | {'P':>6} | {'R':>6}")
    print("-" * 40)
    for c in contam_range:
        r = global_results[f"contam_{c}"]
        wl = r["window_level"]
        sl = r["segment_level"]
        print(f"{c:>6.2f} | {wl['f1']:>6.3f} | {sl['f1']:>6.3f} | {wl['precision']:>6.3f} | {wl['recall']:>6.3f}")

    print(f"\n--- Per-Channel Best ---")
    print(f"{'Channel':>10} | {'Best_c':>6} | {'F1':>6}")
    print("-" * 30)
    for ch, info in per_channel_best.items():
        print(f"{ch:>10} | {info['best_contamination']:>6.2f} | {info['best_f1']:>6.3f}")

    print(f"\n--- Combined ---")
    auc_str2 = f"{combined_seg_metrics['auc']:.3f}" if combined_seg_metrics.get('auc') is not None else "N/A"
    print(f"Segment F1 = {combined_seg_metrics['f1']:.3f}, AUC = {auc_str2}")


if __name__ == "__main__":
    main()
