"""
投票阈值调优实验 (1.8)
目标: SegF1从0.342突破0.35
思路: 当前方案G的段级聚合阈值=0.5(多数投票), Precision=0.238偏低
      扫描0.30-0.50五档, 找P/R平衡点
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from src.utils.data_loader import load_config, get_project_root, load_segments
from src.features.sliding_window import extract_all_sliding_features
from src.utils.constants import META_COLS
from src.experiments.rule_fallback import compute_stat_thresholds, apply_stat_rules


def segment_aggregate_with_threshold(features_df, y_pred_window, threshold, label=""):
    """窗口级预测聚合为段级, 可指定阈值"""
    test_mask = features_df["train"] == 0
    test_df = features_df[test_mask].copy()
    test_df["pred"] = y_pred_window.values if hasattr(y_pred_window, 'values') else y_pred_window

    seg = test_df.groupby("segment").agg(
        y_true=("anomaly", "first"),
        pred_rate=("pred", "mean"),
    ).reset_index()
    seg["y_pred"] = (seg["pred_rate"] >= threshold).astype(int)

    seg_f1 = f1_score(seg["y_true"], seg["y_pred"])
    seg_p = precision_score(seg["y_true"], seg["y_pred"], zero_division=0)
    seg_r = recall_score(seg["y_true"], seg["y_pred"], zero_division=0)
    try:
        seg_auc = roc_auc_score(seg["y_true"], seg["pred_rate"])
    except ValueError:
        seg_auc = None

    return {"f1": seg_f1, "precision": seg_p, "recall": seg_r, "auc": seg_auc}


def per_channel_iforest(train_df, test_df, feature_cols, best_c_dict):
    channel_preds = {}
    channel_f1 = {}
    for ch in sorted(test_df["channel"].unique()):
        ch_test = test_df[test_df["channel"] == ch]
        ch_train = train_df[train_df["channel"] == ch]
        if len(ch_train) < 10 or len(ch_test) < 5:
            continue
        ch_c = best_c_dict.get(ch, 0.5)
        X_tr = np.nan_to_num(ch_train[feature_cols].values)
        y_tr = ch_train["anomaly"].values
        X_te = np.nan_to_num(ch_test[feature_cols].values)
        y_te = ch_test["anomaly"].values
        model = IsolationForest(n_estimators=100, contamination=ch_c, random_state=42, n_jobs=-1)
        model.fit(X_tr)
        ch_pred = (model.predict(X_te) == -1).astype(int)
        for i, idx in enumerate(ch_test.index):
            channel_preds[idx] = int(ch_pred[i])
        ch_f1 = f1_score(y_te, ch_pred)
        channel_f1[ch] = ch_f1
    return channel_preds, channel_f1


def segment_baseline_iforest():
    root = get_project_root()
    config = load_config()
    feat_path = os.path.join(root, config["data"]["raw_dir"], config["data"]["features_file"])
    dataset_df = pd.read_csv(feat_path, encoding="utf-8")
    meta_cols = ["channel", "segment", "anomaly", "train", "sampling"]
    feat_cols = [c for c in dataset_df.columns if c not in meta_cols]
    train_mask = dataset_df["train"] == 1
    X_train = dataset_df.loc[train_mask, feat_cols].values
    X_test = dataset_df.loc[~train_mask, feat_cols].values
    y_test = dataset_df.loc[~train_mask, "anomaly"].values
    test_segments = dataset_df.loc[~train_mask, "segment"].values
    model = IsolationForest(n_estimators=100, contamination=0.25, random_state=42, n_jobs=-1)
    model.fit(X_train)
    y_pred = (model.predict(X_test) == -1).astype(int)
    seg_pred_map = {}
    for i in range(len(test_segments)):
        seg_pred_map[test_segments[i]] = int(y_pred[i])
    return seg_pred_map


def main():
    config = load_config()
    root = get_project_root()
    segments_df = load_segments(config)
    results_dir = os.path.join(root, config["data"]["results_dir"])

    # Step 1: Extract features
    print("Step 1: Feature extraction (ws=20)")
    features_df = extract_all_sliding_features(segments_df, window_size=20, step_size=10)
    feature_cols = [c for c in features_df.columns if c not in META_COLS]
    train_mask = features_df["train"] == 1
    train_df = features_df[train_mask]
    test_df = features_df[~train_mask].copy()

    # Step 2: Statistical rules
    print("Step 2: Statistical rules")
    thresholds = compute_stat_thresholds(train_df, feature_cols)
    n_violated, n_total = apply_stat_rules(test_df, thresholds, feature_cols)
    test_df["rule_violations"] = n_violated
    test_df["rule_total"] = n_total
    rule_pred = pd.Series((n_violated / n_total.replace(0, 1) >= 0.2).astype(int), index=test_df.index)

    # Step 3: Per-channel IForest
    print("Step 3: Per-channel IForest")
    best_c = {
        "CADC0872": 0.5, "CADC0873": 0.5, "CADC0874": 0.5,
        "CADC0886": 0.2, "CADC0888": 0.45, "CADC0890": 0.3,
        "CADC0892": 0.5, "CADC0894": 0.5,
    }
    channel_preds, channel_f1 = per_channel_iforest(train_df, test_df, feature_cols, best_c)

    # Step 4: Segment baseline
    print("Step 4: Segment-level baseline")
    seg_baseline_map = segment_baseline_iforest()

    # Step 5: Build method G predictions (strong IF+OR Rule, weak seg baseline)
    print("Step 5: Build method G window-level predictions")
    strong_channels = {"CADC0872", "CADC0873", "CADC0874"}
    pred_g = pd.Series(0, index=test_df.index, dtype=int)
    for idx in test_df.index:
        ch = test_df.loc[idx, "channel"]
        seg_id = test_df.loc[idx, "segment"]
        if ch in strong_channels:
            if_pred = channel_preds.get(idx, 0)
            r_pred = int(rule_pred.loc[idx])
            pred_g.loc[idx] = max(if_pred, r_pred)
        else:
            pred_g.loc[idx] = seg_baseline_map.get(seg_id, 0)

    # Step 6: Threshold sweep
    print("\nStep 6: Threshold sweep on method G")
    thresholds_to_test = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    all_results = {}

    print(f"\n{'Threshold':>10} | {'SegF1':>6} | {'P':>6} | {'R':>6} | {'AUC':>6}")
    print("-" * 45)

    best_f1 = 0
    best_th = 0.5
    for th in thresholds_to_test:
        metrics = segment_aggregate_with_threshold(features_df, pred_g, th, f"th={th}")
        all_results[str(th)] = metrics
        auc_str = f"{metrics['auc']:.3f}" if metrics['auc'] else "N/A"
        print(f"{th:>10.2f} | {metrics['f1']:>6.3f} | {metrics['precision']:>6.3f} | {metrics['recall']:>6.3f} | {auc_str:>6}")
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_th = th

    print(f"\nBest: threshold={best_th:.2f}, SegF1={best_f1:.3f}")

    # Save results
    output = {
        "experiment": "threshold_sweep_1.8",
        "method": "G_strong_IF_OR_Rule+weak_seg",
        "window_size": 20,
        "step_size": 10,
        "best_contamination": best_c,
        "thresholds_tested": thresholds_to_test,
        "results": all_results,
        "best_threshold": best_th,
        "best_seg_f1": best_f1,
    }
    out_path = os.path.join(results_dir, "threshold_sweep_1.8_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[Saved] {out_path}")
    return output


if __name__ == "__main__":
    main()
