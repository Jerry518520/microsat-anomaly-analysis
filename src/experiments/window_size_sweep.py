"""
窗口大小调参实验
测试 window_size=[20, 30, 40, 50, 60]，固定 step_size=10, contamination=0.4
每次重新提取特征 + IForest训练 + 整体/分通道评估
"""

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from src.utils.data_loader import load_config, get_project_root, load_segments
from src.features.sliding_window import extract_all_sliding_features
from src.utils.constants import META_COLS


def run_iforest_eval(features_df, contamination=0.4):
    """训练IForest并评估（窗口级+段级）"""
    feature_cols = [c for c in features_df.columns if c not in META_COLS]

    train_mask = features_df["train"] == 1
    X_train = np.nan_to_num(features_df.loc[train_mask, feature_cols].values)
    y_train = features_df.loc[train_mask, "anomaly"].values
    X_test = np.nan_to_num(features_df.loc[~train_mask, feature_cols].values)
    y_test = features_df.loc[~train_mask, "anomaly"].values

    model = IsolationForest(n_estimators=100, max_samples="auto",
                            contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X_train)

    y_pred = (model.predict(X_test) == -1).astype(int)
    y_scores = -model.decision_function(X_test)

    # 窗口级指标
    win_metrics = {
        "f1": float(f1_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_scores)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }

    # 段级聚合
    test_df = features_df[~train_mask].copy()
    test_df["pred"] = y_pred
    seg_pred = test_df.groupby("segment").agg(
        anomaly=("anomaly", "first"), pred_rate=("pred", "mean")
    ).reset_index()
    seg_pred["pred_label"] = (seg_pred["pred_rate"] >= 0.5).astype(int)

    seg_metrics = {
        "f1": float(f1_score(seg_pred["anomaly"], seg_pred["pred_label"])),
        "auc": float(roc_auc_score(seg_pred["anomaly"], seg_pred["pred_rate"])),
        "n_segments": int(len(seg_pred)),
    }

    # 分通道
    per_channel = {}
    for ch in sorted(features_df["channel"].unique()):
        ch_data = features_df[features_df["channel"] == ch]
        if ch_data["anomaly"].nunique() < 2:
            continue
        ch_train = ch_data[ch_data["train"] == 1]
        ch_test = ch_data[ch_data["train"] == 0]
        X_tr = np.nan_to_num(ch_train[feature_cols].values)
        y_tr = ch_train["anomaly"].values
        X_te = np.nan_to_num(ch_test[feature_cols].values)
        y_te = ch_test["anomaly"].values
        if len(X_tr) < 10 or len(X_te) < 5:
            continue
        ch_model = IsolationForest(n_estimators=100, max_samples="auto",
                                   contamination=contamination, random_state=42, n_jobs=-1)
        ch_model.fit(X_tr)
        ch_pred = (ch_model.predict(X_te) == -1).astype(int)
        ch_scores = -ch_model.decision_function(X_te)
        per_channel[ch] = {
            "f1": float(f1_score(y_te, ch_pred)),
            "auc": float(roc_auc_score(y_te, ch_scores)),
        }

    return {"window_level": win_metrics, "segment_level": seg_metrics, "per_channel": per_channel}


def main():
    config = load_config()
    root = get_project_root()
    segments_df = load_segments(config)
    results_dir = os.path.join(root, config["data"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    window_sizes = [20, 30, 40, 50, 60]
    contamination = 0.4
    all_results = {}

    for ws in window_sizes:
        print(f"\n{'='*60}")
        print(f"window_size={ws}, step_size=10, contamination={contamination}")
        print(f"{'='*60}")

        t0 = time.time()
        features_df = extract_all_sliding_features(segments_df, window_size=ws, step_size=10)
        elapsed = time.time() - t0
        print(f"  Feature extraction: {elapsed:.1f}s, {len(features_df)} samples")

        result = run_iforest_eval(features_df, contamination=contamination)
        result["n_samples"] = len(features_df)
        result["n_features"] = len([c for c in features_df.columns if c not in META_COLS])
        result["extraction_time_s"] = round(elapsed, 1)

        all_results[f"window_{ws}"] = result

        # 打印摘要
        wl = result["window_level"]
        sl = result["segment_level"]
        print(f"  Window F1={wl['f1']:.3f}, AUC={wl['auc']:.3f}")
        print(f"  Segment F1={sl['f1']:.3f}, AUC={sl['auc']:.3f} ({sl['n_segments']} segs)")
        ch_f1s = [v["f1"] for v in result["per_channel"].values()]
        print(f"  Per-channel mean F1={np.mean(ch_f1s):.3f}, best={max(ch_f1s):.3f}")

    # 保存
    out_path = os.path.join(results_dir, "window_size_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[Results] Saved: {out_path}")

    # 摘要表
    print(f"\n{'='*60}")
    print("WINDOW SIZE SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'WS':>4} | {'Samples':>7} | {'WinF1':>6} | {'SegF1':>6} | {'SegAUC':>6} | {'ChMeanF1':>8} | {'ChBestF1':>8}")
    print("-" * 60)
    for ws in window_sizes:
        r = all_results[f"window_{ws}"]
        ch_f1s = [v["f1"] for v in r["per_channel"].values()]
        print(f"{ws:>4} | {r['n_samples']:>7} | {r['window_level']['f1']:>6.3f} | "
              f"{r['segment_level']['f1']:>6.3f} | {r['segment_level']['auc']:>6.3f} | "
              f"{np.mean(ch_f1s):>8.3f} | {max(ch_f1s):>8.3f}")


if __name__ == "__main__":
    main()
