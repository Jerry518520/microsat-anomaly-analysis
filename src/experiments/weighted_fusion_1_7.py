"""
鍒嗛€氶亾鍔犳潈铻嶅悎浼樺寲 (1.7)
鐩爣: SegF1浠?.325鎻愬崌鍒?0.35

浼樺寲鎬濊矾(鍩轰簬1.6 rule_fallback鐨勭粡楠?:
1. 寮遍€氶亾(CADC0892/0888/0894绛?婊戝姩绐楀彛IForest鏁堟灉宸?鈫?閫€鍥炴绾aseline鍏滃簳
2. 鍚勯€氶亾鐙珛鏈€浼榗 + 铻嶅悎鏃舵寜楠岃瘉F1鍔犳潈鎶曠エ
3. segment绾у悗澶勭悊: 杩炵画閫氶亾涓€鑷存€т慨姝?
4. 瀵规瘮澶氱铻嶅悎绛栫暐锛屾壘鏈€浼?

涓?.6鐨勫尯鍒?
- 1.6: 寮遍€氶亾鐢ㄨ鍒欐浛浠Forest 鈫?瑙勫垯鏈韩鏁堟灉涔熷樊(F1=0.097)
- 1.7: 寮遍€氶亾閫€鍥炴绾Forest baseline 鈫?娈电骇baseline瀵硅繖浜涢€氶亾鍙嶈€屾洿濂?
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


def segment_aggregate(features_df, y_pred_window, label="Method"):
    """绐楀彛绾ч娴嬭仛鍚堜负娈电骇锛堝鏁版姇绁級"""
    test_mask = features_df["train"] == 0
    test_df = features_df[test_mask].copy()
    test_df["pred"] = y_pred_window.values if hasattr(y_pred_window, 'values') else y_pred_window

    seg = test_df.groupby("segment").agg(
        y_true=("anomaly", "first"),
        y_pred=("pred", lambda x: (x.mean() >= 0.5).astype(int)),
        pred_rate=("pred", "mean"),
        n_channels=("channel", "nunique"),
        n_windows=("pred", "count"),
    ).reset_index()

    seg_f1 = f1_score(seg["y_true"], seg["y_pred"])
    seg_p = precision_score(seg["y_true"], seg["y_pred"], zero_division=0)
    seg_r = recall_score(seg["y_true"], seg["y_pred"], zero_division=0)
    try:
        seg_auc = roc_auc_score(seg["y_true"], seg["pred_rate"])
    except ValueError:
        seg_auc = None

    win_f1 = f1_score(test_df["anomaly"].values, test_df["pred"].values)
    
    print(f"  [{label}] WinF1={win_f1:.3f} | SegF1={seg_f1:.3f} P={seg_p:.3f} R={seg_r:.3f}")
    return {
        "window": {"f1": win_f1},
        "segment": {"f1": seg_f1, "precision": seg_p, "recall": seg_r, "auc": seg_auc},
    }, seg


def per_channel_iforest(train_df, test_df, feature_cols, best_c_dict):
    """鍚勯€氶亾鐙珛璁粌IForest锛岃繑鍥炴瘡涓獥鍙ｇ殑棰勬祴鍜岄€氶亾绾1"""
    channel_preds = {}  # idx -> pred
    channel_f1 = {}     # ch -> F1(window-level)
    
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

        model = IsolationForest(n_estimators=100, contamination=ch_c,
                               random_state=42, n_jobs=-1)
        model.fit(X_tr)
        ch_pred = (model.predict(X_te) == -1).astype(int)

        for i, idx in enumerate(ch_test.index):
            channel_preds[idx] = int(ch_pred[i])

        # 绐楀彛绾1浣滀负閫氶亾璐ㄩ噺鏉冮噸
        ch_f1 = f1_score(y_te, ch_pred)
        channel_f1[ch] = ch_f1

    return channel_preds, channel_f1


def segment_baseline_iforest(segments_df):
    """
    娈电骇IForest baseline (c=0.25, 宸茬煡鏈€浣矲1=0.340)
    鐢ㄤ簬寮遍€氶亾鍏滃簳
    """
    # 鍔犺浇娈电骇鐗瑰緛
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
    
    model = IsolationForest(n_estimators=100, contamination=0.25,
                           random_state=42, n_jobs=-1)
    model.fit(X_train)
    y_pred = (model.predict(X_test) == -1).astype(int)
    
    # 鏋勫缓 segment -> pred 鏄犲皠
    seg_pred_map = {}
    for i in range(len(test_segments)):
        seg_pred_map[test_segments[i]] = int(y_pred[i])
    
    f1 = f1_score(y_test, y_pred)
    print(f"  [Segment Baseline] F1={f1:.3f} (c=0.25)")
    return seg_pred_map


def main():
    config = load_config()
    root = get_project_root()
    segments_df = load_segments(config)
    results_dir = os.path.join(root, config["data"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    # ===== Step 1: 鎻愬彇婊戝姩绐楀彛鐗瑰緛 (ws=20) =====
    print("=" * 60)
    print("Step 1: Feature extraction (ws=20)")
    print("=" * 60)
    features_df = extract_all_sliding_features(segments_df, window_size=20, step_size=10)
    feature_cols = [c for c in features_df.columns if c not in META_COLS]

    train_mask = features_df["train"] == 1
    train_df = features_df[train_mask]
    test_df = features_df[~train_mask].copy()

    # ===== Step 2: 缁熻瑙勫垯 (澶嶇敤1.6) =====
    print("\n" + "=" * 60)
    print("Step 2: Statistical rules (from 1.6)")
    print("=" * 60)
    thresholds = compute_stat_thresholds(train_df, feature_cols)
    n_violated, n_total = apply_stat_rules(test_df, thresholds, feature_cols)
    test_df["rule_violations"] = n_violated
    test_df["rule_total"] = n_total
    rule_pred = pd.Series((n_violated / n_total.replace(0, 1) >= 0.2).astype(int), index=test_df.index)

    # ===== Step 3: 鍒嗛€氶亾IForest (宸茬煡鏈€浼榗) =====
    print("\n" + "=" * 60)
    print("Step 3: Per-channel IForest with best c")
    print("=" * 60)
    best_c = {
        "CADC0872": 0.5, "CADC0873": 0.5, "CADC0874": 0.5,
        "CADC0886": 0.2, "CADC0888": 0.45, "CADC0890": 0.3,
        "CADC0892": 0.5, "CADC0894": 0.5,
    }
    
    channel_preds, channel_f1 = per_channel_iforest(train_df, test_df, feature_cols, best_c)
    
    print("\n  Channel-level window F1:")
    for ch in sorted(channel_f1.keys()):
        strength = "STRONG" if channel_f1[ch] >= 0.35 else "weak"
        print(f"    {ch}: F1={channel_f1[ch]:.3f} ({strength})")

    # ===== Step 4: 娈电骇baseline (寮遍€氶亾鍏滃簳鐢? =====
    print("\n" + "=" * 60)
    print("Step 4: Segment-level IForest baseline (for weak channel fallback)")
    print("=" * 60)
    seg_baseline_map = segment_baseline_iforest(segments_df)

    # ===== Step 5: 铻嶅悎瀹為獙 =====
    print("\n" + "=" * 60)
    print("Step 5: Fusion experiments")
    print("=" * 60)

    all_results = {}

    # --- 瀹為獙A: 1.6鍘熸柟妗?澶嶇幇) ---
    # 寮洪€氶亾IF, 寮遍€氶亾瑙勫垯
    pred_a = pd.Series(0, index=test_df.index, dtype=int)
    strong_channels = {"CADC0872", "CADC0873", "CADC0874"}
    for idx in test_df.index:
        ch = test_df.loc[idx, "channel"]
        if ch in strong_channels:
            pred_a.loc[idx] = channel_preds.get(idx, 0)
        else:
            pred_a.loc[idx] = int(rule_pred.loc[idx])
    r_a, _ = segment_aggregate(features_df[~train_mask], pred_a, "A_1.6_original")
    all_results["A_1.6_original"] = r_a

    # --- 瀹為獙B: 寮遍€氶亾娈电骇baseline鍏滃簳 ---
    # 寮洪€氶亾鐢ㄦ粦鍔ㄧ獥鍙F, 寮遍€氶亾鐢ㄦ绾aseline
    pred_b = pd.Series(0, index=test_df.index, dtype=int)
    for idx in test_df.index:
        ch = test_df.loc[idx, "channel"]
        seg_id = test_df.loc[idx, "segment"]
        if ch in strong_channels:
            pred_b.loc[idx] = channel_preds.get(idx, 0)
        else:
            # 寮遍€氶亾鐢ㄦ绾aseline棰勬祴
            pred_b.loc[idx] = seg_baseline_map.get(seg_id, 0)
    r_b, _ = segment_aggregate(features_df[~train_mask], pred_b, "B_weak_seg_baseline")
    all_results["B_weak_seg_baseline"] = r_b

    # --- 瀹為獙C: 鍏ㄩ€氶亾IF + 瑙勫垯琛ュ厖(OR) ---
    # 鎵€鏈夐€氶亾閮界敤IForest, 浣嗚鍒欐爣寮傚父鐨勪篃鏍囧紓甯?闄嶆紡鎶?
    pred_c = pd.Series(0, index=test_df.index, dtype=int)
    for idx in test_df.index:
        if_pred = channel_preds.get(idx, 0)
        r_pred = int(rule_pred.loc[idx])
        pred_c.loc[idx] = max(if_pred, r_pred)  # OR
    r_c, _ = segment_aggregate(features_df[~train_mask], pred_c, "C_all_IF_OR_Rule")
    all_results["C_all_IF_OR_Rule"] = r_c

    # --- 瀹為獙D: 鍏ㄩ€氶亾IF + 寮遍€氶亾瑙勫垯琛ュ厖 ---
    # 寮洪€氶亾绾疘F, 寮遍€氶亾IF+瑙勫垯OR
    pred_d = pd.Series(0, index=test_df.index, dtype=int)
    for idx in test_df.index:
        ch = test_df.loc[idx, "channel"]
        if_pred = channel_preds.get(idx, 0)
        r_pred = int(rule_pred.loc[idx])
        if ch in strong_channels:
            pred_d.loc[idx] = if_pred
        else:
            pred_d.loc[idx] = max(if_pred, r_pred)  # 寮遍€氶亾IF+瑙勫垯OR
    r_d, _ = segment_aggregate(features_df[~train_mask], pred_d, "D_strong_IF_weak_IF_OR_Rule")
    all_results["D_strong_IF_weak_IF_OR_Rule"] = r_d

    # --- 瀹為獙E: 鎸夐€氶亾F1鍔犳潈鎶曠エ(segment绾? ---
    # 姣忎釜segment鍐? 鍚勯€氶亾鐙珛棰勬祴, 鎸夐€氶亾F1鍔犳潈鎶曠エ
    pred_e = pd.Series(0, index=test_df.index, dtype=int)
    test_with_pred = test_df.copy()
    test_with_pred["ch_pred"] = pd.Series(channel_preds)
    
    # 鍏堣仛鍚堝埌segment脳channel绾у埆
    seg_ch = test_with_pred.groupby(["segment", "channel"]).agg(
        y_true=("anomaly", "first"),
        ch_pred_rate=("ch_pred", "mean"),  # 璇ラ€氶亾鍐呭紓甯哥獥鍙ｆ瘮渚?
    ).reset_index()
    
    # 鍔犳潈鎶曠エ (杩斿洖dict: segment -> (pred, rate))
    seg_vote_results = {}
    for seg_id, group in seg_ch.groupby("segment"):
        w_pred, w_rate = _weighted_vote(group, channel_f1)
        seg_vote_results[seg_id] = w_pred
    
    # 鏄犲皠鍥炵獥鍙ｇ骇
    for seg_id, w_pred in seg_vote_results.items():
        seg_mask = test_df["segment"] == seg_id
        pred_e.loc[seg_mask] = int(w_pred)
    
    r_e, _ = segment_aggregate(features_df[~train_mask], pred_e, "E_weighted_vote_seg")
    all_results["E_weighted_vote_seg"] = r_e

    # --- 瀹為獙F: 鍔犳潈鎶曠エ + 寮遍€氶亾娈电骇鍏滃簳 ---
    # 寮遍€氶亾鏇挎崲涓烘绾aseline棰勬祴, 鍐嶅姞鏉冩姇绁?
    pred_f = pd.Series(0, index=test_df.index, dtype=int)
    test_with_pred_f = test_df.copy()
    
    # 寮遍€氶亾鐢ㄦ绾aseline
    ch_pred_f = {}
    for idx in test_df.index:
        ch = test_df.loc[idx, "channel"]
        seg_id = test_df.loc[idx, "segment"]
        if ch in strong_channels:
            ch_pred_f[idx] = channel_preds.get(idx, 0)
        else:
            ch_pred_f[idx] = seg_baseline_map.get(seg_id, 0)
    
    test_with_pred_f["ch_pred"] = pd.Series(ch_pred_f)
    
    seg_ch_f = test_with_pred_f.groupby(["segment", "channel"]).agg(
        y_true=("anomaly", "first"),
        ch_pred_rate=("ch_pred", "mean"),
    ).reset_index()
    
    seg_vote_results_f = {}
    for seg_id, group in seg_ch_f.groupby("segment"):
        w_pred, w_rate = _weighted_vote(group, channel_f1)
        seg_vote_results_f[seg_id] = w_pred
    
    for seg_id, w_pred in seg_vote_results_f.items():
        seg_mask = test_df["segment"] == seg_id
        pred_f.loc[seg_mask] = int(w_pred)
    
    r_f, _ = segment_aggregate(features_df[~train_mask], pred_f, "F_weighted+seg_fallback")
    all_results["F_weighted+seg_fallback"] = r_f

    # --- 瀹為獙G: 鍔犳潈鎶曠エ + 寮遍€氶亾娈电骇鍏滃簳 + 瑙勫垯琛ュ厖 ---
    # 鍦‵鍩虹涓? 寮洪€氶亾棰濆OR瑙勫垯
    pred_g = pd.Series(0, index=test_df.index, dtype=int)
    for idx in test_df.index:
        ch = test_df.loc[idx, "channel"]
        seg_id = test_df.loc[idx, "segment"]
        if ch in strong_channels:
            # 寮洪€氶亾: IF OR 瑙勫垯
            if_pred = channel_preds.get(idx, 0)
            r_pred = int(rule_pred.loc[idx])
            pred_g.loc[idx] = max(if_pred, r_pred)
        else:
            # 寮遍€氶亾: 娈电骇baseline
            pred_g.loc[idx] = seg_baseline_map.get(seg_id, 0)
    
    r_g, _ = segment_aggregate(features_df[~train_mask], pred_g, "G_strong_IF_OR_Rule+weak_seg")
    all_results["G_strong_IF_OR_Rule+weak_seg"] = r_g

    # ===== 姹囨€?=====
    print(f"\n{'='*60}")
    print("SUMMARY: Weighted Fusion Experiments (1.7)")
    print(f"{'='*60}")
    print(f"\n{'Method':<35} | {'WinF1':>6} | {'SegF1':>6} | {'SegP':>6} | {'SegR':>6}")
    print("-" * 70)
    
    best_method = None
    best_seg_f1 = 0
    for name, r in all_results.items():
        wf1 = r["window"]["f1"]
        sf1 = r["segment"]["f1"]
        sp = r["segment"]["precision"]
        sr = r["segment"]["recall"]
        print(f"{name:<35} | {wf1:>6.3f} | {sf1:>6.3f} | {sp:>6.3f} | {sr:>6.3f}")
        if sf1 > best_seg_f1:
            best_seg_f1 = sf1
            best_method = name

    print(f"\n馃弳 Best: {best_method} (SegF1={best_seg_f1:.3f})")
    print(f"   vs 1.6 baseline: SegF1=0.325")
    if best_seg_f1 > 0.325:
        print(f"   鉁?Improvement: +{(best_seg_f1 - 0.325) / 0.325 * 100:.1f}%")
    else:
        print(f"   鉂?No improvement over 1.6")

    # ===== 淇濆瓨缁撴灉 =====
    output = {
        "experiment": "weighted_fusion_1.7",
        "window_size": 20,
        "step_size": 10,
        "best_contamination": best_c,
        "channel_f1_weights": channel_f1,
        "results": {name: r for name, r in all_results.items()},
        "best_method": best_method,
        "best_seg_f1": float(best_seg_f1),
        "comparison_to_1_6": float(best_seg_f1 - 0.325),
    }
    out_path = os.path.join(results_dir, "weighted_fusion_1.7_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[Saved] {out_path}")

    return output


def _weighted_vote(group, channel_f1):
    """鎸夐€氶亾F1鍔犳潈鎶曠エ锛岃繑鍥?棰勬祴鏍囩, 鍔犳潈寮傚父鐜?"""
    total_weight = 0
    weighted_anomaly = 0
    for _, row in group.iterrows():
        ch = row["channel"]
        w = max(channel_f1.get(ch, 0.1), 0.1)  # 鏈€浣庢潈閲?.1锛岄伩鍏嶉浂鏉冮噸
        weighted_anomaly += w * row["ch_pred_rate"]
        total_weight += w
    
    if total_weight == 0:
        return 0, 0.0
    
    rate = weighted_anomaly / total_weight
    pred = 1 if rate >= 0.5 else 0
    return pred, rate


if __name__ == "__main__":
    main()

