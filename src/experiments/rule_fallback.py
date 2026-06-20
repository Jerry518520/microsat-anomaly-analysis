"""
确定性规则兜底模块 (1.6)
核心创新：将物理约束和数据分布特征转化为可解释规则，与IForest形成互补

三层规则体系：
1. 统计阈值规则（sigma/IQR）- 基于训练集正常样本分布
2. 时序连续性规则 - 连续N个窗口异常才确认，降低误报
3. 通道自适应融合 - 强通道用规则+IF融合，弱通道降级为段级baseline
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from src.utils.data_loader import load_config, get_project_root, load_segments
from src.utils.constants import META_COLS
from src.utils.stat_rules import compute_stat_thresholds, apply_stat_rules, sigma_rule, iqr_rule
from src.features.sliding_window import extract_all_sliding_features


# ============================================================
# 规则1: 统计阈值规则（3-sigma + IQR）
# 已迁移到 src/utils/stat_rules.py，通过 import 复用
# ============================================================


# ============================================================
# 规则2: 时序连续性规则
# ============================================================
def apply_temporal_consistency(segments_df, pred_by_segment):
    """
    连续N个窗口异常才确认（降低误报）
    实现方式：对segment级别做时序平滑
    """
    # 每个segment内按顺序对窗口预测做平滑
    smoothed = {}
    for seg_id, group in segments_df.groupby("segment"):
        seg_preds = group.sort_values("window_idx")["pred_iforest"].values if "window_idx" in group.columns else group["pred_iforest"].values
        # 简单滑动均值平滑
        window_size = 3
        smoothed_preds = np.convolve(seg_preds, np.ones(window_size)/window_size, mode='same')
        # 阈值0.5：平滑后≥0.5视为异常
        final = (smoothed_preds >= 0.5).astype(int)
        for i, local_idx in enumerate(group.index):
            smoothed[local_idx] = final[i] if i < len(final) else 0
    return pd.Series(smoothed)


# ============================================================
# 规则3: 通道自适应融合
# ============================================================
def fuse_iforest_and_rules(iforest_pred, rule_pred, channels, channel_quality):
    """
    融合IForest和规则预测
    强通道(CADC0872/73/74): IF正常+规则异常 → 标异常（降漏报）
    弱通道(CADC0892/88/94): 规则+段级baseline兜底
    双重一致 → 高置信确认
    """
    fused = pd.Series(-1, index=iforest_pred.index, dtype=int)

    for idx in iforest_pred.index:
        ch = channels.loc[idx, "channel"]
        if_pred = int(iforest_pred.loc[idx])
        r_pred = int(rule_pred.loc[idx]) if rule_pred.loc[idx] >= 0 else 0

        # 强通道：使用IForest结果，规则作为补充确认
        if channel_quality.get(ch, {}).get("strength") == "strong":
            # IF异常 → 异常（主判断）
            # IF正常 + 规则异常(≥2个特征违反) → 标记（降漏报）
            if if_pred == 1:
                fused.loc[idx] = 1
            else:
                # IF认为正常，但规则认为异常
                fused.loc[idx] = r_pred
        else:
            # 弱通道：使用规则结果（IForest效果差）
            fused.loc[idx] = r_pred

    return fused


# ============================================================
# 段级聚合评估
# ============================================================
def segment_aggregate_eval(features_df, y_pred_window, label="Method"):
    """窗口级预测聚合为段级（多数投票），输出完整指标"""
    test_mask = features_df["train"] == 0
    test_df = features_df[test_mask].copy()
    test_df["pred"] = y_pred_window.values if hasattr(y_pred_window, 'values') else y_pred_window

    seg = test_df.groupby("segment").agg(
        y_true=("anomaly", "first"),
        y_pred=("pred", lambda x: (x.mean() >= 0.5).astype(int)),
        pred_rate=("pred", "mean"),
        channel=("channel", "first"),
        n_windows=("pred", "count"),
    ).reset_index()

    seg_f1 = f1_score(seg["y_true"], seg["y_pred"])
    seg_p = precision_score(seg["y_true"], seg["y_pred"], zero_division=0)
    seg_r = recall_score(seg["y_true"], seg["y_pred"], zero_division=0)
    try:
        seg_auc = roc_auc_score(seg["y_true"], seg["pred_rate"])
    except ValueError:
        seg_auc = None

    # 窗口级指标
    y_test_win = test_df["anomaly"].values
    y_pred_win = test_df["pred"].values
    win_f1 = f1_score(y_test_win, y_pred_win)
    win_p = precision_score(y_test_win, y_pred_win, zero_division=0)
    win_r = recall_score(y_test_win, y_pred_win, zero_division=0)

    print(f"  [{label}] Window F1={win_f1:.3f} P={win_p:.3f} R={win_r:.3f}")
    auc_str = f"{seg_auc:.3f}" if seg_auc is not None else "N/A"
    print(f"  [{label}] Segment F1={seg_f1:.3f} P={seg_p:.3f} R={seg_r:.3f} AUC={auc_str}")

    return {
        "window": {"f1": win_f1, "precision": win_p, "recall": win_r},
        "segment": {"f1": seg_f1, "precision": seg_p, "recall": seg_r, "auc": seg_auc},
    }, seg


# ============================================================
# 主实验
# ============================================================
def main():
    config = load_config()
    root = get_project_root()
    segments_df = load_segments(config)
    results_dir = os.path.join(root, config["data"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: 提取特征 (ws=20, 已确认最优)
    print("=" * 60)
    print("Step 1: Feature extraction (window=20)")
    print("=" * 60)
    features_df = extract_all_sliding_features(segments_df, window_size=20, step_size=10)
    feature_cols = [c for c in features_df.columns if c not in META_COLS]

    train_mask = features_df["train"] == 1
    X_train = np.nan_to_num(features_df.loc[train_mask, feature_cols].values)
    y_train = features_df.loc[train_mask, "anomaly"].values
    X_test = np.nan_to_num(features_df.loc[~train_mask, feature_cols].values)
    y_test = features_df.loc[~train_mask, "anomaly"].values

    train_df = features_df[train_mask]
    test_df = features_df[~train_mask].copy()

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Features: {len(feature_cols)}")

    # Step 2: 训练 IForest (c=0.5，已确认全局最佳)
    print("\n" + "=" * 60)
    print("Step 2: IForest (c=0.5)")
    print("=" * 60)
    if_model = IsolationForest(n_estimators=100, contamination=0.5,
                               random_state=42, n_jobs=-1)
    if_model.fit(X_train)
    if_pred = pd.Series((if_model.predict(X_test) == -1).astype(int), index=test_df.index)
    if_scores = -if_model.decision_function(X_test)
    test_df["pred_iforest"] = if_pred

    result_iforest, _ = segment_aggregate_eval(features_df[~train_mask], if_pred, "IForest")

    # Step 3: 统计阈值规则
    print("\n" + "=" * 60)
    print("Step 3: Statistical threshold rules")
    print("=" * 60)
    thresholds = compute_stat_thresholds(train_df, feature_cols)

    # 保存阈值参数（可解释性核心）
    thresh_out = {}
    for ch, ch_th in thresholds.items():
        thresh_out[ch] = {}
        for feat, vals in ch_th.items():
            thresh_out[ch][feat] = {
                "mean": vals["mean"],
                "sigma": vals["std"],
                "q1": vals["q1"],
                "q3": vals["q3"],
                "iqr": vals["iqr"],
                "sigma_lo": vals["mean"] - 3 * vals["std"],
                "sigma_hi": vals["mean"] + 3 * vals["iqr"],
                "iqr_lo": vals["q1"] - 1.5 * vals["iqr"],
                "iqr_hi": vals["q3"] + 1.5 * vals["iqr"],
            }

    n_violated, n_total = apply_stat_rules(test_df, thresholds, feature_cols)
    test_df["rule_violations"] = n_violated
    test_df["rule_total"] = n_total

    # 规则预测：违反特征数≥总特征数的20%视为异常
    rule_threshold_pct = 0.2
    rule_pred = pd.Series((n_violated / n_total.replace(0, 1) >= rule_threshold_pct).astype(int), index=test_df.index)

    result_rules, _ = segment_aggregate_eval(features_df[~train_mask], rule_pred, "Rule Only")

    # Step 4: 融合实验
    print("\n" + "=" * 60)
    print("Step 4: IForest + Rules fusion experiments")
    print("=" * 60)

    # 通道质量评估（基于已知分通道F1）
    channel_quality = {
        "CADC0872": {"strength": "strong", "if_f1": 0.509},
        "CADC0873": {"strength": "strong", "if_f1": 0.474},
        "CADC0874": {"strength": "strong", "if_f1": 0.591},
        "CADC0886": {"strength": "weak",   "if_f1": 1.000},
        "CADC0888": {"strength": "weak",   "if_f1": 0.296},
        "CADC0890": {"strength": "weak",   "if_f1": 0.500},
        "CADC0892": {"strength": "weak",   "if_f1": 0.134},
        "CADC0894": {"strength": "weak",   "if_f1": 0.284},
    }

    # 实验A: IF正常+规则异常→异常（降低漏报）
    fusion_a = pd.Series(0, index=test_df.index, dtype=int)
    fusion_a[(if_pred == 1)] = 1  # IF标异常 → 异常
    fusion_a[(if_pred == 0) & (rule_pred == 1)] = 1  # IF正常但规则异常 → 也标异常
    result_fusion_a, _ = segment_aggregate_eval(features_df[~train_mask], fusion_a, "Fusion-A (IF_normal+Rule→Anomaly)")

    # 实验B: 强通道用IF，弱通道用规则
    fusion_b = pd.Series(0, index=test_df.index, dtype=int)
    for idx in test_df.index:
        ch = test_df.loc[idx, "channel"]
        cq = channel_quality.get(ch, {})
        if cq.get("strength") == "strong":
            fusion_b.loc[idx] = int(if_pred.loc[idx])
        else:
            fusion_b.loc[idx] = int(rule_pred.loc[idx])
    result_fusion_b, _ = segment_aggregate_eval(features_df[~train_mask], fusion_b, "Fusion-B (strong=IF, weak=Rule)")

    # 实验C: 时序连续性（segment内滑动均值）
    test_df["pred_iforest"] = if_pred
    smoothed_dict = {}
    for seg_id, group in test_df.groupby("segment"):
        preds = group["pred_iforest"].values.astype(float)
        n = len(preds)
        if n < 3:
            smoothed = preds  # 太短不平滑
        else:
            kernel = np.ones(3) / 3
            smoothed = np.convolve(preds, kernel, mode='same')
        for i, idx in enumerate(group.index):
            smoothed_dict[idx] = smoothed[i]
    smoothed = pd.Series(smoothed_dict)
    fusion_c_pred = (smoothed >= 0.5).astype(int)
    result_fusion_c, _ = segment_aggregate_eval(features_df[~train_mask], fusion_c_pred, "Fusion-C (Temporal smoothing)")

    # 实验D: IF + 规则双重一致才确认（降低误报）
    fusion_d = pd.Series(0, index=test_df.index, dtype=int)
    fusion_d[(if_pred == 1) & (rule_pred == 1)] = 1  # 双重一致
    # IF异常但规则不异常 → 连续性确认
    result_fusion_d, _ = segment_aggregate_eval(features_df[~train_mask], fusion_d, "Fusion-D (IF AND Rule)")

    # 实验E: IF OR Rule（最激进，降低漏报）
    fusion_e = pd.Series(0, index=test_df.index, dtype=int)
    fusion_e[(if_pred == 1) | (rule_pred == 1)] = 1
    result_fusion_e, _ = segment_aggregate_eval(features_df[~train_mask], fusion_e, "Fusion-E (IF OR Rule)")

    # Step 5: 最佳方案验证 (分通道独立最优c + 规则融合)
    print("\n" + "=" * 60)
    print("Step 5: Per-channel best c + rule fusion")
    print("=" * 60)
    best_per_channel_c = {
        "CADC0872": 0.5, "CADC0873": 0.5, "CADC0874": 0.5,
        "CADC0886": 0.2, "CADC0888": 0.45, "CADC0890": 0.3,
        "CADC0892": 0.5, "CADC0894": 0.5,
    }

    fusion_final = pd.Series(0, index=test_df.index, dtype=int)
    channel_final_metrics = {}
    for ch in sorted(test_df["channel"].unique()):
        ch_test = test_df[test_df["channel"] == ch]
        ch_train = train_df[train_df["channel"] == ch]
        if len(ch_train) < 10 or len(ch_test) < 5:
            continue

        ch_c = best_per_channel_c.get(ch, 0.5)
        X_tr = np.nan_to_num(ch_train[feature_cols].values)
        y_tr = ch_train["anomaly"].values
        X_te = np.nan_to_num(ch_test[feature_cols].values)
        y_te = ch_test["anomaly"].values

        # 训练IF
        model = IsolationForest(n_estimators=100, contamination=ch_c,
                               random_state=42, n_jobs=-1)
        model.fit(X_tr)
        ch_if_pred = pd.Series((model.predict(X_te) == -1).astype(int), index=ch_test.index)

        # 规则预测
        ch_rule_pred = pd.Series(
            (test_df.loc[ch_test.index, "rule_violations"] / test_df.loc[ch_test.index, "rule_total"].replace(0, 1)
             >= rule_threshold_pct).astype(int),
            index=ch_test.index
        )

        # 融合策略：强通道用IF，弱通道用规则
        cq = channel_quality.get(ch, {})
        if cq.get("strength") == "strong":
            fused = ch_if_pred
        else:
            fused = ch_rule_pred

        fusion_final.loc[ch_test.index] = fused.values
        ch_seg_f1 = f1_score(y_te, fused.values)
        channel_final_metrics[ch] = {"f1": float(ch_seg_f1), "best_c": ch_c}

    result_final, seg_final = segment_aggregate_eval(features_df[~train_mask], fusion_final, "Final (Per-ch best c + Fusion)")

    # ========== 汇总表格 ==========
    print(f"\n{'='*60}")
    print("SUMMARY: Rule Fallback Experiments (ws=20)")
    print(f"{'='*60}")
    print(f"\n{'Method':<30} | {'WinF1':>6} | {'SegF1':>6} | {'SegP':>6} | {'SegR':>6}")
    print("-" * 65)

    all_results = {
        "A_IForest_only":       result_iforest,
        "B_Rule_only":          result_rules,
        "C_Fusion_A":           result_fusion_a,
        "D_Fusion_B":           result_fusion_b,
        "E_Fusion_C_temporal":  result_fusion_c,
        "D_Fusion_D_and":       result_fusion_d,
        "E_Fusion_E_or":        result_fusion_e,
        "F_Final":              result_final,
    }

    for name, r in all_results.items():
        wl = r["window"]
        sl = r["segment"]
        label = name.split("_", 1)[1] if "_" in name else name
        print(f"{label:<30} | {wl['f1']:>6.3f} | {sl['f1']:>6.3f} | {sl['precision']:>6.3f} | {sl['recall']:>6.3f}")

    # 保存结果
    output = {
        "experiment": "rule_fallback",
        "window_size": 20,
        "results_summary": {name: {
            "window": r["window"],
            "segment": r["segment"],
        } for name, r in all_results.items()},
        "per_channel_final_metrics": channel_final_metrics,
        "thresholds": thresh_out,
        "best_method": max(all_results.keys(), key=lambda k: all_results[k]["segment"]["f1"]),
    }

    out_path = os.path.join(results_dir, "rule_fallback_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    # 保存高置信异常案例（用于人工复核）
    test_df["pred_final"] = fusion_final.values
    test_df["if_scores"] = if_scores
    high_conf = test_df.nlargest(20, "if_scores")[
        ["channel", "segment", "anomaly", "pred_final", "if_scores", "rule_violations"]
    ].to_dict("records")
    hc_path = os.path.join(results_dir, "high_confidence_cases.json")
    with open(hc_path, "w", encoding="utf-8") as f:
        json.dump(high_conf, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Results: {out_path}")
    print(f"[Saved] High-confidence cases: {hc_path}")


if __name__ == "__main__":
    main()
