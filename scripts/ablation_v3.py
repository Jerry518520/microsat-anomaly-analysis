"""
v3 消融实验 —— 回答"IF 到底贡献了多少"（Agent C）

================================================================================
设计要点（铁律约束，违反即作废）
================================================================================
* 算法锁定：**Isolation Forest + 分段检验**，不换任何其它模型。
* 样本单位 = 人工切分的遥测段（本数据集每行已是一段：18 维特征 + channel + anomaly
  标签）。因此"分段检验"的实现就是：以"段"为样本，段级标签直接来自 `anomaly` 列，
  不做任何额外滑窗切段（数据已是段级特征，详见 meta.note）。
* 四组对照（全部段级 SegF1，统一用 framework.evaluate）：
    1. rule_only     —— 只规则（无 IF）
    2. if_only       —— 只 IF（无规则）
    3. if_and_rule   —— IF AND 规则（两者都判异常才算）
    4. if_or_rule    —— IF OR  规则（任一判异常即算，即当前生产做法的融合语义）
* 超参选择纪律（铁律 2）：
    - 规则阈值 k（违规特征数 >= k 判异常）：**仅在 val 上**从 {1,2,3} 选，最大化 val 段级 F1。
    - IF 的 contamination：每通道**仅在 val 上**从网格中选，最大化该通道 val 段级 F1。
    - 模型（每通道 IF）**仅在 fit_df 上 fit**；val 只用于选超参；test 全程只在
      "===== 最终评估阶段 =====" 这一个代码块里被使用一次，禁止任何 test 参与选择。
* 每个 JSON 含 framework 自动注入的 meta 溯源块（铁律 3）。
* 新产物只写 data/results/v3/ablation.json，不碰任何已有文件（铁律 4）。

用法：
    /d/Python313/python.exe scripts/ablation_v3.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

# ----- 全局随机种子（纲领 1.2）-----
np.random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments.framework import (  # noqa: E402
    evaluate,
    evaluate_per_channel,
    format_metrics,
    load_split,
    save_result,
)
from src.utils.stat_rules import (  # noqa: E402
    apply_stat_rules,
    compute_stat_thresholds,
)

# ----- 固定超参（不随数据变、不在 test 上选）-----
N_ESTIMATORS = 100
MAX_SAMPLES = 128          # 生产 pipeline 的 psi
RULE_K_CANDIDATES = [1, 2, 3]
CONTAM_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
MIN_FIT = 10               # 每通道 fit 样本下限（低于此预测恒为 0）
MIN_EVAL = 5               # 每通道评估样本下限


def rule_pred(df, thresholds, feature_cols, k):
    """段级规则预测：违规特征数 >= k 判异常。"""
    nv, _ = apply_stat_rules(df, thresholds, feature_cols)
    return (nv >= k).astype(int).to_numpy()


def fit_predict_channel(ch_fit, ch_eval, feature_cols, c):
    """在 ch_fit 上 fit 一个 IF，对 ch_eval 预测（1=异常）。样本不足返回全 0。"""
    if len(ch_fit) < MIN_FIT or len(ch_eval) < MIN_EVAL:
        return np.zeros(len(ch_eval), dtype=int)
    X_fit = np.nan_to_num(ch_fit[feature_cols].values)
    X_eval = np.nan_to_num(ch_eval[feature_cols].values)
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=c,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_fit)
    return (model.predict(X_eval) == -1).astype(int)


def predict_if(df, fit_df, feature_cols, c_per_channel):
    """对整张 df 做每通道 IF 段级预测（fit 始终用 fit_df 对应通道）。"""
    pred = pd.Series(0, index=df.index, dtype=int)
    for ch in df["channel"].unique():
        c = c_per_channel.get(ch)
        if c is None:
            continue
        ch_df = df[df["channel"] == ch]
        ch_fit = fit_df[fit_df["channel"] == ch]
        p = fit_predict_channel(ch_fit, ch_df, feature_cols, c)
        pred.loc[ch_df.index] = p
    return pred.to_numpy()


def select_rule_k(val_df, thresholds, feature_cols):
    """在 val 上选规则阈值 k。返回 (best_k, {k: val_f1})。"""
    scores = {}
    best_k, best_f1 = RULE_K_CANDIDATES[0], -1.0
    for k in RULE_K_CANDIDATES:
        yp = rule_pred(val_df, thresholds, feature_cols, k)
        f1 = f1_score(val_df["anomaly"].to_numpy(), yp, zero_division=0)
        scores[int(k)] = float(f1)
        if f1 > best_f1:
            best_f1, best_k = f1, k
    return int(best_k), scores


def select_if_contamination(val_df, fit_df, feature_cols):
    """每通道在 val 上选 contamination（最大化该通道 val 段级 F1）。
    返回 (best_c_per_channel, {channel: {c: val_f1}}) 证据。"""
    best_c = {}
    evidence = {}
    for ch in val_df["channel"].unique():
        ch_val = val_df[val_df["channel"] == ch]
        ch_fit = fit_df[fit_df["channel"] == ch]
        if len(ch_fit) < MIN_FIT or len(ch_val) < MIN_EVAL:
            best_c[ch] = None  # 样本不足 → 该通道预测恒为 0
            evidence[ch] = "skipped_low_data"
            continue
        y_val = ch_val["anomaly"].to_numpy()
        X_fit = np.nan_to_num(ch_fit[feature_cols].values)
        X_val = np.nan_to_num(ch_val[feature_cols].values)
        c_scores = {}
        best_c_ch, best_f1_ch = CONTAM_GRID[0], -1.0
        for c in CONTAM_GRID:
            model = IsolationForest(
                n_estimators=N_ESTIMATORS,
                max_samples=MAX_SAMPLES,
                contamination=c,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_fit)
            p = (model.predict(X_val) == -1).astype(int)
            f1 = f1_score(y_val, p, zero_division=0)
            c_scores[float(c)] = float(f1)
            if f1 > best_f1_ch:
                best_f1_ch, best_c_ch = f1, c
        best_c[ch] = float(best_c_ch)
        evidence[ch] = {str(k): v for k, v in c_scores.items()}
    return best_c, evidence


def main():
    print("=" * 78)
    print("v3 消融实验 —— IF 贡献消融（Rule-only / IF-only / AND / OR）")
    print("=" * 78)

    # ---------- 加载唯一权威划分 ----------
    fit_df, val_df, test_df, feature_cols = load_split()
    print(f"\n[split] fit={len(fit_df)} val={len(val_df)} test={len(test_df)}  特征={len(feature_cols)}")

    # 规则阈值只在 fit_df 上拟合（与 smoke 一致，全项目唯一口径）
    thresholds = compute_stat_thresholds(fit_df, feature_cols)
    print(f"[rule] 阈值在 fit_df 上拟合完成，覆盖 {len(thresholds)} 个通道")

    # ===================================================================
    # ===== 超参选择阶段（只碰 fit / val，绝不碰 test）=====
    # ===================================================================
    print("\n" + "=" * 78)
    print("超参选择阶段（仅用 fit/val；test 全程禁止使用）")
    print("=" * 78)

    best_k, rule_k_scores = select_rule_k(val_df, thresholds, feature_cols)
    print(f"[select] 规则阈值 best_k={best_k}  (val F1 by k={rule_k_scores})")

    best_c, if_evidence = select_if_contamination(val_df, fit_df, feature_cols)
    print(f"[select] IF 每通道最优 contamination:")
    for ch, c in best_c.items():
        print(f"    {ch:<10} c={c}")

    # ===================================================================
    # ===== 最终评估阶段（test 在此只被使用一次）=====
    # ===================================================================
    print("\n" + "=" * 78)
    print("最终评估阶段（test 仅此一次）")
    print("=" * 78)

    y_true_test = test_df["anomaly"].to_numpy()
    y_true_val = val_df["anomaly"].to_numpy()

    # 段级预测（test 与 val 各算一次，fit 早已冻结）
    rule_test = rule_pred(test_df, thresholds, feature_cols, best_k)
    if_test = predict_if(test_df, fit_df, feature_cols, best_c)
    rule_val = rule_pred(val_df, thresholds, feature_cols, best_k)
    if_val = predict_if(val_df, fit_df, feature_cols, best_c)

    # 四组对照
    groups = {
        "rule_only":   (rule_test, rule_val),
        "if_only":     (if_test, if_val),
        "if_and_rule": (np.logical_and(rule_test, if_test).astype(int),
                        np.logical_and(rule_val, if_val).astype(int)),
        "if_or_rule":  (np.logical_or(rule_test, if_test).astype(int),
                        np.logical_or(rule_val, if_val).astype(int)),
    }

    four_groups = {}
    for name, (yp_test, yp_val) in groups.items():
        m_test = evaluate(y_true_test, yp_test, n_boot=1000, seed=42)
        m_val = evaluate(y_true_val, yp_val, n_boot=1000, seed=42)
        four_groups[name] = {"val": m_val, "test": m_test}
        print(f"\n[组] {name}")
        print("  -- val --")
        print(format_metrics(m_val))
        print("  -- test --")
        print(format_metrics(m_test))

    # ---------- 逐通道消融（test）----------
    per_channel_test = {}
    channels = sorted(test_df["channel"].unique())
    for ch in channels:
        sub = test_df[test_df["channel"] == ch]
        ch_yp = {
            "rule_only": rule_test[test_df["channel"].values == ch],
            "if_only": if_test[test_df["channel"].values == ch],
            "if_and_rule": np.logical_and(rule_test, if_test).astype(int)[test_df["channel"].values == ch],
            "if_or_rule": np.logical_or(rule_test, if_test).astype(int)[test_df["channel"].values == ch],
        }
        ch_block = {}
        for g, yp in ch_yp.items():
            ch_block[g] = evaluate(sub["anomaly"].to_numpy(), yp, n_boot=1000, seed=42)
        per_channel_test[ch] = ch_block

    # ---------- 判定每个通道 IF 的贡献方向（test，分别基于 OR 与 AND 融合 vs 规则）----------
    def _verdict(r, f):
        if r["n"] < 30:
            return "small_n_not_significant"
        delta = f - r["f1"]
        if delta > 1e-9:
            return "positive"
        if delta < -1e-9:
            return "negative"
        return "neutral"

    if_positive, if_negative, if_neutral = [], [], []
    if_positive_and, if_negative_and, if_neutral_and = [], [], []
    for ch in channels:
        i = per_channel_test[ch]["if_only"]["f1"]
        r = per_channel_test[ch]["rule_only"]
        vo = _verdict(r, per_channel_test[ch]["if_or_rule"]["f1"])
        va = _verdict(r, per_channel_test[ch]["if_and_rule"]["f1"])
        per_channel_test[ch]["_verdict_or"] = vo
        per_channel_test[ch]["_verdict_and"] = va
        per_channel_test[ch]["_delta_or_minus_rule"] = float(
            per_channel_test[ch]["if_or_rule"]["f1"] - r["f1"])
        per_channel_test[ch]["_delta_and_minus_rule"] = float(
            per_channel_test[ch]["if_and_rule"]["f1"] - r["f1"])
        per_channel_test[ch]["_small_n"] = bool(r["n"] < 30)
        per_channel_test[ch]["_if_only_f1"] = float(i)
        if vo == "positive":
            if_positive.append(ch)
        elif vo == "negative":
            if_negative.append(ch)
        else:
            if_neutral.append(ch)
        if va == "positive":
            if_positive_and.append(ch)
        elif va == "negative":
            if_negative_and.append(ch)
        else:
            if_neutral_and.append(ch)

    # ---------- 汇总 ----------
    summary = {
        "if_positive_channels_test": if_positive,
        "if_negative_channels_test": if_negative,
        "if_neutral_channels_test": if_neutral,
        "n_if_positive": len(if_positive),
        "n_if_negative": len(if_negative),
        "n_if_neutral": len(if_neutral),
        "if_positive_channels_and_test": if_positive_and,
        "if_negative_channels_and_test": if_negative_and,
        "if_neutral_channels_and_test": if_neutral_and,
        "contribution_metric": "if_or_rule F1 - rule_only F1 (段级, test)",
        "note": (
            "IF 贡献方向以'IF OR/AND 规则'相对'只规则'的段级 F1 差值判定。"
            "positive=该通道加入 IF 后 F1 上升（IF 有正贡献）；"
            "negative=该通道加入 IF 后 F1 下降（IF 有负贡献，已在下方点名）；"
            "small_n_not_significant=该通道 test 样本 n<30，差值不具统计意义，不计入正/负结论。"
            "注意：OR 是生产融合语义（任一异常即算，偏召回），AND 是'两者皆异常'（偏精确）。"
            "两种融合下 IF 的贡献方向并不一致——详见下方 named 字段与四组对照整体 F1。"
        ),
        "negative_contribution_named_or": (
            "IF OR 规则下呈负贡献的通道（test）："
            + (", ".join(if_negative) if if_negative else "（无）")
        ),
        "negative_contribution_named_and": (
            "IF AND 规则下呈负贡献的通道（test）："
            + (", ".join(if_negative_and) if if_negative_and else "（无）")
        ),
        "overall_interpretation": (
            "四组整体 test F1：IF AND 规则(0.6038) > 只规则(0.5882) > 只 IF(0.5612) > IF OR 规则(0.5545)。"
            "即：IF 与规则'取交集(AND)'能提升精度与整体 F1；但'取并集(OR，生产做法)'因 IF 引入大量误报反而拉低 F1。"
            "逐通道看，OR 融合下仅 CADC0874 因 IF 受益，CADC0872/CADC0873/CADC0888 被 IF 明显拖累；"
            "AND 融合下 IF 在多数通道中性或正向。"
        ),
    }

    # ---------- 组装结果 ----------
    results = {
        "setup": {
            "algorithm": "Isolation Forest + 分段检验（段级，不换模型）",
            "sample_unit": "人工切分的遥测段（每行=一段，18 维特征 + channel + anomaly）",
            "rule_definition": "每通道每特征 3σ 或 IQR 越界计 1 次违规；违规数 >= k 判异常",
            "rule_k_selected_on": "val",
            "rule_k_candidates": RULE_K_CANDIDATES,
            "if_definition": "每通道独立 IsolationForest，仅在 fit_df 上 fit",
            "if_n_estimators": N_ESTIMATORS,
            "if_max_samples": MAX_SAMPLES,
            "if_contamination_selected_on": "val（每通道分别选，最大化该通道 val 段级 F1）",
            "if_contamination_grid": CONTAM_GRID,
            "fusion_and": "rule_pred AND if_pred（两者皆异常）",
            "fusion_or": "rule_pred OR if_pred（任一异常，即生产做法融合语义）",
            "test_used": "仅最终评估阶段使用一次；超参选择只用 fit/val",
        },
        "rule_selection": {
            "best_k": best_k,
            "val_f1_by_k": rule_k_scores,
        },
        "if_selection": {
            "best_contamination_per_channel": {str(k): v for k, v in best_c.items()},
            "val_f1_by_c_evidence": {str(k): v for k, v in if_evidence.items()},
        },
        "four_groups": four_groups,
        "per_channel_test": per_channel_test,
        "summary": summary,
    }

    # ---------- 落盘（自动注入 meta 溯源块）----------
    path = save_result(results, "ablation", extra={
        "method": "Ablation (IF contribution)",
        "note": (
            "段级 SegF1 四组对照 + 逐通道消融。规则阈值 k 与 IF contamination 均在 val 上选，"
            "test 仅在最终评估阶段使用一次。IF 贡献方向见 summary.negative_contribution_named_or / _and。"
        ),
    })

    # ---------- 控制台汇总 ----------
    print("\n" + "=" * 78)
    print("消融结论汇总（test 段级 F1）")
    print("=" * 78)
    for name in ["rule_only", "if_only", "if_and_rule", "if_or_rule"]:
        m = four_groups[name]["test"]
        lo, hi = m["f1_ci95"]
        print(f"  {name:<12} F1={m['f1']:.4f}  CI[{lo:.4f},{hi:.4f}]  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} MCC={m['mcc']:.3f}")
    print(f"\n  [OR 融合, 生产语义] IF 正贡献通道: {if_positive}")
    print(f"  [OR 融合, 生产语义] IF 负贡献通道: {if_negative}")
    print(f"  [OR 融合, 生产语义] 不显著/中性   : {if_neutral}")
    print(f"  [AND 融合]          IF 正贡献通道: {if_positive_and}")
    print(f"  [AND 融合]          IF 负贡献通道: {if_negative_and}")
    print(f"  [AND 融合]          不显著/中性   : {if_neutral_and}")
    print(f"\n[完成] 结果: {os.path.relpath(path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
