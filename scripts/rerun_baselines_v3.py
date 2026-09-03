"""
v3 基线重跑 —— Agent B 产出论文主表四行（Rule-only / IF-global / IF-perchannel / IF+Rule）

========================================================================
铁律（违反即作废）
  1. 数字只能由脚本产出（本文件即唯一来源，最终落盘到 data/results/v3/*.json）
  2. 超参（规则 k、IF contamination c）**只在 val 上选**；test 全程只碰一次
  3. 每个 JSON 都通过 framework.save_result() 注入 meta 溯源块
  4. 不删除/覆盖 data/results/ 下任何现有文件；只写 data/results/v3/
  5. 跑不出/未验证的写"未验证"，不利数字照实报

数据口径（来自 framework.load_split）：
  - 每行 = 一个人工切分并标注的遥测段（segment），`anomaly` 列即段级标签，
    `channel` 列标明该段所属通道（共 9 通道）。
  - fit (1275) 上 fit 模型/拟合规则阈值；val (319) 上选超参；
    test (529) 上最终评估，只允许被触碰一次。
  - **分段检验（segment-level）**：样本单位=段，评估=段级 F1（直接用 `anomaly` 标签）。

global vs per-channel 的切分决定（写入每个 JSON 的 meta.note）：
  - global IF：把所有通道的段混在一起，用 18 维特征拟合「单一」IsolationForest
    （忽略通道结构，所有段共享一个模型）。
  - per-channel IF：在每个通道内部单独拟合一个 IsolationForest（同一组 18 维特征），
    尊重通道结构；每个通道的 contamination 在该通道的 val 段上单独扫描选取。
  - IF+Rule：分通道 IF 预测 ∪ 规则预测（段级 OR 融合），规则用 3σ/IQR 阈值，
    阈值在 fit 上拟合、违规计数阈值 k 在 val 上选取。

用法：
    python scripts/rerun_baselines_v3.py
========================================================================
"""

import os
import sys

import numpy as np

# ---- 路径 ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

np.random.seed(42)  # 纲领 1.2：脚本开头设全局种子

from sklearn.ensemble import IsolationForest  # noqa: E402

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

RANDOM_STATE = 42
N_ESTIMATORS = 200

# 规则基线超参候选：段内"被判违规的特征数" >= k 才判该段异常
RULE_K_CANDIDATES = [1, 2, 3]

# IF contamination 扫描网格（val 上选）
CONTAM_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# IF+Rule 融合方式
FUSION = "OR"  # 段级 OR：规则或分通道 IF 任一判异常即判异常


# ------------------------------------------------------------------ 工具
def _if_predict(clf, X):
    """IsolationForest 输出：-1 表示异常 -> 转成 1/0 标签。"""
    return (clf.predict(X) == -1).astype(int)


def _fit_if(X_fit, c):
    clf = IsolationForest(
        contamination=c, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
    )
    clf.fit(X_fit)
    return clf


# ------------------------------------------------------------------ 规则基线
def run_rule_only(fit_df, val_df, test_df, feature_cols):
    print("\n" + "=" * 74)
    print("方法 1/4  Rule-only（纯 3σ/IQR 规则，零模型，段级投票）")
    print("=" * 74)

    thresholds = compute_stat_thresholds(fit_df, feature_cols)
    n_channels = len(thresholds)
    print(f"[fit] 规则阈值在 fit 上拟合，覆盖 {n_channels} 个通道")

    # ---- 在 val 上选 k（违规计数阈值），这是规则唯一的超参 ----
    best_k, best_val_f1 = None, -1.0
    val_k_detail = {}
    for k in RULE_K_CANDIDATES:
        nv_val, _ = apply_stat_rules(val_df, thresholds, feature_cols)
        y_val = (nv_val >= k).astype(int).to_numpy()
        m = evaluate(val_df["anomaly"].to_numpy(), y_val)
        val_k_detail[k] = m["f1"]
        print(f"    k={k}: val F1={m['f1']:.4f}")
        if m["f1"] > best_val_f1:
            best_val_f1, best_k = m["f1"], k

    print(f"[选参] 在 val 上选出的 k = {best_k}  (val F1={best_val_f1:.4f})")

    # ---- val 评估（用选出的最佳 k）----
    nv_val, _ = apply_stat_rules(val_df, thresholds, feature_cols)
    y_val = (nv_val >= best_k).astype(int).to_numpy()
    val_metrics = evaluate(val_df["anomaly"].to_numpy(), y_val)
    print(format_metrics(val_metrics, "  Rule-only @ val"))

    # ---- test 评估（只碰这一次）----
    nv_test, _ = apply_stat_rules(test_df, thresholds, feature_cols)
    y_test = (nv_test >= best_k).astype(int).to_numpy()
    test_metrics = evaluate(test_df["anomaly"].to_numpy(), y_test)
    print(format_metrics(test_metrics, "  Rule-only @ test"))
    per_ch_test = evaluate_per_channel(test_df, y_test)

    extra = {
        "method": "Rule-only",
        "method_code": "Rule-only",
        "rule_type": "3-sigma OR IQR, per-channel",
        "hyperparam": "k (段内被判违规的特征数阈值)",
        "k_candidates": RULE_K_CANDIDATES,
        "best_k": int(best_k),
        "val_f1_at_best_k": float(best_val_f1),
        "val_k_detail": {str(k): float(v) for k, v in val_k_detail.items()},
        "rule_thresholds_fit_on": "fit_df (官方 train 的 80%)",
        "threshold_fit_on": "fit_df",
        "evaluated_on": "test_df (官方 test, 529)",
        "feature_count": len(feature_cols),
        "n_channels_with_thresholds": n_channels,
        "per_channel_test": per_ch_test,
        "small_sample_channels_test": [
            ch for ch, m in per_ch_test.items() if m.get("small_sample_warning")
        ],
        "note": (
            "纯规则基线：每通道拟合 3σ/IQR 阈值（fit），段内被判违规的特征数 >= k 判异常；"
            "k 在 val 上扫描选取。段级（segment-level）评估，样本单位=段，标签来自 anomaly 列。"
            "无模型。这是论文叙事基线。"
        ),
    }
    save_result(test_metrics, "rule_only", extra=extra)
    return best_k, thresholds


# ------------------------------------------------------------------ 全局 IF
def run_if_global(fit_df, val_df, test_df, feature_cols):
    print("\n" + "=" * 74)
    print("方法 2/4  IF-global（全局 Isolation Forest，18 维特征一起）")
    print("=" * 74)

    X_fit = fit_df[feature_cols].to_numpy()
    X_val = val_df[feature_cols].to_numpy()
    X_test = test_df[feature_cols].to_numpy()

    # ---- 在 val 上扫 contamination c ----
    best_c, best_val_f1, val_c_detail = None, -1.0, {}
    for c in CONTAM_GRID:
        clf = _fit_if(X_fit, c)
        y_val = _if_predict(clf, X_val)
        m = evaluate(val_df["anomaly"].to_numpy(), y_val)
        val_c_detail[float(c)] = m["f1"]
        print(f"    c={c:.2f}: val F1={m['f1']:.4f}")
        if m["f1"] > best_val_f1:
            best_val_f1, best_c = m["f1"], c

    print(f"[选参] 在 val 上选出的 c = {best_c:.2f}  (val F1={best_val_f1:.4f})")

    # ---- val 评估（用选出的最佳 c，重新 fit 一次以保证一致）----
    clf = _fit_if(X_fit, best_c)
    y_val = _if_predict(clf, X_val)
    val_metrics = evaluate(val_df["anomaly"].to_numpy(), y_val)
    print(format_metrics(val_metrics, "  IF-global @ val"))

    # ---- test 评估（只碰这一次）----
    y_test = _if_predict(clf, X_test)
    test_metrics = evaluate(test_df["anomaly"].to_numpy(), y_test)
    print(format_metrics(test_metrics, "  IF-global @ test"))
    per_ch_test = evaluate_per_channel(test_df, y_test)

    extra = {
        "method": "IF-global",
        "method_code": "IF-global",
        "model": "IsolationForest",
        "model_scope": "global (所有 18 维特征 + 所有通道混在一起，单一模型)",
        "n_estimators": N_ESTIMATORS,
        "hyperparam": "contamination c",
        "c_grid": [float(c) for c in CONTAM_GRID],
        "best_c": float(best_c),
        "val_f1_at_best_c": float(best_val_f1),
        "val_c_detail": {str(c): float(v) for c, v in val_c_detail.items()},
        "model_fit_on": "fit_df (所有通道 18 维特征)",
        "evaluated_on": "test_df (官方 test, 529)",
        "feature_count": len(feature_cols),
        "per_channel_test": per_ch_test,
        "small_sample_channels_test": [
            ch for ch, m in per_ch_test.items() if m.get("small_sample_warning")
        ],
        "note": (
            "global IF：把所有通道的段混在一起，用 18 维特征拟合单一 IsolationForest，"
            "忽略通道结构。contamination c 在 val 上扫描选取。段级评估。"
        ),
    }
    save_result(test_metrics, "if_global", extra=extra)


# ------------------------------------------------------------------ 分通道 IF
def run_if_perchannel(fit_df, val_df, test_df, feature_cols):
    print("\n" + "=" * 74)
    print("方法 3/4  IF-perchannel（分通道 Isolation Forest）")
    print("=" * 74)

    channels = sorted(fit_df["channel"].unique())
    per_ch_best_c = {}
    per_ch_val_f1 = {}
    y_val_all = np.zeros(len(val_df), dtype=int)
    y_test_all = np.zeros(len(test_df), dtype=int)

    for ch in channels:
        fit_ch = fit_df[fit_df["channel"] == ch]
        val_ch = val_df[val_df["channel"] == ch]
        test_ch = test_df[test_df["channel"] == ch]
        X_fit = fit_ch[feature_cols].to_numpy()
        X_val = val_ch[feature_cols].to_numpy()
        X_test = test_ch[feature_cols].to_numpy()

        best_c, best_val_f1, cd = None, -1.0, {}
        for c in CONTAM_GRID:
            if len(X_fit) == 0:
                break
            clf = _fit_if(X_fit, c)
            yv = _if_predict(clf, X_val)
            m = evaluate(val_ch["anomaly"].to_numpy(), yv)
            cd[float(c)] = m["f1"]
            if m["f1"] > best_val_f1:
                best_val_f1, best_c = m["f1"], c

        if best_c is None:
            print(f"    {ch}: 无 fit 数据，跳过")
            per_ch_best_c[ch] = None
            per_ch_val_f1[ch] = None
            continue

        per_ch_best_c[ch] = float(best_c)
        per_ch_val_f1[ch] = float(best_val_f1)
        print(f"    {ch}: val n={len(val_ch):3d} 选 c={best_c:.2f}  val F1={best_val_f1:.4f}")

        clf = _fit_if(X_fit, best_c)
        y_val_all[val_ch.index.to_numpy()] = _if_predict(clf, X_val)
        y_test_all[test_ch.index.to_numpy()] = _if_predict(clf, X_test)

    # ---- val 评估（用每通道最佳 c）----
    val_metrics = evaluate(val_df["anomaly"].to_numpy(), y_val_all)
    print(format_metrics(val_metrics, "  IF-perchannel @ val"))

    # ---- test 评估（只碰这一次）----
    test_metrics = evaluate(test_df["anomaly"].to_numpy(), y_test_all)
    print(format_metrics(test_metrics, "  IF-perchannel @ test"))
    per_ch_test = evaluate_per_channel(test_df, y_test_all)

    extra = {
        "method": "IF-perchannel",
        "method_code": "IF-perchannel",
        "model": "IsolationForest",
        "model_scope": "per-channel (每个通道单独拟合一个 IF，同组 18 维特征)",
        "n_estimators": N_ESTIMATORS,
        "hyperparam": "contamination c（按通道在各自 val 段上选取）",
        "c_grid": [float(c) for c in CONTAM_GRID],
        "val_f1": float(val_metrics["f1"]),
        "per_channel_best_c": {str(k): v for k, v in per_ch_best_c.items()},
        "per_channel_val_f1": {str(k): v for k, v in per_ch_val_f1.items()},
        "model_fit_on": "fit_df（按通道分别 fit）",
        "evaluated_on": "test_df (官方 test, 529)",
        "feature_count": len(feature_cols),
        "n_channels": len(channels),
        "per_channel_test": per_ch_test,
        "small_sample_channels_test": [
            ch for ch, m in per_ch_test.items() if m.get("small_sample_warning")
        ],
        "note": (
            "per-channel IF：在每个通道内部单独拟合 IsolationForest（同一组 18 维特征），"
            "尊重通道结构；每个通道的 contamination c 在该通道的 val 段上单独扫描选取。"
            "val/test 上的段级预测由各通道模型组合得到。段级评估。"
        ),
    }
    save_result(test_metrics, "if_perchannel", extra=extra)


# ------------------------------------------------------------------ IF + Rule 融合
def run_if_plus_rule(fit_df, val_df, test_df, feature_cols, rule_k, thresholds):
    print("\n" + "=" * 74)
    print(f"方法 4/4  IF+Rule（分通道 IF ∪ 规则融合，融合方式={FUSION}）")
    print("=" * 74)

    channels = sorted(fit_df["channel"].unique())

    # 规则预测（阈值已 fit 于 fit，k 已选于 val）
    def rule_pred(df):
        nv, _ = apply_stat_rules(df, thresholds, feature_cols)
        return (nv >= rule_k).astype(int).to_numpy()

    # 分通道 IF 预测（复用各通道在 val 上选出的最佳 c）
    per_ch_best_c = {}
    y_val_if = np.zeros(len(val_df), dtype=int)
    y_test_if = np.zeros(len(test_df), dtype=int)
    for ch in channels:
        fit_ch = fit_df[fit_df["channel"] == ch]
        val_ch = val_df[val_df["channel"] == ch]
        test_ch = test_df[test_df["channel"] == ch]
        X_fit = fit_ch[feature_cols].to_numpy()
        X_val = val_ch[feature_cols].to_numpy()
        X_test = test_ch[feature_cols].to_numpy()
        # 该通道 best_c 在 val 上选（与 run_if_perchannel 同口径）
        best_c, best_val_f1 = None, -1.0
        for c in CONTAM_GRID:
            if len(X_fit) == 0:
                break
            clf = _fit_if(X_fit, c)
            m = evaluate(val_ch["anomaly"].to_numpy(), _if_predict(clf, X_val))
            if m["f1"] > best_val_f1:
                best_val_f1, best_c = m["f1"], c
        if best_c is None:
            per_ch_best_c[ch] = None
            continue
        per_ch_best_c[ch] = float(best_c)
        clf = _fit_if(X_fit, best_c)
        y_val_if[val_ch.index.to_numpy()] = _if_predict(clf, X_val)
        y_test_if[test_ch.index.to_numpy()] = _if_predict(clf, X_test)

    y_val_rule = rule_pred(val_df)
    y_test_rule = rule_pred(test_df)

    # OR 融合（段级）
    y_val = ((y_val_if == 1) | (y_val_rule == 1)).astype(int)
    y_test = ((y_test_if == 1) | (y_test_rule == 1)).astype(int)

    val_metrics = evaluate(val_df["anomaly"].to_numpy(), y_val)
    print(format_metrics(val_metrics, "  IF+Rule @ val"))
    test_metrics = evaluate(test_df["anomaly"].to_numpy(), y_test)
    print(format_metrics(test_metrics, "  IF+Rule @ test"))
    per_ch_test = evaluate_per_channel(test_df, y_test)

    extra = {
        "method": "IF+Rule",
        "method_code": "IF+Rule",
        "fusion": FUSION,
        "fusion_desc": "段级 OR：规则(3σ/IQR)或分通道 IF 任一判异常即判异常",
        "rule_k": int(rule_k),
        "rule_thresholds_fit_on": "fit_df",
        "per_channel_best_c": {str(k): v for k, v in per_ch_best_c.items()},
        "val_f1": float(val_metrics["f1"]),
        "model": "IsolationForest (per-channel) OR 3-sigma/IQR rule",
        "n_estimators": N_ESTIMATORS,
        "evaluated_on": "test_df (官方 test, 529)",
        "feature_count": len(feature_cols),
        "per_channel_test": per_ch_test,
        "small_sample_channels_test": [
            ch for ch, m in per_ch_test.items() if m.get("small_sample_warning")
        ],
        "note": (
            "IF+Rule 融合：分通道 IsolationForest（每通道 contamination c 在 val 上选）"
            "与 3σ/IQR 规则（k 在 val 上选）做段级 OR 融合。段级评估。"
            "AND 融合与 OR 融合的对照见 Agent C 消融实验（scripts/ablation_v3.py）。"
        ),
    }
    save_result(test_metrics, "if_plus_rule", extra=extra)


# ------------------------------------------------------------------ 主流程
def main():
    print("=" * 74)
    print("v3 基线重跑 —— Agent B 主表四行（铁律：超参只在 val 选，test 只碰一次）")
    print("=" * 74)

    fit_df, val_df, test_df, feature_cols = load_split()
    print(f"\n[load_split] fit={len(fit_df)} val={len(val_df)} test={len(test_df)} "
          f"features={len(feature_cols)}")

    # 1. 规则基线（返回在 val 上选出的 k 与阈值，供 4 融合复用）
    best_k, thresholds = run_rule_only(fit_df, val_df, test_df, feature_cols)

    # 2. 全局 IF
    run_if_global(fit_df, val_df, test_df, feature_cols)

    # 3. 分通道 IF
    run_if_perchannel(fit_df, val_df, test_df, feature_cols)

    # 4. IF + Rule 融合
    run_if_plus_rule(fit_df, val_df, test_df, feature_cols, best_k, thresholds)

    print("\n" + "=" * 74)
    print("完成：4 个 JSON 已写入 data/results/v3/")
    print("  - rule_only.json")
    print("  - if_global.json")
    print("  - if_perchannel.json")
    print("  - if_plus_rule.json")
    print("=" * 74)


if __name__ == "__main__":
    main()
