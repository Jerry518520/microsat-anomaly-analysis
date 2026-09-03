"""
v3 实验框架自检脚本 —— 跑通完整链路一次

链路：
    1. load_split() 拿到 fit / val / test
    2. 在 **fit** 上用 src/utils/stat_rules.py 拟合规则阈值（3σ + IQR）
    3. 在 **test** 上评估纯规则基线（违反特征数 >= 2 判异常）——这是最终评估，只跑这一次
    4. evaluate() 出全套指标（F1/P/R/MCC/混淆矩阵/bootstrap 95% CI）
    5. save_result(..., 'smoke_rule_only') 落盘
    6. 打印全部指标 + CI + 混淆矩阵 + 九通道明细

用法：
    python scripts/smoke_test_framework.py
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

np.random.seed(42)  # 纲领 1.2：脚本开头设全局种子

from src.experiments.framework import (  # noqa: E402
    evaluate,
    evaluate_per_channel,
    format_metrics,
    load_split,
    save_result,
)
from src.utils.stat_rules import apply_stat_rules, compute_stat_thresholds  # noqa: E402

RULE_K = 2  # 违反特征数 >= 2 判异常
REFERENCE_F1 = 0.5873  # 此前实测的纯规则(k=2)参考值（划分口径不同，不要求逐位相同）
TOLERANCE_OK = 0.02  # 偏差 <= 0.02 属正常
TOLERANCE_STOP = 0.05  # 偏差 > 0.05 必须停下来报告，不得硬凑


def main():
    print("=" * 74)
    print("v3 框架自检 —— Rule-only 基线（纯 3σ/IQR 规则，零模型）")
    print("=" * 74)

    # ---------- 1. 加载划分 ----------
    fit_df, val_df, test_df, feature_cols = load_split()
    print(f"\n[1] load_split() 成功")
    print(f"    fit  ={len(fit_df):5d}  异常={int(fit_df['anomaly'].sum()):4d}  "
          f"异常率={fit_df['anomaly'].mean():.4f}")
    print(f"    val  ={len(val_df):5d}  异常={int(val_df['anomaly'].sum()):4d}  "
          f"异常率={val_df['anomaly'].mean():.4f}")
    print(f"    test ={len(test_df):5d}  异常={int(test_df['anomaly'].sum()):4d}  "
          f"异常率={test_df['anomaly'].mean():.4f}")
    print(f"    feature_cols = {len(feature_cols)} 个")

    # ---------- 2. 在 fit 上拟合规则阈值 ----------
    # 注意：阈值只在 fit_df 上拟合，全程不碰 test_df
    thresholds = compute_stat_thresholds(fit_df, feature_cols)
    n_channels = len(thresholds)
    print(f"\n[2] compute_stat_thresholds(fit_df) 完成，覆盖 {n_channels} 个通道")

    # ---------- 3. 在 test 上评估（最终评估，只此一次） ----------
    n_violated, n_total = apply_stat_rules(test_df, thresholds, feature_cols)
    y_pred = (n_violated >= RULE_K).astype(int).to_numpy()
    y_true = test_df["anomaly"].to_numpy()

    print(f"[3] apply_stat_rules(test_df) 完成，判据：违反特征数 >= {RULE_K}")
    print(f"    预测异常段数 = {int(y_pred.sum())} / 真值异常段数 = {int(y_true.sum())}")

    # ---------- 4. 评估 ----------
    metrics = evaluate(y_true, y_pred, n_boot=1000, seed=42)
    print(f"\n[4] evaluate() —— Rule-only 在 test 上的指标")
    print(format_metrics(metrics))

    # ---------- 5. 九通道明细 ----------
    per_ch = evaluate_per_channel(test_df, y_pred)
    print(f"\n[5] 九通道明细（test，判据 k={RULE_K}）")
    print(f"    {'channel':<10} {'n':>4} {'anom':>5} {'F1':>8} {'P':>8} {'R':>8} {'TP':>4} {'FP':>4} {'FN':>4}  备注")
    print("    " + "-" * 72)
    for ch, m in sorted(per_ch.items()):
        note = "n 过小，不具统计意义" if m.get("small_sample_warning") else ""
        print(f"    {ch:<10} {m['n']:>4} {m['n_anomaly_true']:>5} {m['f1']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}  {note}")

    # ---------- 6. 落盘 ----------
    extra = {
        "method": "Rule-only",
        "method_code": "Rule-only",
        "rule_k": RULE_K,
        "rule_type": "3-sigma OR IQR, per-channel",
        "threshold_fit_on": "fit_df (官方 train 的 80%)",
        "evaluated_on": "test_df (官方 test, 529)",
        "feature_count": len(feature_cols),
        "n_channels_with_thresholds": n_channels,
        "reference_f1_previous_measurement": REFERENCE_F1,
        "f1_deviation_from_reference": round(metrics["f1"] - REFERENCE_F1, 6),
        "per_channel": per_ch,
        "split_counts": {
            "fit": len(fit_df), "val": len(val_df), "test": len(test_df),
        },
        "note": "自检产物，非正式论文数字；阈值在 fit 上拟合，未使用 test 做任何参数选择",
    }
    path = save_result(metrics, "smoke_rule_only", extra=extra)

    # ---------- 7. 与参考值对比 ----------
    dev = metrics["f1"] - REFERENCE_F1
    print(f"\n[7] 与参考值对比")
    print(f"    本次 F1 = {metrics['f1']:.4f}   参考 F1 = {REFERENCE_F1:.4f}   偏差 = {dev:+.4f}")
    if abs(dev) <= TOLERANCE_OK:
        print(f"    偏差 <= {TOLERANCE_OK}，属正常（划分口径已变更，不要求逐位相同）")
    elif abs(dev) <= TOLERANCE_STOP:
        print(f"    [注意] 偏差在 {TOLERANCE_OK}~{TOLERANCE_STOP} 之间，超出正常范围，建议人工复核")
    else:
        print(f"    [!!停止信号!!] 偏差 > {TOLERANCE_STOP}，请停下来报告，不要硬凑数字")

    print(f"\n[完成] 结果文件: {os.path.relpath(path, PROJECT_ROOT)}")
    print("=" * 74)


if __name__ == "__main__":
    main()
