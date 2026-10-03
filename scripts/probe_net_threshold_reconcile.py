"""探针：核验「净阈值规则层 F1」的真实值，定位与两个 agent 报告的分歧。

分歧现象：
  algo-diagnose 报：规则层 test F1（全量阈值）= 0.5882，净阈值 = 0.6643，Δ=+0.0762
  prod-gap       报：同上（复现 0.664286）
  但主理人实测：配置C（净阈值 + IF18维 + 门控重选）只有 0.5741，反而低于基线 0.6182

三者不可能同时成立。本脚本用最简口径逐层拆解，只测规则层，不牵扯 IF：
  口径 1：全量阈值 + nv>=2                    -> 应得 0.5882
  口径 2：净阈值 + nv>=2                      -> 争议点，0.6643 还是别的？
  口径 3：净阈值 + k 在 val 上重选（1/2/3）    -> 若 k 被选成别的，口径2不成立
  口径 4：净阈值 + 逐通道选 k
并打印 nv 分布对比，说明净阈值把违规数压到了什么量级。

同时核验 prod-gap 的护栏（正常段<30 退回全量）是否真的零成本。

纪律：阈值只在 fit 上拟合；k 只在 val 上选；test 只在口径定死后评一次。

用法：
    /d/Python313/python.exe scripts/probe_net_threshold_reconcile.py
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments.framework import load_split  # noqa: E402
from src.utils.stat_rules import (  # noqa: E402
    apply_stat_rules,
    compute_stat_thresholds,
)

MIN_NORMAL = 30


def qf1(yt, yp):
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(yt), np.asarray(yp), zero_division=0))


def net_thresholds(fit_df, feature_cols, min_normal=MIN_NORMAL):
    """净阈值：只用 fit 内 anomaly==0 的段拟合；正常段不足则退回全量。

    注意：compute_stat_thresholds 返回 {channel: {feature: {...}}}，
    因此必须一次性传入「所有通道的正常段」，再逐通道取；不能逐通道传入
    单通道子集，那样会得到 {ch: {ch: {...}}} 的双层嵌套，
    apply_stat_rules 里 `col not in ch_th` 恒真 -> 全部特征被跳过 -> nv 恒为 0。
    """
    out = {}
    for ch in fit_df["channel"].unique():
        sub = fit_df[fit_df["channel"] == ch]
        normal = sub[sub["anomaly"] == 0]
        if len(normal) >= min_normal:
            one = compute_stat_thresholds(normal, feature_cols)
            out[ch] = one[ch]          # <- 取内层，修正双层嵌套
        else:
            out[ch] = compute_stat_thresholds(sub, feature_cols)[ch]
    return out


def main():
    fit_df, val_df, test_df, feature_cols = load_split()

    th_full = compute_stat_thresholds(fit_df, feature_cols)
    th_net = net_thresholds(fit_df, feature_cols)

    # ---- 1. nv 分布对比：净阈值把违规数压到多少 --------------------
    print("=" * 84)
    print("1. nv 分布对比（test 段级）")
    print("=" * 84)
    nv_f, _ = apply_stat_rules(test_df, th_full, feature_cols)
    nv_n, _ = apply_stat_rules(test_df, th_net, feature_cols)
    yt = test_df["anomaly"].to_numpy()
    print(f"{'nv':>4}{'全量:段数':>11}{'全量:异常':>11}{'全量F1':>9}"
          f"{'净:段数':>10}{'净:异常':>10}{'净F1':>9}")
    print("-" * 64)
    for t in range(0, 10):
        mf, mn = (nv_f == t).to_numpy(), (nv_n == t).to_numpy()
        f_f1 = qf1(yt, (nv_f >= t).astype(int).to_numpy()) if t > 0 else 0.0
        f_n1 = qf1(yt, (nv_n >= t).astype(int).to_numpy()) if t > 0 else 0.0
        print(f"{t:>4}{int(mf.sum()):>11}{int(yt[mf].sum()):>11}{f_f1:>9.4f}"
              f"{int(mn.sum()):>10}{int(yt[mn].sum()):>10}{f_n1:>9.4f}")
    print(f"{'>=10':>4}{int((nv_f >= 10).sum()):>11}{int(yt[(nv_f >= 10).to_numpy()].sum()):>11}"
          f"{qf1(yt, (nv_f >= 10).astype(int).to_numpy()):>9.4f}"
          f"{int((nv_n >= 10).sum()):>10}{int(yt[(nv_n >= 10).to_numpy()].sum()):>10}"
          f"{qf1(yt, (nv_n >= 10).astype(int).to_numpy()):>9.4f}")

    # ---- 2. 四个口径的规则层 test F1 --------------------------------
    print()
    print("=" * 84)
    print("2. 规则层 test F1：四个口径逐个验")
    print("=" * 84)
    yv = val_df["anomaly"].to_numpy()
    nv_vf, _ = apply_stat_rules(val_df, th_full, feature_cols)
    nv_vn, _ = apply_stat_rules(val_df, th_net, feature_cols)

    # 口径 1：全量 + 固定 k=2
    f1_1 = qf1(yt, (nv_f >= 2).astype(int).to_numpy())
    print(f"  口径1  全量阈值 + 固定 nv>=2          test F1 = {f1_1:.4f}   (应= 0.5882)")

    # 口径 2：净阈值 + 固定 k=2
    f1_2 = qf1(yt, (nv_n >= 2).astype(int).to_numpy())
    print(f"  口径2  净阈值   + 固定 nv>=2          test F1 = {f1_2:.4f}   (争议点)")

    # 口径 3：净阈值 + k 在 val 重选
    best_k, bk = None, -1.0
    for k in (1, 2, 3, 4):
        f1v = qf1(yv, (nv_vn >= k).astype(int).to_numpy())
        print(f"         val 上 k={k}: 净阈值 val F1 = {f1v:.4f}")
        if f1v > bk:
            bk, best_k = f1v, k
    f1_3 = qf1(yt, (nv_n >= best_k).astype(int).to_numpy())
    print(f"  口径3  净阈值   + val 重选 k={best_k}        test F1 = {f1_3:.4f}")

    # 口径 3b：全量 + k 在 val 重选（对照）
    bfk, bfkv = None, -1.0
    for k in (1, 2, 3, 4):
        f1v = qf1(yv, (nv_vf >= k).astype(int).to_numpy())
        if f1v > bfkv:
            bfkv, bfk = f1v, k
    f1_3b = qf1(yt, (nv_f >= bfk).astype(int).to_numpy())
    print(f"  口径3b 全量阈值 + val 重选 k={bfk}        test F1 = {f1_3b:.4f}")

    # 口径 4：净阈值 + 逐通道选 k（更激进）
    yt_all, yp_all = [], []
    picks = {}
    for ch in sorted(test_df["channel"].unique()):
        cv = val_df[val_df["channel"] == ch]
        ev = test_df[test_df["channel"] == ch]
        nvcv, _ = apply_stat_rules(cv, th_net, feature_cols)
        nvce, _ = apply_stat_rules(ev, th_net, feature_cols)
        bk2, bkf2 = None, -1.0
        for k in (1, 2, 3, 4):
            f1v = qf1(cv["anomaly"].to_numpy(), (nvcv >= k).astype(int).to_numpy())
            if f1v > bkf2:
                bkf2, bk2 = f1v, k
        picks[ch] = bk2
        yt_all.append(ev["anomaly"].to_numpy())
        yp_all.append((nvce >= bk2).astype(int).to_numpy())
    f1_4 = qf1(np.concatenate(yt_all), np.concatenate(yp_all))
    print(f"  口径4  净阈值   + 逐通道选 k            test F1 = {f1_4:.4f}")
    print(f"         逐通道 k: " + ", ".join(f"{k[-4:]}={v}" for k, v in picks.items()))

    # ---- 3. 护栏是否零成本 -----------------------------------------
    print()
    print("=" * 84)
    print("3. 护栏检查：净阈值是否真的'只在正常段上拟合'")
    print("=" * 84)
    for ch in sorted(fit_df["channel"].unique()):
        sub = fit_df[fit_df["channel"] == ch]
        nn = int((sub["anomaly"] == 0).sum())
        src = "净阈值(仅正常段)" if nn >= MIN_NORMAL else "全量(护栏触发)"
        print(f"  {ch}: fit={len(sub):>4}  正常={nn:>4}  -> {src}")

    # ---- 4. 结论定位 -----------------------------------------------
    print()
    print("=" * 84)
    print("4. 分歧定位")
    print("=" * 84)
    print(f"  口径1 全量+nv>=2      = {f1_1:.4f}   与 fusion.json rule_only 0.5882 对齐: "
          f"{'是' if abs(f1_1-0.5882) < 5e-4 else '否'}")
    print(f"  口径2 净阈值+nv>=2    = {f1_2:.4f}   与 agent 报 0.6643 对齐: "
          f"{'是' if abs(f1_2-0.6643) < 5e-3 else '否'}")
    print(f"  口径3 净阈值+重选k={best_k}  = {f1_3:.4f}")
    print()
    print("  => 若口径 2 已得 0.6643，则「净阈值单点+0.076」成立；")
    print("     主理人配置 C 只得 0.5741 的原因就**不是净阈值无效**，")
    print("     而是门控在净阈值下选出的算子（if/AND/OR）在 test 上不占优——")
    print("     需单独核对门控重选那一步，不能据此否定净阈值。")


if __name__ == "__main__":
    main()
