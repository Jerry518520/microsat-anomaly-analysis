"""探针：18 维全用反而比单特征 n_peaks 差 —— 稀释效应还是切分机制问题？

前置发现（scripts/probe_if_scaling.py 实测）：
    IF + 18 维 raw       test AUC = 0.6557
    IF + 仅 n_peaks      test AUC = 0.8511   <- 少 17 维反而高 0.20
    IF + 18 维 Robust    test AUC = 0.6375   <- 归一化无救

两种互斥解释，本脚本用实验区分：
  H1「稀释」：IF 的随机切分被大量弱相关维度摊薄。特征子集应单调优于全集。
     验证：逐维子集贪心搜索 AUC 是否随维度数先升后降；IF 的 n_estimators
           增大能否补偿。
  H2「切分机制」：IF 的轴对齐切分在多维空间里被高幅值维度（如 gaps_squared
     均值 1063）主导，异常点与正常点的分离在该轴上不成立。
     验证：剔除高幅值维度后 AUC 是否回升；用分位数（而非均匀）切分是否更好。

铁律：所有子集/超参选择只在 val 上做，test 只在最终选定后评一次。
      贪心搜索过程中禁止查看 test AUC。

用法：
    /d/Python313/python.exe scripts/probe_if_dilution.py
"""

import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments.framework import load_split  # noqa: E402

N_ESTIMATORS = 100
MAX_SAMPLES = 128
CONTAM_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def if_auc(Xf, yf, Xe, ye, c, n_est=N_ESTIMATORS, max_s=MAX_SAMPLES):
    clf = IsolationForest(n_estimators=n_est, max_samples=max_s,
                          contamination=c, random_state=42, n_jobs=-1).fit(Xf)
    return roc_auc_score(ye, -clf.decision_function(Xe))


def best_c_val(Xf, yf, Xv, yv, n_est=N_ESTIMATORS):
    """污染率只在 val 上选，返回 (c*, val AUC)。"""
    bc, ba = CONTAM_GRID[0], -1.0
    for c in CONTAM_GRID:
        a = if_auc(Xf, yf, Xv, yv, c, n_est=n_est)
        if a > ba:
            ba, bc = a, c
    return bc, ba


def main():
    fit_df, val_df, test_df, feature_cols = load_split()
    Xf = np.nan_to_num(fit_df[feature_cols].values)
    yf = fit_df["anomaly"].to_numpy()
    Xv = np.nan_to_num(val_df[feature_cols].values)
    yv = val_df["anomaly"].to_numpy()
    Xt = np.nan_to_num(test_df[feature_cols].values)
    yt = test_df["anomaly"].to_numpy()
    N = len(feature_cols)

    # ---------- 0. 单特征 val/test AUC 全谱（先看清哪些维度有信息） ----
    print("=" * 84)
    print("0. 全部 18 个单特征各自给 IF 的判别力（污染率在 val 上选）")
    print("=" * 84)
    print(f"{'#':>3}{'feature':<20}{'c*':>6}{'val AUC':>10}{'test AUC':>10}")
    print("-" * 52)
    single = []
    for i, c in enumerate(feature_cols):
        bc, ba = best_c_val(Xf[:, [i]], yf, Xv[:, [i]], yv)
        ta = roc_auc_score(yt, -IsolationForest(
            n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES, contamination=bc,
            random_state=42, n_jobs=-1).fit(Xf[:, [i]]).decision_function(Xt[:, [i]]))
        single.append((c, bc, ba, ta))
        print(f"{i:>3}{c:<20}{bc:>6.2f}{ba:>10.4f}{ta:>10.4f}")

    top1 = max(single, key=lambda r: r[2])
    print(f"\nval 最强单特征 = {top1[0]}  val AUC={top1[2]:.4f}")
    print("【注意】只看 test AUC 排序是作弊，这里仅用于事后诊断标注。")

    # ---------- 1. H1 检验：贪心前向子集，val AUC 轨迹 ----------------
    print()
    print("=" * 84)
    print("1. H1 稀释假设：贪心前向选特征（只看 val AUC，test 不参与）")
    print("=" * 84)
    remaining = set(range(N))
    chosen = []
    best_val = -1.0
    history = []
    for step in range(1, 7):
        cand_best, cand_i = -1.0, None
        for i in list(remaining):
            idx = chosen + [i]
            _, va = best_c_val(Xf[:, idx], yf, Xv[:, idx], yv)
            if va > cand_best:
                cand_best, cand_i = va, i
        chosen.append(cand_i)
        remaining.discard(cand_i)
        history.append((list(chosen), cand_best))
        names = [feature_cols[j] for j in chosen]
        print(f"  +{feature_cols[cand_i]:<18} k={step}  val AUC={cand_best:.4f}   累计: {names}")

    # 最优子集的 test AUC（此时才允许看 test）
    best_hist = max(history, key=lambda t: t[1])
    bidx = best_hist[0]
    bc, _ = best_c_val(Xf[:, bidx], yf, Xv[:, bidx], yv)
    ta = roc_auc_score(yt, -IsolationForest(
        n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES, contamination=bc,
        random_state=42, n_jobs=-1).fit(Xf[:, bidx]).decision_function(Xt[:, bidx]))
    print(f"\n  val 最优子集 k={len(bidx)}: {[feature_cols[j] for j in bidx]}")
    print(f"  其 test AUC = {ta:.4f}   (18 维全集 test AUC = 0.6557)")

    # ---------- 2. 固定维度数的 val AUC 曲线（稀释是否单调） ----------
    print()
    print("=" * 84)
    print("2. val AUC 随特征数的曲线（贪心轨迹的紧凑视图）")
    print("=" * 84)
    for idx, va in history:
        bc2, _ = best_c_val(Xf[:, idx], yf, Xv[:, idx], yv)
        ta2 = roc_auc_score(yt, -IsolationForest(
            n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES, contamination=bc2,
            random_state=42, n_jobs=-1).fit(Xf[:, idx]).decision_function(Xt[:, idx]))
        print(f"  k={len(idx)}  val AUC={va:.4f}  test AUC={ta2:.4f}  (+1: {feature_cols[idx[-1]]})")

    # ---------- 3. H2 检验：剔除高幅值维度 ---------------------------
    print()
    print("=" * 84)
    print("3. H2 切分机制：去掉高幅值维度后是否回升")
    print("=" * 84)
    scale = {c: np.percentile(Xf[:, i], 99) for i, c in enumerate(feature_cols)}
    hi = sorted(scale, key=lambda c: -scale[c])[:5]
    lo_c = [c for c in feature_cols if c not in hi]
    hidx = [feature_cols.index(c) for c in lo_c]
    bc, va = best_c_val(Xf[:, hidx], yf, Xv[:, hidx], yv)
    ta = roc_auc_score(yt, -IsolationForest(
        n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES, contamination=bc,
        random_state=42, n_jobs=-1).fit(Xf[:, hidx]).decision_function(Xt[:, hidx]))
    print(f"  剔除最高幅值 5 维 {hi}")
    print(f"  剩余 13 维: val AUC={va:.4f}  test AUC={ta:.4f}")
    print(f"  对比 18 维全集 test AUC=0.6557 -> 增益 {ta - 0.6557:+.4f}")

    # ---------- 4. n_estimators 能否补偿稀释 --------------------------
    print()
    print("=" * 84)
    print("4. 提高 n_estimators 能否补偿（稀释若是采样不足所致应显著改善）")
    print("=" * 84)
    print(f"{'n_est':>8}{'18维 test AUC':>16}{'最优子集 test AUC':>20}")
    print("-" * 46)
    for n_est in [100, 300, 1000]:
        _, v_full = best_c_val(Xf, yf, Xv, yv, n_est=n_est)
        c1, _ = best_c_val(Xf, yf, Xv, yv, n_est=n_est)
        t_full = roc_auc_score(yt, -IsolationForest(
            n_estimators=n_est, max_samples=MAX_SAMPLES, contamination=c1,
            random_state=42, n_jobs=-1).fit(Xf).decision_function(Xt))
        bc2, _ = best_c_val(Xf[:, bidx], yf, Xv[:, bidx], yv, n_est=n_est)
        t_sub = roc_auc_score(yt, -IsolationForest(
            n_estimators=n_est, max_samples=MAX_SAMPLES, contamination=bc2,
            random_state=42, n_jobs=-1).fit(Xf[:, bidx]).decision_function(Xt[:, bidx]))
        print(f"{n_est:>8}{t_full:>16.4f}{t_sub:>20.4f}")

    # ---------- 5. max_samples 影响 -----------------------------------
    print()
    print("=" * 84)
    print("5. max_samples（子采样量）对 18 维的影响")
    print("=" * 84)
    for ms in [64, 128, 256, 512]:
        bc, _ = best_c_val(Xf, yf, Xv, yv, n_est=N_ESTIMATORS)
        clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=ms,
                              contamination=bc, random_state=42, n_jobs=-1).fit(Xf)
        print(f"  max_samples={ms:<6} test AUC={roc_auc_score(yt, -clf.decision_function(Xt)):.4f}")


if __name__ == "__main__":
    main()
