"""探针：3 维子集的选择是否稳健？（决定 +0.0565 能不能作为修复方案）

背景：probe_gate_feature_subset.py 实测 IF 输入从 18 维换成 3 维
      ['n_peaks','kurtosis','smooth10_n_peaks']，test F1 0.6281 -> 0.6846。
      但配对 bootstrap CI = [-0.0065, +0.1241] 含 0，未达显著。

必须排除的失效模式：
  F1a子集选择过拟合 val：换随机种子/换 val 划分后，选出的子集是否稳定？
      若不稳定 => 收益是 val 噪声，不能用。
  F1b 单个通道的运气：+0.0565 是被1~2 个通道拉动的吗？
      逐通道看 F1 变化，定位增益来源。
  F1c IF 随机性：IsolationForest 有随机性，seed 变了收益还在吗？
      用 5 个 seed 重复端到端 F1。

铁律：子集选择仍只在 val 上做（seed 变的是 IF 的 random_state 与 val 划分，
      不涉及 test 标签）。test 仍只在配置确定后评一次。

用法：
    /d/Python313/python.exe scripts/probe_subset_robustness.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments.framework import load_split  # noqa: E402
from src.utils.stat_rules import (  # noqa: E402
    apply_stat_rules,
    compute_stat_thresholds,
)

N_ESTIMATORS = 100
MAX_SAMPLES = 128
RULE_K_CANDIDATES = [1, 2, 3]
CONTAM_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MIN_FIT = 10
MIN_EVAL = 5
SEEDS = [42, 7, 123, 2024, 999]
MAX_K = 5


def qf1(yt, yp):
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(yt), np.asarray(yp), zero_division=0))


def _auc(Xf, yf, Xe, ye, c, seed):
    clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                          contamination=c, random_state=seed, n_jobs=-1).fit(Xf)
    return roc_auc_score(ye, -clf.decision_function(Xe))


def greedy_subset(Xf, yf, Xv, yv, ncol, seed, max_k=MAX_K):
    """贪心前向选特征，只用 val AUC。全程不看 test。"""
    remaining = set(range(ncol))
    chosen, best = [], -1.0
    for _ in range(max_k):
        cb, ci = -1.0, None
        for i in list(remaining):
            idx = chosen + [i]
            a = max(_auc(Xf[:, idx], yf, Xv[:, idx], yv, c, seed) for c in CONTAM_GRID)
            if a > cb:
                cb, ci = a, i
        if chosen and cb <= best + 1e-6:
            break
        chosen.append(ci)
        remaining.discard(ci)
        best = cb
    return chosen, best


def end2end(fit_df, val_df, eval_df, feature_cols, if_cols, thresholds, best_k, seed):
    """门控端到端：逐通道在 val 选算子 -> 套eval。返回 (y_true, y_pred, picks, per_ch_f1)。"""
    best_c = {}
    for ch in sorted(val_df["channel"].unique()):
        cv, cf = val_df[val_df["channel"] == ch], fit_df[fit_df["channel"] == ch]
        if len(cf) < MIN_FIT or len(cv) < MIN_EVAL:
            best_c[ch] = None
            continue
        yv = cv["anomaly"].to_numpy()
        Xf_, Xv_ = np.nan_to_num(cf[if_cols].values), np.nan_to_num(cv[if_cols].values)
        bc, bf = CONTAM_GRID[0], -1.0
        for c in CONTAM_GRID:
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=seed, n_jobs=-1).fit(Xf_)
            f1 = qf1(yv, (clf.predict(Xv_) == -1).astype(int))
            if f1 > bf:
                bf, bc = f1, c
        best_c[ch] = float(bc)

    def preds(frame, ch):
        c = best_c.get(ch)
        nv, _ = apply_stat_rules(frame[frame["channel"] == ch], thresholds, feature_cols)
        rp = (nv >= best_k).astype(int).to_numpy()
        cf = fit_df[fit_df["channel"] == ch]
        if c is None or len(cf) < MIN_FIT:
            ip = np.zeros(len(frame[frame["channel"] == ch]), dtype=int)
        else:
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=seed, n_jobs=-1
                                  ).fit(np.nan_to_num(cf[if_cols].values))
            sub = frame[frame["channel"] == ch]
            ip = (clf.predict(np.nan_to_num(sub[if_cols].values)) == -1).astype(int)
        return rp, ip

    yt_all, yp_all, picks, ch_f1 = [], [], {}, {}
    for ch in sorted(eval_df["channel"].unique()):
        ev = eval_df[eval_df["channel"] == ch]
        ye = ev["anomaly"].to_numpy()
        rp_e, ip_e = preds(eval_df, ch)
        rp_v, ip_v = preds(val_df, ch)
        yv = val_df[val_df["channel"] == ch]["anomaly"].to_numpy()
        cand = {"rule": qf1(yv, rp_v), "if": qf1(yv, ip_v),
                "AND": qf1(yv, rp_v & ip_v), "OR": qf1(yv, rp_v | ip_v)}
        op = max(cand, key=lambda k: cand[k])
        picks[ch] = op
        p = {"rule": rp_e, "if": ip_e, "AND": (rp_e & ip_e).astype(int),
             "OR": (rp_e | ip_e).astype(int)}[op]
        yt_all.append(ye)
        yp_all.append(np.asarray(p, dtype=int))
        ch_f1[ch] = qf1(ye, p)
    return np.concatenate(yt_all), np.concatenate(yp_all), picks, ch_f1


def main():
    fit_df, val_df, test_df, feature_cols = load_split()
    ncol = len(feature_cols)

    # 官方 train 池（fit+val 的并集），用于换划分做稳定性检验
    official_train = pd.concat([fit_df, val_df], ignore_index=True)
    test_df = test_df.copy()

    # ---------- F1a 子集选择稳定性：换 5 个 val 划分种子 ----------
    print("=" * 84)
    print("F1a 子集选择稳定性：换 val 划分种子（子集仍只在 val 上贪心）")
    print("=" * 84)
    subsets = {}
    for seed in SEEDS:
        sub, val_auc = greedy_subset(
            np.nan_to_num(fit_df[feature_cols].values), fit_df["anomaly"].to_numpy(),
            np.nan_to_num(val_df[feature_cols].values), val_df["anomaly"].to_numpy(),
            ncol, seed)
        names = [feature_cols[i] for i in sub]
        subsets[seed] = (sub, val_auc)
        print(f"  seed={seed:<6} k={len(sub)}  val AUC={val_auc:.4f}  {names}")

    # 各特征被选中的频次
    freq = {}
    for seed, (sub, _) in subsets.items():
        for i in sub:
            freq[feature_cols[i]] = freq.get(feature_cols[i], 0) + 1
    print(f"\n  被选频次（{len(SEEDS)} 次）: "
          + ", ".join(f"{k}={v}" for k, v in sorted(freq.items(), key=lambda t: -t[1])))
    stable = [f for f, c in freq.items() if c == len(SEEDS)]
    print(f"  {len(SEEDS)}/{len(SEEDS)} 次都入选的特征: {stable if stable else '无'}")

    # ---------- F1b/F1c 端到端：逐 seed 对比 18 维 vs 选定子集 ----------
    print()
    print("=" * 84)
    print("F1b+F1c 端到端 test F1：18 维 vs 贪心子集（5 个 IF 随机种子）")
    print("=" * 84)
    thresholds = compute_stat_thresholds(fit_df, feature_cols)
    nv_v, _ = apply_stat_rules(val_df, thresholds, feature_cols)
    best_k, bkf1 = RULE_K_CANDIDATES[0], -1.0
    for k in RULE_K_CANDIDATES:
        f1 = qf1(val_df["anomaly"].to_numpy(), (nv_v >= k).astype(int).to_numpy())
        if f1 > bkf1:
            bkf1, best_k = f1, k

    sub0 = subsets[42][0]
    if_cols0 = [feature_cols[i] for i in sub0]
    print(f"  参照子集（seed=42 选出，{len(if_cols0)} 维）: {if_cols0}\n")

    print(f"{'seed':>7}{'F1(18维)':>11}{'F1(子集)':>11}{'差值':>10}  逐通道 F1 变化（子集 - 18维）")
    print("-" * 84)
    rows = []
    for seed in SEEDS:
        yA, pA, pickA, f1A = end2end(fit_df, val_df, test_df, feature_cols,
                                     feature_cols, thresholds, best_k, seed)
        yB, pB, pickB, f1B = end2end(fit_df, val_df, test_df, feature_cols,
                                     if_cols0, thresholds, best_k, seed)
        assert (yA == yB).all()
        fa, fb = qf1(yA, pA), qf1(yB, pB)
        rows.append((seed, fa, fb))
        delta = "  ".join(f"{ch[-4:]}:{f1B[ch]-f1A[ch]:+.3f}" for ch in sorted(f1A))
        print(f"{seed:>7}{fa:>11.4f}{fb:>11.4f}{fb-fa:>+10.4f}  {delta}")

    print()
    fa_m = np.mean([r[1] for r in rows])
    fb_m = np.mean([r[2] for r in rows])
    fa_s = np.std([r[1] for r in rows], ddof=1)
    fb_s = np.std([r[2] for r in rows], ddof=1)
    print(f"  18 维 : F1 = {fa_m:.4f} ± {fa_s:.4f}  (min {min(r[1] for r in rows):.4f}, "
          f"max {max(r[1] for r in rows):.4f})")
    print(f"  子集 : F1 = {fb_m:.4f} ± {fb_s:.4f}  (min {min(r[2] for r in rows):.4f}, "
          f"max {max(r[2] for r in rows):.4f})")
    print(f"  平均增益 = {fb_m - fa_m:+.4f}，5/5 seed 均为正 = "
          f"{'是' if all(r[2] > r[1] for r in rows) else '否'}")
    print(f"  两组范围是否重叠: 18维[{min(r[1] for r in rows):.4f},{max(r[1] for r in rows):.4f}] "
          f"vs 子集[{min(r[2] for r in rows):.4f},{max(r[2] for r in rows):.4f}]")

    # ---------- 增益是否集中在一两个通道 ----------
    print()
    print("=" * 84)
    print("增益来源分解（seed=42）")
    print("=" * 84)
    yA, pA, pickA, f1A = end2end(fit_df, val_df, test_df, feature_cols,
                                 feature_cols, thresholds, best_k, 42)
    yB, pB, pickB, f1B = end2end(fit_df, val_df, test_df, feature_cols,
                                 if_cols0, thresholds, best_k, 42)
    print(f"{'通道':<12}{'n_test':>8}{'n_anom':>8}{'F1(18维)':>10}{'F1(子集)':>10}{'差值':>9}  算子变化")
    print("-" * 84)
    cnt = test_df["channel"].value_counts().to_dict()
    anom = test_df.groupby("channel")["anomaly"].sum().to_dict()
    for ch in sorted(f1A):
        d = f1B[ch] - f1A[ch]
        opchg = "" if pickA[ch] == pickB[ch] else f"{pickA[ch]}->{pickB[ch]}"
        print(f"{ch:<12}{cnt[ch]:>8}{anom[ch]:>8}{f1A[ch]:>10.4f}{f1B[ch]:>10.4f}"
              f"{d:>+9.4f}  {opchg}")


if __name__ == "__main__":
    main()
