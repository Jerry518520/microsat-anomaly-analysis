"""探针（决定性）：若 IF 输入换成有效特征子集，8 种融合策略的排序是否翻转？

为什么必须做这个实验：
    fusion.json 里 8 种融合策略的排序（rule_only 0.5882 > if_only 0.5612 >
    gate 0.6281 ...）是在「IF 用全部 18 维」这条前提下算出来的。
    而 probe_if_dilution.py 实测：18 维里有 10 维 val AUC < 0.5（反向），
    IF 的 test AUC 被压到 0.6557；只用 3 维则升到 0.8673。

    若 IF 变强，**if_only 会大涨、门控的配方会重排**，
    那么论文里「规则层是主梁、纯 IF 弱于纯规则」这条核心结论可能不成立。

本脚本在同一口径下重算 8 种融合策略，两个 IF 输入配置：
    配置 甲：IF 用 18 维（复现 fusion.json，应逐位对上 0.6281/0.5612/...）
    配置 乙：IF 用 3 维 val 贪心子集
输出两种配置下 8 策略的 val/test F1 排序对比 + 结论是否翻转的判定。

纪律：
  - 全部超参（k、contamination、w、tau、门控算子）在 val 上选；
  - test 只在每种策略完全确定后评一次；
  - 特征子集沿用已验证的 ['n_peaks','kurtosis','smooth10_n_peaks']。

用法：
    /d/Python313/python.exe scripts/probe_all8_rerank.py
"""

import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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
W_GRID = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
TAU_GRID = [round(x, 3) for x in np.arange(0.0, 1.001, 0.05)]
TAU_GRID_CAL = [round(x, 3) for x in np.arange(0.1, 0.9001, 0.05)]
IF_SUBSET = ["n_peaks", "kurtosis", "smooth10_n_peaks"]


def qf1(yt, yp):
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(yt), np.asarray(yp), zero_division=0))


def rule_nv(df, th, fcols):
    nv, _ = apply_stat_rules(df, th, fcols)
    return nv.astype(float).to_numpy()


def run_all(fit_df, val_df, test_df, fcols, if_cols, seed=42):
    """在给定 IF 输入配置下重算 8 种融合策略。"""
    th = compute_stat_thresholds(fit_df, fcols)
    yv = val_df["anomaly"].to_numpy()
    yt = test_df["anomaly"].to_numpy()

    # --- k in val
    nvv = rule_nv(val_df, th, fcols)
    best_k, bk = RULE_K_CANDIDATES[0], -1.0
    for k in RULE_K_CANDIDATES:
        f1 = qf1(yv, (nvv >= k).astype(int))
        if f1 > bk:
            bk, best_k = f1, k

    # --- per-channel contamination in val
    best_c = {}
    for ch in sorted(val_df["channel"].unique()):
        cv, cf = val_df[val_df["channel"] == ch], fit_df[fit_df["channel"] == ch]
        if len(cf) < MIN_FIT or len(cv) < MIN_EVAL:
            best_c[ch] = None
            continue
        y = cv["anomaly"].to_numpy()
        Xf, Xv = np.nan_to_num(cf[if_cols].values), np.nan_to_num(cv[if_cols].values)
        bc, bf = CONTAM_GRID[0], -1.0
        for c in CONTAM_GRID:
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=seed, n_jobs=-1).fit(Xf)
            f1 = qf1(y, (clf.predict(Xv) == -1).astype(int))
            if f1 > bf:
                bf, bc = f1, c
        best_c[ch] = float(bc)

    def layers(df):
        """返回 (rule_pred, if_pred, if_score) 三个对齐的数组。"""
        rp = (rule_nv(df, th, fcols) >= best_k).astype(int)
        ip = np.zeros(len(df), dtype=int)
        sc = np.zeros(len(df))
        for ch in df["channel"].unique():
            m = (df["channel"] == ch).to_numpy()
            c = best_c.get(ch)
            cf = fit_df[fit_df["channel"] == ch]
            if c is None or len(cf) < MIN_FIT:
                continue
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=seed, n_jobs=-1
                                  ).fit(np.nan_to_num(cf[if_cols].values))
            X = np.nan_to_num(df.loc[m, if_cols].values)
            ip[m] = (clf.predict(X) == -1).astype(int)
            sc[m] = -clf.decision_function(X)
        return rp, ip, sc

    rv, iv, sv = layers(val_df)
    rt, it_, st = layers(test_df)
    res = {}

    res["rule_only"] = (qf1(yv, rv), qf1(yt, rt))
    res["if_only"] = (qf1(yv, iv), qf1(yt, it_))
    res["if_and_rule"] = (qf1(yv, iv & rv), qf1(yt, it_ & rt))
    res["if_or_rule"] = (qf1(yv, iv | rv), qf1(yt, it_ | rt))

    # soft_global: w/tauselected on val
    def nz(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

    rn, sn = nz(rule_nv(val_df, th, fcols)), nz(sv)
    bw, bt, bf_ = 0.0, 0.0, -1.0
    for w in W_GRID:
        f = w * sn + (1 - w) * rn
        for t in TAU_GRID:
            v = qf1(yv, (f >= t).astype(int))
            if v > bf_:
                bf_, bw, bt = v, w, t
    rnt, snt = nz(rule_nv(test_df, th, fcols)), nz(st)
    res["soft_global"] = (bf_, qf1(yt, ((bw * snt + (1 - bw) * rnt) >= bt).astype(int)))

    # soft_perchannel
    pv = np.zeros(len(val_df), dtype=int)
    pt = np.zeros(len(test_df), dtype=int)
    for ch in sorted(val_df["channel"].unique()):
        mv = (val_df["channel"] == ch).to_numpy()
        mt = (test_df["channel"] == ch).to_numpy()
        av, at_ = nz(sv[mv]), nz(st[mt])
        rv2, rt2 = nz(rule_nv(val_df, th, fcols)[mv]), nz(rule_nv(test_df, th, fcols)[mt])
        yv2 = val_df.loc[mv, "anomaly"].to_numpy()
        b2w, b2t, b2f = 0.0, 0.0, -1.0
        for w in W_GRID:
            f = w * av + (1 - w) * rv2
            for t in TAU_GRID:
                v = qf1(yv2, (f >= t).astype(int))
                if v > b2f:
                    b2f, b2w, b2t = v, w, t
        pv[mv] = ((b2w * av + (1 - b2w) * rv2) >= b2t).astype(int)
        pt[mt] = ((b2w * at_ + (1 - b2w) * rt2) >= b2t).astype(int)
    res["soft_perchannel"] = (qf1(yv, pv), qf1(yt, pt))

    # gate_perchannel
    picks, gv, gt = {}, np.zeros(len(val_df), dtype=int), np.zeros(len(test_df), dtype=int)
    for ch in sorted(val_df["channel"].unique()):
        mv = (val_df["channel"] == ch).to_numpy()
        mt = (test_df["channel"] == ch).to_numpy()
        y2 = val_df.loc[mv, "anomaly"].to_numpy()
        cand = {"rule": qf1(y2, rv[mv]), "if": qf1(y2, iv[mv]),
                "AND": qf1(y2, iv[mv] & rv[mv]), "OR": qf1(y2, iv[mv] | rv[mv])}
        op = max(cand, key=lambda k: cand[k])
        picks[ch] = op
        gv[mv] = {"rule": rv[mv], "if": iv[mv], "AND": iv[mv] & rv[mv],
                  "OR": iv[mv] | rv[mv]}[op]
        gt[mt] = {"rule": rt[mt], "if": it_[mt], "AND": it_[mt] & rt[mt],
                  "OR": it_[mt] | rt[mt]}[op]
    res["gate_perchannel"] = (qf1(yv, gv), qf1(yt, gt))

    # soft_calibrated
    lr = LogisticRegression(max_iter=1000).fit(np.c_[rn, sn], yv)
    cv_ = lr.predict_proba(np.c_[rn, sn])[:, 1]
    ct_ = lr.predict_proba(np.c_[rnt, snt])[:, 1]
    b3f, b3t = -1.0, 0.5
    for t in TAU_GRID_CAL:
        v = qf1(yv, (cv_ >= t).astype(int))
        if v > b3f:
            b3f, b3t = v, t
    res["soft_calibrated"] = (b3f, qf1(yt, (ct_ >= b3t).astype(int)))

    return res, picks, best_k


def main():
    fit_df, val_df, test_df, feature_cols = load_split()

    print("=" * 88)
    print("配置甲：IF 用全部 18 维（复现基线）")
    print("=" * 88)
    rA, pickA, kA = run_all(fit_df, val_df, test_df, feature_cols, feature_cols)

    print()
    print("=" * 88)
    print("配置乙：IF 用 3 维 val 贪心子集")
    print("=" * 88)
    rB, pickB, kB = run_all(fit_df, val_df, test_df, feature_cols, IF_SUBSET)

    # ---- 排序对比 ----
    print()
    print("=" * 88)
    print("8 种策略横向对比（test F1）")
    print("=" * 88)
    order = ["rule_only", "if_only", "if_and_rule", "if_or_rule",
             "soft_global", "soft_perchannel", "gate_perchannel", "soft_calibrated"]
    print(f"{'策略':<20}{'甲val':>9}{'甲test':>9}{'排名':>6}"
          f"{'乙val':>9}{'乙test':>9}{'排名':>6}{'Δtest':>9}")
    print("-" * 88)
    rankA = {k: i + 1 for i, k in enumerate(sorted(order, key=lambda k: -rA[k][1]))}
    rankB = {k: i + 1 for i, k in enumerate(sorted(order, key=lambda k: -rB[k][1]))}
    for k in order:
        print(f"{k:<20}{rA[k][0]:>9.4f}{rA[k][1]:>9.4f}{rankA[k]:>6}"
              f"{rB[k][0]:>9.4f}{rB[k][1]:>9.4f}{rankB[k]:>6}"
              f"{rB[k][1]-rA[k][1]:>+9.4f}")

    # ---- 核心结论是否翻转 ----
    print()
    print("=" * 88)
    print("论文核心结论是否翻转？")
    print("=" * 88)
    print("原文结论：「规则层是主梁，纯 IF 单独弱于纯规则」")
    print(f"  配置甲：rule_only={rA['rule_only'][1]:.4f}  if_only={rA['if_only'][1]:.4f}"
          f"  -> rule {'>' if rA['rule_only'][1] > rA['if_only'][1] else '<'} if "
          f"（差 {rA['rule_only'][1]-rA['if_only'][1]:+.4f}）")
    print(f"  配置乙：rule_only={rB['rule_only'][1]:.4f}  if_only={rB['if_only'][1]:.4f}"
          f"  -> rule {'>' if rB['rule_only'][1] > rB['if_only'][1] else '<'} if "
          f"（差 {rB['rule_only'][1]-rB['if_only'][1]:+.4f}）")
    flip = (rA['rule_only'][1] > rA['if_only'][1]) != (rB['rule_only'][1] > rB['if_only'][1])
    print(f"\n  => 结论{'**翻转**' if flip else '未翻转'}")

    print(f"\n  最优策略：甲 = {min(rankA, key=lambda k: rankA[k])}"
          f"  乙 = {min(rankB, key=lambda k: rankB[k])}")
    print(f"  甲最优 test F1 = {max(v[1] for v in rA.values()):.4f}")
    print(f"  乙最优 test F1 = {max(v[1] for v in rB.values()):.4f}")

    # ---- 门控配方变化 ----
    print()
    print("=" * 88)
    print("门控配方逐通道变化")
    print("=" * 88)
    print(f"{'通道':<12}{'甲(18维)':>12}{'乙(3维)':>12}   变化")
    print("-" * 60)
    for ch in sorted(pickA):
        tag = "" if pickA[ch] == pickB[ch] else f"  <- {pickA[ch]}->{pickB[ch]}"
        print(f"{ch:<12}{pickA[ch]:>12}{pickB[ch]:>12}{tag}")
    print(f"\n  甲配方占比: " + str({o: sum(1 for v in pickA.values() if v == o)
                                    for o in set(pickA.values())}))
    print(f"  乙配方占比: " + str({o: sum(1 for v in pickB.values() if v == o)
                                    for o in set(pickB.values())}))

    # ---- IF 判别力对照 ----
    print()
    print("=" * 88)
    print("IF 层判别力对照（segment 级test AUC）")
    print("=" * 88)
    Xf = np.nan_to_num(fit_df[IF_SUBSET].values)
    Xt = np.nan_to_num(test_df[IF_SUBSET].values)
    clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                          contamination=0.15, random_state=42, n_jobs=-1).fit(Xf)
    print(f"  IF(3维)  test AUC = {roc_auc_score(test_df['anomaly'].to_numpy(), -clf.decision_function(Xt)):.4f}")
    Xf18 = np.nan_to_num(fit_df[feature_cols].values)
    Xt18 = np.nan_to_num(test_df[feature_cols].values)
    clf18 = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                            contamination=0.15, random_state=42, n_jobs=-1).fit(Xf18)
    print(f"  IF(18维) test AUC = {roc_auc_score(test_df['anomaly'].to_numpy(), -clf18.decision_function(Xt18)):.4f}")


if __name__ == "__main__":
    main()
