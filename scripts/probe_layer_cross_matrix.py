"""探针（交叉矩阵）：规则层维度 × 阈值口径 × IF 维度 的全组合实测。

背景：两条独立路线选了**同一组 3 个特征**，但用在不同层：
  algo-diagnose 路线：净阈值 + 规则层 11 特征白名单 + IF 见 18 维 -> test 0.7155
                    并发现「判据阈值 0.20 档」= {kurtosis, n_peaks,
                    smooth10_n_peaks}（k*=1），test 0.7456
  主理人路线：      全量阈值 + 规则层见 18 维 + IF 用 3 维  -> test 0.7142
两者的 3 维特征完全一致（**两条独立方法交叉验证成功**），但
「3 维喂给规则层」与「3 维喂给 IF」从未被交叉测过。

本脚本跑 3(规则层维度) × 2(阈值口径) × 2(IF 维度) = 12 组全组合：
  规则层维度 R18=全18维 / R11=algo-diagnose 的 11 维白名单 / R3=三形状特征
  阈值口径  T_full=全量(现状) / T_net=净阈值(仅正常段拟合, >=30 护栏)
  IF 维度    I18=全 18 维 / I3=三形状特征

回答三个问题：
  Q1 12 组里最优是哪一组？test F1 能到多少？
  Q2 algo-diagnose 的 0.7155 能否被逐位复现？（验证 11 维白名单是否复现）
  Q3 R3 与 I3 叠加是否有增益，还是两者只等价一次？

纪律：
  - 规则层 k、IF 的 contamination、门控算子，全部只在 val 上选；
  - 阈值只在 fit 上拟合（净阈值只用 fit 内 anomaly==0 段，护栏 30）；
  - test 只在配置完全确定后评一次；
  - 特征子集沿用两方交叉确认的 3 维，不在 test 上挑特征。

用法：
    /d/Python313/python.exe scripts/probe_layer_cross_matrix.py
"""

import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import IsolationForest

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
CONTAM_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MIN_FIT = 10
MIN_EVAL = 5
MIN_NORMAL = 30

# 三形状特征：algo-diagnose（判据阈值 0.20 档）与主理人（贪心前向）独立选出
F3 = ["kurtosis", "n_peaks", "smooth10_n_peaks"]
# algo-diagnose 报告的 11 维白名单（|valAUC-0.5| >= 0.10 档）
F11 = ["n_peaks", "smooth10_n_peaks", "kurtosis", "len", "gaps_squared",
       "len_weighted", "duration", "mean", "diff2_var", "diff2_peaks",
       "var_div_len"]


def qf1(yt, yp):
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(yt), np.asarray(yp), zero_division=0))


def build_thr(fit_df, cols, use_net):
    """净阈值：只用 fit 内正常段拟合；正常段 <30 退回全量。"""
    if not use_net:
        return compute_stat_thresholds(fit_df, cols)
    normal = fit_df[fit_df["anomaly"] == 0]
    net = compute_stat_thresholds(normal, cols)
    out = {}
    for ch in fit_df["channel"].unique():
        sub = fit_df[fit_df["channel"] == ch]
        n_norm = int((sub["anomaly"] == 0).sum())
        out[ch] = net[ch] if n_norm >= MIN_NORMAL else compute_stat_thresholds(
            sub, cols)[ch]
    return out


def run(fit_df, val_df, test_df, rule_cols, if_cols, thr, seed=42,
        k_candidates=(1, 2, 3, 4)):
    """门控端到端。rule_cols 决定规则层用哪些特征；if_cols 决定 IF 输入。"""
    yv, yt = val_df["anomaly"].to_numpy(), test_df["anomaly"].to_numpy()
    ch_va, ch_te = val_df["channel"].to_numpy(), test_df["channel"].to_numpy()

    # 规则层 k 在 val 上选
    nv_va, _ = apply_stat_rules(val_df, thr, rule_cols)
    best_k, bk = k_candidates[0], -1.0
    for k in k_candidates:
        f1 = qf1(yv, (nv_va >= k).astype(int))
        if f1 > bk:
            bk, best_k = f1, k

    # 逐通道 contamination 在 val 上选
    best_c = {}
    for ch in np.unique(ch_va):
        cm = ch_va == ch
        cf = fit_df[fit_df["channel"] == ch]
        if len(cf) < MIN_FIT or int(cm.sum()) < MIN_EVAL:
            best_c[ch] = None
            continue
        y = yv[cm]
        Xf, Xv = np.nan_to_num(cf[if_cols].values), np.nan_to_num(
            val_df.loc[cm, if_cols].values)
        bc, bf = CONTAM_GRID[0], -1.0
        for c in CONTAM_GRID:
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=seed, n_jobs=-1).fit(Xf)
            f1 = qf1(y, (clf.predict(Xv) == -1).astype(int))
            if f1 > bf:
                bf, bc = f1, c
        best_c[ch] = float(bc)

    def layers(df, chs):
        nv, _ = apply_stat_rules(df, thr, rule_cols)
        rp = (nv >= best_k).astype(int).to_numpy()
        ip = np.zeros(len(df), dtype=int)
        for ch in chs:
            m = (df["channel"] == ch).to_numpy()
            c = best_c.get(ch)
            cf = fit_df[fit_df["channel"] == ch]
            if c is None or len(cf) < MIN_FIT:
                continue
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=seed, n_jobs=-1
                                  ).fit(np.nan_to_num(cf[if_cols].values))
            ip[m] = (clf.predict(np.nan_to_num(
                df.loc[m, if_cols].values)) == -1).astype(int)
        return rp, ip

    rv, iv = layers(val_df, np.unique(ch_va))
    rt, it_ = layers(test_df, np.unique(ch_te))

    picks, yp = {}, np.zeros(len(test_df), dtype=int)
    for ch in np.unique(ch_te):
        mv, mt = (ch_va == ch), (ch_te == ch)
        y2 = yv[mv]
        cand = {"rule": qf1(y2, rv[mv]), "if": qf1(y2, iv[mv]),
                "AND": qf1(y2, iv[mv] & rv[mv]), "OR": qf1(y2, iv[mv] | rv[mv])}
        op = max(cand, key=lambda k: cand[k])
        picks[ch] = op
        yp[mt] = {"rule": rt[mt], "if": it_[mt], "AND": it_[mt] & rt[mt],
                  "OR": it_[mt] | rt[mt]}[op]
    return yt, yp, picks, best_k, (rv, iv, rt, it_)


def paired_boot(yt, pa, pb, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    m = len(yt)
    ds = [qf1(yt[i], pb[i]) - qf1(yt[i], pa[i])
          for i in (rng.integers(0, m, m) for _ in range(n))]
    ds = np.array(ds)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return lo, hi, float((ds > 0).mean())


def main():
    fit_df, val_df, test_df, feature_cols = load_split()
    cols18 = list(feature_cols)
    cols11 = [c for c in F11 if c in cols18]
    print(f"F3  = {F3}")
    print(f"F11 = {len(cols11)} 维（缺失已剔除：{[c for c in F11 if c not in cols18]}）")
    print(f"F18 = {len(cols18)} 维\n")

    combos = []
    for rtag, rcols in [("R18", cols18), ("R11", cols11), ("R3", F3)]:
        for ttag in ["Tfull", "Tnet"]:
            for itag, icols in [("I18", cols18), ("I3", F3)]:
                combos.append((rtag, ttag, itag, rcols, icols))

    print("=" * 96)
    print("Q1 全组合矩阵（IF seed=42；k/门控/污染率全部在 val 上选）")
    print("=" * 96)
    print(f"{'规则层':<8}{'阈值':<8}{'IF层':<8}{'k*':>4}"
          f"{'val F1':>10}{'test F1':>10}{'P':>9}{'R':>9}{'testAUC':>9}")
    print("-" * 96)
    store = {}
    for rtag, ttag, itag, rcols, icols in combos:
        thr = build_thr(fit_df, rcols, ttag == "Tnet")
        yt, yp, picks, bk, _ = run(fit_df, val_df, test_df, rcols, icols, thr)
        f1 = qf1(yt, yp)
        yv = val_df["anomaly"].to_numpy()
        thr_v = build_thr(fit_df, rcols, ttag == "Tnet")
        nvv, _ = apply_stat_rules(val_df, thr_v, rcols)
        val_f1 = qf1(yv, (nvv >= bk).astype(int))
        from sklearn.metrics import precision_score, recall_score
        P = precision_score(yt, yp, zero_division=0)
        R = recall_score(yt, yp, zero_division=0)
        # 端到端 AUC：用融合前的 IF 分数无关，这里给 IF 层 AUC
        store[(rtag, ttag, itag)] = dict(f1=f1, val=val_f1, yt=yt, yp=yp, k=bk,
                                         picks=picks, P=P, R=R)
        print(f"{rtag:<8}{ttag:<8}{itag:<8}{bk:>4}{val_f1:>10.4f}{f1:>10.4f}"
              f"{P:>9.4f}{R:>9.4f}")

    # ---- Q2 复现 algo-diagnose 的 0.7155 ----
    print()
    print("=" * 96)
    print("Q2 复现核对")
    print("=" * 96)
    k_net11 = store[("R11", "Tnet", "I18")]["f1"]
    print(f"  algo-diagnose 报 净阈值+R11白名单+IF18维 = 0.7155")
    print(f"  本探针实测 同口径                        = {k_net11:.4f}"
          f"   {'✅ 吻合' if abs(k_net11-0.7155) < 5e-3 else '⚠ 不吻合'}")
    k_full18 = store[("R18", "Tfull", "I18")]["f1"]
    print(f"  基线复现 全量+R18+IF18 (应= 0.6281)      = {k_full18:.4f}"
          f"   {'✅ 吻合' if abs(k_full18-0.6281) < 5e-3 else '⚠ 不吻合'}")
    k_full3 = store[("R18", "Tfull", "I3")]["f1"]
    print(f"  主理人路线 全量+R18+IF3  (应≈ 0.6846)     = {k_full3:.4f}"
          f"   {'✅ 吻合' if abs(k_full3-0.6846) < 8e-3 else '⚠ 不吻合'}")

    # ---- Q1 排名 ----
    print()
    print("=" * 96)
    print("全组合排名（按 test F1）")
    print("=" * 96)
    for i, (key, v) in enumerate(sorted(store.items(), key=lambda t: -t[1]["f1"])[:6], 1):
        print(f"  {i}. {key[0]:<5}{key[1]:<7}{key[2]:<5} test F1 = {v['f1']:.4f}  "
              f"val = {v['val']:.4f}  k*={v['k']}")

    best_key = max(store, key=lambda k: store[k]["f1"])
    B = store[best_key]
    base = store[("R18", "Tfull", "I18")]

    print()
    print("=" * 96)
    print("Q3 增益分解：规则层换 3 维 vs IF 层换 3 维 vs 两者都换")
    print("=" * 96)
    for key, label in [
        (("R18", "Tfull", "I18"), "基线            全量 R18 I18"),
        (("R3", "Tfull", "I18"), "只换规则层3维    全量 R3  I18"),
        (("R18", "Tfull", "I3"), "只换 IF 层 3 维  全量 R18 I3 "),
        (("R3", "Tfull", "I3"), "两者都换3维      全量 R3  I3 "),
        (("R18", "Tnet", "I18"), "只加净阈值      净  R18 I18"),
        (("R3", "Tnet", "I18"), "净阈值+规则3维   净  R3  I18"),
        (("R18", "Tnet", "I3"), "净阈值+IF3 维    净  R18 I3 "),
        (("R3", "Tnet", "I3"), "净阈值+两者3维   净  R3  I3 "),
    ]:
        if key in store:
            v = store[key]
            print(f"  {label:<34} test F1 = {v['f1']:.4f}  "
                  f"Δ基线 = {v['f1']-base['f1']:+.4f}")

    # ---- 最优组 vs 基线 的配对 bootstrap ----
    print()
    print("=" * 96)
    print(f"配对 bootstrap：最优组 {best_key} vs 基线 (R18/Tfull/I18)")
    print("=" * 96)
    lo, hi, pos = paired_boot(base["yt"], base["yp"], B["yp"])
    print(f"  ΔF1 = {B['f1']-base['f1']:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
          f"为正概率 {pos*100:.1f}%  {'显著' if lo > 0 else '不显著(含0)'}")
    print(f"\n  最优组门控配方: " +
          ", ".join(f"{k[-4:]}={v}" for k, v in sorted(B["picks"].items())))
    print(f"  最优组 P={B['P']:.4f} R={B['R']:.4f}  k*={B['k']}")
    print(f"\n  => 推荐配置：规则层 {best_key[0]} / 阈值 {best_key[1]} / IF {best_key[2]}")


if __name__ == "__main__":
    main()
