"""探针：穷举验证「门控算子选择靠 tie-break 抛硬币」并扩展到 R3 配置。

背景（data-truth 报告，待核验）：
  fusion_v3.py:265-269 用严格 `>` 遍历 cand(dict 序 rule→if→AND→OR)，
  并列时保留初值。data-truth 报 9 通道中 7 个存在并列，
  穷举 512 种合规组合后 val F1 全等0.6029、test F1 在0.6224~0.6473 波动，
  即 0.6281 落在该波动区间内 -> 「门控优于 AND」不可辩护。

本脚本要回答三个问题：
  Q1 核验 data-truth 的 512 穷举结论（我独立重算，不采信转述）
  Q2 扩展到 R3/Tfull/I3 配置：它的门控选择是否也存在同样问题？
     若存在，则 0.7857 同样需要降级表述
  Q3 R3 配置的 k*=1 是否也是并列/无区分度的产物？
     若 val 上 k=1/2/3/4 的 F1 几乎相等，则 k 也是抛硬币

纪律：穷举只用 val 选，test 只在组合确定后评；不做任何test 选参。

用法：
    /d/Python313/python.exe scripts/probe_tiebreak_enumerate.py
"""

import os
import sys
import warnings
from itertools import product

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
OPS = ["rule", "if", "AND", "OR"]
F3 = ["kurtosis", "n_peaks", "smooth10_n_peaks"]


def qf1(yt, yp):
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(yt), np.asarray(yp), zero_division=0))


def build_op_bank(fit_df, val_df, test_df, rule_cols, if_cols, thr, seed=42,
                  k=None):
    """为每个通道算出 4 个算子在 val/test 上的预测向量与 F1。

    k 为 None 时在 val 上选（1~4）；否则固定用给定 k。
    注意：R3 配置下 val 选出 k=1，若沿用基线的 k=2 会显著低估——
    这正是本脚本第一版复现 0.7436 而非 0.7857 的原因。
    """
    yv, yt = val_df["anomaly"].to_numpy(), test_df["anomaly"].to_numpy()
    ch_v, ch_t = val_df["channel"].to_numpy(), test_df["channel"].to_numpy()

    nv_v, _ = apply_stat_rules(val_df, thr, rule_cols)
    nv_t, _ = apply_stat_rules(test_df, thr, rule_cols)

    if k is None:
        bk, bf = 1, -1.0
        for kk in (1, 2, 3, 4):
            f1 = qf1(yv, (nv_v >= kk).astype(int))
            if f1 > bf:
                bf, bk = f1, kk
    else:
        bk = k

    # 逐通道 contamination 在 val 上选
    best_c = {}
    for ch in np.unique(ch_v):
        cm = ch_v == ch
        cf = fit_df[fit_df["channel"] == ch]
        if len(cf) < MIN_FIT or int(cm.sum()) < MIN_EVAL:
            best_c[ch] = None
            continue
        Xf, Xv = np.nan_to_num(cf[if_cols].values), np.nan_to_num(
            val_df.loc[cm, if_cols].values)
        bc, bf = CONTAM_GRID[0], -1.0
        for c in CONTAM_GRID:
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=seed, n_jobs=-1).fit(Xf)
            f1 = qf1(yv[cm], (clf.predict(Xv) == -1).astype(int))
            if f1 > bf:
                bf, bc = f1, c
        best_c[ch] = float(bc)

    bank = {}
    for ch in np.unique(ch_t):
        mv, mt = (ch_v == ch), (ch_t == ch)
        y2v, y2t = yv[mv], yt[mt]
        rp_v = (nv_v[mv] >= bk).astype(int)
        rp_t = (nv_t[mt] >= bk).astype(int)
        c = best_c.get(ch)
        cf = fit_df[fit_df["channel"] == ch]
        if c is None or len(cf) < MIN_FIT:
            ip_v = np.zeros(int(mv.sum()), dtype=int)
            ip_t = np.zeros(int(mt.sum()), dtype=int)
        else:
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=seed, n_jobs=-1
                                  ).fit(np.nan_to_num(cf[if_cols].values))
            ip_v = (clf.predict(np.nan_to_num(
                val_df.loc[mv, if_cols].values)) == -1).astype(int)
            ip_t = (clf.predict(np.nan_to_num(
                test_df.loc[mt, if_cols].values)) == -1).astype(int)
        bank[ch] = {
            "val_f1": {op: qf1(y2v, p) for op, p in
                       [("rule", rp_v), ("if", ip_v),
                        ("AND", rp_v & ip_v), ("OR", rp_v | ip_v)]},
            "test_pred": {"rule": rp_t, "if": ip_t,
                          "AND": rp_t & ip_t, "OR": rp_t | ip_t},
        }
    return bank, yv, yt


def analyse(bank, yv, yt, test_df, tag):
    print()
    print("=" * 92)
    print(f"标签：{tag}")
    print("=" * 92)
    print(f"{'通道':<12}{'val正例':>9}{'rule':>8}{'if':>8}{'AND':>8}{'OR':>8}"
          f"{'并列数':>8}{'maxF1':>8}  最优算子集")
    print("-" * 92)
    ch_t = test_df["channel"].to_numpy()
    ties, free = [], []
    for ch in sorted(bank):
        vf = bank[ch]["val_f1"]
        mx = max(vf.values())
        tied = [op for op in OPS if abs(vf[op] - mx) < 1e-12]
        n_an_v = None
        if len(tied) == len(OPS):
            ties.append(ch)
        elif len(tied) > 1:
            free.append(ch)
        print(f"{ch:<12}{'-':>9}{vf['rule']:>8.4f}{vf['if']:>8.4f}"
              f"{vf['AND']:>8.4f}{vf['OR']:>8.4f}{len(tied):>8}{mx:>8.4f}  {tied}")

    print(f"\n  全并列（4 算子 F1 相同，选谁都一样）：{len(ties)}/9 -> {ties}")
    print(f"  部分并列（>=2 个算子并列）：          {len(free)}/9 -> {free}")
    real = 9 - len(ties) - len(free)
    print(f"  真·唯一最优：                        {real}/9")

    # 穷举所有合规组合
    # 全并列通道（4 算子同分）选谁都等价，固定取 dict 序第一个 'rule'；
    # 部分并列通道才有 2 个自由度。
    n_var = len(free)
    if n_var == 0:
        print("\n  无并列可变通道 -> 组合唯一")
        return ties, free, None
    total = 2 ** n_var
    print(f"\n  穷举 {total} 种合规组合（{n_var} 个部分并列通道各 2 选）...")
    print(f"  （{len(ties)} 个全并列通道固定取 'rule'，选谁都等价，不计入自由度）")
    ch_t = test_df["channel"].to_numpy()
    test_f1s = []
    for combo in product([0, 1], repeat=n_var):
        choice = {ch: "rule" for ch in ties}
        # 真·唯一最优通道也固定
        for ch in bank:
            if ch not in free and ch not in ties:
                choice[ch] = max(bank[ch]["val_f1"],
                                 key=lambda o: bank[ch]["val_f1"][o])
        for c, ch in zip(combo, free):
            vf = bank[ch]["val_f1"]
            mx = max(vf.values())
            choice[ch] = [op for op in OPS if abs(vf[op] - mx) < 1e-12][c]
        tp = [bank[ch]["test_pred"][choice[ch]] for ch in sorted(bank)]
        y_all = np.concatenate([yt[ch_t == ch] for ch in sorted(bank)])
        test_f1s.append(qf1(y_all, np.concatenate(tp)))
    test_f1s = np.array(test_f1s)
    print(f"  test F1：min={test_f1s.min():.4f}  max={test_f1s.max():.4f}  "
          f"极差={test_f1s.max()-test_f1s.min():.4f}")
    q = np.percentile(test_f1s, [0, 25, 50, 75, 100])
    print(f"    min={q[0]:.4f}  Q1={q[1]:.4f}  中位={q[2]:.4f}  "
          f"Q3={q[3]:.4f}  max={q[4]:.4f}")
    return ties, free, test_f1s


def main():
    fit_df, val_df, test_df, feature_cols = load_split()

    # ============ 配置甲：基线 18 维 ============
    thr18 = compute_stat_thresholds(fit_df, feature_cols)
    bankA, yv, yt = build_op_bank(fit_df, val_df, test_df, feature_cols,
                                  feature_cols, thr18, k=2)
    resA = analyse(bankA, yv, yt, test_df, "配置甲：基线 R18/Tfull/I18（k=2）")

    # 记录的 gate 值对照
    ch_t = test_df["channel"].to_numpy()
    pick = {ch: max(bankA[ch]["val_f1"], key=lambda o: bankA[ch]["val_f1"][o])
            for ch in bankA}
    yp = np.concatenate([bankA[ch]["test_pred"][pick[ch]] for ch in sorted(bankA)])
    y_all = np.concatenate([yt[ch_t == ch] for ch in sorted(bankA)])
    print(f"\n  按 fusion_v3 的 max() 实际选出：{pick}")
    print(f"  复现 gate_perchannel test F1 = {qf1(y_all, yp):.4f}  (应= 0.6281)")
    if resA and len(resA) == 3:
        _, _, tf = resA
        print(f"  0.6281 是否落在 tie-break 波动区间 "
              f"[{tf.min():.4f}, {tf.max():.4f}] 内 = "
              f"{'是' if tf.min() <= 0.6281 <= tf.max() else '否'}")
        print(f"  AND 基线 0.6038 是否也落在该区间内 = "
              f"{'是' if tf.min() <= 0.6038 <= tf.max() else '否'}")
        print("  => 若两者都落在区间内，则「门控 vs AND」的 +0.0243 "
              "**在 test 上不可区分**。")

    # ============ 配置乙：R3 ============
    print()
    print("#" * 92)
    print("# 配置乙：R3/Tfull/I3（k=1 口径）")
    print("#" * 92)
    thr3 = compute_stat_thresholds(fit_df, F3)
    nv_v3, _ = apply_stat_rules(val_df, thr3, F3)
    yv3 = val_df["anomaly"].to_numpy()
    print()
    print("=" * 92)
    print("Q3 R3 配置的 k 选择是否有区分度（k*=1 是真最优还是并列）")
    print("=" * 92)
    print(f"{'k':>4}{'val F1':>12}{'与最优差':>12}")
    print("-" * 28)
    kf1 = {k: qf1(yv3, (nv_v3 >= k).astype(int)) for k in (1, 2, 3, 4, 5)}
    for k, f1 in kf1.items():
        print(f"{k:>4}{f1:>12.4f}{f1 - max(kf1.values()):>+12.4f}")
    tied_k = [k for k, f1 in kf1.items() if abs(f1 - max(kf1.values())) < 1e-12]
    print(f"\n  并列最优的 k：{tied_k}  -> "
          f"{'k 有区分度' if len(tied_k) == 1 else '**k 也是并列，k*=1 属任意取值**'}")

    # R3 的门控 bank（k=1）
    print()
    bankB, _, _ = build_op_bank(fit_df, val_df, test_df, F3, F3, thr3, k=1)
    resB = analyse(bankB, yv, yt, test_df, "配置乙：R3/Tfull/I3（k=1）")

    # R3 实际选出与复现
    pickB = {ch: max(bankB[ch]["val_f1"], key=lambda o: bankB[ch]["val_f1"][o])
             for ch in bankB}
    ypB = np.concatenate([bankB[ch]["test_pred"][pickB[ch]] for ch in sorted(bankB)])
    print(f"\n  选出：{pickB}")
    print(f"  复现 R3/I3 test F1 = {qf1(y_all, ypB):.4f}  (应≈ 0.7857)")
    if resB and len(resB) == 3:
        _, _, tfB = resB
        print(f"  tie-break 波动区间 = [{tfB.min():.4f}, {tfB.max():.4f}]，"
              f"极差 {tfB.max()-tfB.min():.4f}")
        print(f"  0.7857 落在区间内 = {'是' if tfB.min() <= 0.7857 <= tfB.max() else '否'}")
        print("  => R3 配置的门控选择"
              + ("同样存在并列问题，0.7857 需同样降级表述"
                 if resB[0] or resB[1] else "无并列，0.7857 来自唯一确定的算子"))


if __name__ == "__main__":
    main()
