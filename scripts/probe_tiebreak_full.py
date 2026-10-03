"""探针 v2：穷举门控 tie-break 的**完整**波动区间（含全并列通道的所有算子）。

修正 v1 的 bug：
  v1 把「全并列通道」（4 算子 val F1 相同）固定取 'rule'，理由是选谁等价。
  这在 val 上成立（0884 val 零正类 -> F1 恒 0），但在 test 上**不成立**：
  0884/0890 的 test 有正类，不同算子的预测向量不同 -> test F1 不同。
  结果 v1 只枚举 32 组合、上界 0.6393，**低估了 tie-break 波动**。
  algo-diagnose 独立枚举 512 组合给出上界 0.6473，正好暴露此bug。

本脚本 v2 对每个通道枚举「val F1 等于该通道最大值的全部算子」，
  总组合数 = 各通道并列数之积（上限 4^9），不做任何简化。

要回答的问题（全部只用 val 选，test 仅在组合确定后评估）：
  Q1 复现 algo-diagnose 的 512 组合区间 [0.6224, 0.6473]（基线 R18/Tfull/I18）
  Q2 基线 0.6281 在区间中的位置；AND 0.6038 是否落在区间内
  Q3 R3/Tfull/I3 配置的完整区间；0.7857 在其中的位置
  Q4 R3 的 k*=1 是否有区分度

用法：
    /d/Python313/python.exe scripts/probe_tiebreak_full.py
"""

import os
import sys
import warnings
from itertools import product

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

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
COMBO_CAP = 400000


def qf1(yt, yp):
    return float(f1_score(np.asarray(yt), np.asarray(yp), zero_division=0))


def build_op_bank(fit_df, val_df, test_df, rule_cols, if_cols, thr, seed=42, k=None):
    """为每通道算出 4 算子在 val/test 的预测向量与 val F1。"""
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
    sel_k = bk

    best_c = {}
    for ch in np.unique(ch_v):
        cm = ch_v == ch
        cf = fit_df[fit_df["channel"] == ch]
        if len(cf) < MIN_FIT or int(cm.sum()) < MIN_EVAL:
            best_c[ch] = None
            continue
        Xf = np.nan_to_num(cf[if_cols].values)
        Xv = np.nan_to_num(val_df.loc[cm, if_cols].values)
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
        cf = fit_df[fit_df["channel"] == ch]
        rp_v, rp_t = (nv_v[mv] >= sel_k).astype(int), (nv_t[mt] >= sel_k).astype(int)
        c = best_c.get(ch)
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
            "n_val": int(mv.sum()),
            "n_val_pos": int(yv[mv].sum()),
            "n_test": int(mt.sum()),
            "k": sel_k,
            "c": best_c.get(ch),
            "val_f1": {op: qf1(yv[mv], p) for op, p in
                       [("rule", rp_v), ("if", ip_v),
                        ("AND", rp_v & ip_v), ("OR", rp_v | ip_v)]},
            "test_pred": {"rule": rp_t, "if": ip_t,
                          "AND": rp_t & ip_t, "OR": rp_t | ip_t},
        }
    return bank, yv, yt, sel_k


def enumerate_full(bank, yt, test_df, tag, ref=None):
    """对每通道枚举「val F1 等于最大值的全部算子」，穷举所有合规组合的 test F1。"""
    print()
    print("=" * 100)
    print(f"标签：{tag}")
    print("=" * 100)
    print(f"{'通道':<11}{'val段':>6}{'val正':>6}{'test段':>7}{'test正':>7}"
          f"{'k':>3}{'rule':>8}{'if':>8}{'AND':>8}{'OR':>8}{'并列数':>7}  并列算子")
    print("-" * 100)

    ch_t = test_df["channel"].to_numpy()
    chans = sorted(bank)
    opt_sets = {}
    for ch in chans:
        vf = bank[ch]["val_f1"]
        mx = max(vf.values())
        tied = [op for op in OPS if abs(vf[op] - mx) < 1e-12]
        opt_sets[ch] = tied
        print(f"{ch:<11}{bank[ch]['n_val']:>6}{bank[ch]['n_val_pos']:>6}"
              f"{bank[ch]['n_test']:>7}"
              f"{int(yt[ch_t == ch].sum()):>7}{bank[ch]['k']:>3}"
              f"{vf['rule']:>8.4f}{vf['if']:>8.4f}{vf['AND']:>8.4f}{vf['OR']:>8.4f}"
              f"{len(tied):>7}  {tied}")

    n_full = sum(1 for ch in chans if len(opt_sets[ch]) == len(OPS))
    n_part = sum(1 for ch in chans if 1 < len(opt_sets[ch]) < len(OPS))
    n_uniq = sum(1 for ch in chans if len(opt_sets[ch]) == 1)
    print(f"\n  全并列 {n_full}/9   部分并列 {n_part}/9   唯一最优 {n_uniq}/9")

    total = 1
    for ch in chans:
        total *= len(opt_sets[ch])
    if total > COMBO_CAP:
        print(f"  组合数 {total} 超过上限 {COMBO_CAP} -> 随机采样 {COMBO_CAP} 组")
        rng = np.random.default_rng(0)
        combos = []
        for _ in range(COMBO_CAP):
            combos.append([rng.choice(opt_sets[ch]) for ch in chans])
        exact = False
    else:
        combos = list(product(*[opt_sets[ch] for ch in chans]))
        exact = True
    print(f"  穷举 {len(combos)} 种合规组合"
          f"{'（精确）' if exact else '（采样）'}"
          f"  = 各通道并列数之积 {' x '.join(str(len(opt_sets[ch])) for ch in chans)}")

    y_all = np.concatenate([yt[ch_t == ch] for ch in chans])
    f1s = np.empty(len(combos))
    for i, combo in enumerate(combos):
        f1s[i] = qf1(y_all, np.concatenate(
            [bank[ch]["test_pred"][combo[j]] for j, ch in enumerate(chans)]))

    lo, hi = f1s.min(), f1s.max()
    q = np.percentile(f1s, [0, 10, 25, 50, 75, 90, 100])
    print(f"\n  test F1 全组合分布：min={lo:.4f}  Q10={q[1]:.4f}  Q25={q[2]:.4f}"
          f"  中位={q[3]:.4f}  Q75={q[4]:.4f}  Q90={q[5]:.4f}  max={hi:.4f}")
    print(f"  极差 = {hi - lo:.4f}   唯一取值个数 = {len(np.unique(np.round(f1s, 6)))}")

    # 复现 fusion_v3 的 max() 实际选出（严格 > + dict 序 rule→if→AND→OR）
    pick = {ch: max(OPS, key=lambda o: (bank[ch]["val_f1"][o], -OPS.index(o)))
            for ch in chans}
    print(f"  max() 实际选出：{pick}")
    rep = qf1(y_all, np.concatenate(
        [bank[ch]["test_pred"][pick[ch]] for ch in chans]))
    print(f"  复现 test F1 = {rep:.4f}")
    pct = float((f1s < rep - 1e-12).mean() * 100)
    print(f"  该值在组合分布中的百分位 = {pct:.1f}%")

    if ref is not None:
        for name, rv in ref.items():
            inside = lo - 1e-12 <= rv <= hi + 1e-12
            print(f"  {name}={rv:.4f} 落在区间内 = {'是' if inside else '否'}"
                  f"  （距下界 {rv - lo:+.4f} / 距上界 {rv - hi:+.4f}）")
    return {"lo": lo, "hi": hi, "rep": rep, "pct": pct, "f1s": f1s,
            "opt_sets": opt_sets, "pick": pick, "total": len(combos),
            "n_full": n_full, "n_part": n_part, "n_uniq": n_uniq}


def main():
    fit_df, val_df, test_df, feature_cols = load_split()
    print(f"特征数={len(feature_cols)}  fit={len(fit_df)}  val={len(val_df)}  "
          f"test={len(test_df)}")

    # ---------- 配置甲：基线 R18/Tfull/I18，k=2 ----------
    thr18 = compute_stat_thresholds(fit_df, feature_cols)
    bankA, yv, yt, kA = build_op_bank(fit_df, val_df, test_df, feature_cols,
                                      feature_cols, thr18, k=2)
    print(f"\n[配置甲] 固定 k=2（基线 fusion_v3 实得值）")
    resA = enumerate_full(bankA, yt, test_df,
                          "配置甲：基线 R18/Tfull/I18（k=2）",
                          ref={"基线 gate": 0.6281, "AND 基线": 0.6038,
                               "rule_only": 0.5882, "algo-diagnose 上界": 0.6473,
                               "algo-diagnose 下界": 0.6224})

    # ---------- 配置乙：R3/Tfull/I3，k=1 ----------
    print()
    print("#" * 100)
    print("# 配置乙：R3/Tfull/I3")
    print("#" * 100)
    thr3 = compute_stat_thresholds(fit_df, F3)

    # Q4：k 的区分度
    nv_v3, _ = apply_stat_rules(val_df, thr3, F3)
    yv3 = val_df["anomaly"].to_numpy()
    print("\nQ4  R3 配置的 k 选择是否有区分度")
    print("-" * 52)
    kf1 = {k: qf1(yv3, (nv_v3 >= k).astype(int)) for k in (1, 2, 3, 4, 5)}
    mx = max(kf1.values())
    for k, f1 in kf1.items():
        print(f"  k={k}  val F1={f1:.4f}  与最优差 {f1 - mx:+.4f}")
    tied_k = [k for k, f1 in kf1.items() if abs(f1 - mx) < 1e-12]
    print(f"  并列最优 k = {tied_k} -> "
          f"{'k 有唯一区分度' if len(tied_k) == 1 else '**k 并列，k*=1 属任意取值**'}")

    bankB, _, _, _ = build_op_bank(fit_df, val_df, test_df, F3, F3, thr3, k=1)
    print("\n[配置乙] 固定 k=1（val 上选出）")
    resB = enumerate_full(bankB, yt, test_df,
                          "配置乙：R3/Tfull/I3（k=1）",
                          ref={"R3/I3 交叉矩阵值": 0.7857,
                               "本脚本上一版(有bug)": 0.7436})

    # ---------- 结论对照 ----------
    print()
    print("=" * 100)
    print("结论对照：两个配置的 tie-break 脆弱性")
    print("=" * 100)
    print(f"{'配置':<26}{'组合数':>8}{'唯一值':>8}{'区间':>20}"
          f"{'极差':>9}{'报告值':>9}{'百分位':>9}")
    print("-" * 100)
    for tag, r, rv in (("基线 R18/I18", resA, 0.6281),
                       ("R3/I3", resB, 0.7857)):
        print(f"{tag:<26}{r['total']:>8}{len(np.unique(np.round(r['f1s'], 6))):>8}"
              f"   [{r['lo']:.4f}, {r['hi']:.4f}]{r['hi'] - r['lo']:>9.4f}"
              f"{rv:>9.4f}{r['pct']:>8.1f}%")
    print()
    print("  判读标准：")
    print("   (a) 若报告值落在区间内 -> 该数字由 tie-break 抛硬币决定，"
          "不可表述为'选到了最优算子'")
    print("   (b) 若对照基线（AND 0.6038 / rule_only 0.5882）落在区间**外**下界之下"
          " -> 方向性结论仍成立")
    print("   (c) 极差越大 -> 单一报告值越不可信，必须给区间而非点估计")


if __name__ == "__main__":
    main()
