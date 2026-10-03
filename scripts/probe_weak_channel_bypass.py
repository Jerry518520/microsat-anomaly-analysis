"""探针：核验「生产弱通道完全绕过规则层」这个结构缺陷值多少 F1。

代码事实（src/integration/anomaly_rag_pipeline.py:283-298）：
    if ch in STRONG_CHANNELS:            # 只有 0872/0873/0874
        pred_g = max(if_pred, r_pred)    # 强通道才用 max() = OR
    else:
        pred_g = seg_baseline_map[...]   # 弱通道 = 一个全局 IF(contamination=0.25)
                                        # **完全不看规则层**

而 STRONG_CHANNELS 只有 3 个通道（0872/0873/0874），另外 6 个通道
（0884/0886/0888/0890/0892/0894）都不走规则层。

algo-diagnose 报「对 0888 硬套 IF OR 规则=0.5676，远低于规则层 0.8000」，
说明给弱通道补上规则层是明显收益。本脚本量化三件事：

  1. 弱通道6 个通道在 test 上的段数与真值异常数（有多少量可以被改善）
  2. 弱通道现用「全局 IF(contam=0.25)」vs 「规则层」逐通道 F1
  3. 若把 STRONG_CHANNELS 扩到全部通道（即弱通道也走 max(IF, rule)），
     端到端 test F1 能到多少 —— 这是纯代码改动，不动算法

纪律：只读复算，不改 anomaly_rag_pipeline.py；contamination 沿用生产值 0.25。

用法：
    /d/Python313/python.exe scripts/probe_weak_channel_bypass.py
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

STRONG = {"CADC0872", "CADC0873", "CADC0874"}
PROD_C = 0.25
PROD_PSI = 128


def qf1(yt, yp):
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(yt), np.asarray(yp), zero_division=0))


def main():
    fit_df, val_df, test_df, feature_cols = load_split()
    th = compute_stat_thresholds(fit_df, feature_cols)
    nv_te, _ = apply_stat_rules(test_df, th, feature_cols)
    nv_va, _ = apply_stat_rules(val_df, th, feature_cols)
    yv, yt = val_df["anomaly"].to_numpy(), test_df["anomaly"].to_numpy()

    bk, bf = None, -1.0
    for k in (1, 2, 3):
        f1 = qf1(yv, (nv_va >= k).astype(int))
        if f1 > bf:
            bf, bk = f1, k
    rule_t = (nv_te >= bk).astype(int)
    print(f"[规则层] k={bk}（val 选出）\n")

    # 全局 IF（生产的段级 baseline 口径）
    Xf = np.nan_to_num(fit_df[feature_cols].values)
    Xt = np.nan_to_num(test_df[feature_cols].values)
    g = IsolationForest(n_estimators=100, max_samples=PROD_PSI,
                        contamination=PROD_C, random_state=42, n_jobs=-1).fit(Xf)
    global_if = (g.predict(Xt) == -1).astype(int)

    # 逐通道 IF（生产的 channel_preds 口径）
    ch_te = test_df["channel"].to_numpy()
    chan_if = np.zeros(len(test_df), dtype=int)
    for ch in np.unique(ch_te):
        m = ch_te == ch
        cf = fit_df[fit_df["channel"] == ch]
        if len(cf) < 10:
            continue
        mdl = IsolationForest(n_estimators=100, max_samples=PROD_PSI,
                              contamination=PROD_C, random_state=42,
                              n_jobs=-1).fit(np.nan_to_num(cf[feature_cols].values))
        chan_if[m] = (mdl.predict(Xt[m]) == -1).astype(int)

    # ---- 1. 强/弱通道量级 ----
    print("=" * 86)
    print("1. 强/弱通道的量级（哪些段能靠补规则层被改善）")
    print("=" * 86)
    print(f"{'通道':<12}{'归属':<8}{'test段':>8}{'真值异常':>10}{'异常占比':>10}")
    print("-" * 56)
    for ch in sorted(np.unique(ch_te)):
        m = ch_te == ch
        tag = "强(OR)" if ch in STRONG else "弱(纯IF)"
        print(f"{ch:<12}{tag:<8}{int(m.sum()):>8}{int(yt[m].sum()):>10}"
              f"{yt[m].mean():>10.4f}")
    mw = ~np.isin(ch_te, list(STRONG))
    print(f"\n  弱通道合计：{int(mw.sum())} 段，其中真值异常 {int(yt[mw].sum())} 段"
          f"（占全部异常的 {yt[mw].sum()/yt.sum()*100:.1f}%）")
    print("  => 弱通道承载了大部分异常，**却完全不使用规则层**。")

    # ---- 2. 逐通道：弱通道现用全局 IF vs 规则层 ----
    print()
    print("=" * 86)
    print("2. 逐通道 F1：弱通道现用「全局IF(0.25)」 vs 「规则层」")
    print("=" * 86)
    print(f"{'通道':<12}{'归属':<9}{'全局IF':>9}{'通道IF':>9}{'规则层':>9}"
          f"{'max(IF,rule)':>14}{'最优者':>10}")
    print("-" * 74)
    for ch in sorted(np.unique(ch_te)):
        m = ch_te == ch
        y = yt[m]
        f_g, f_c = qf1(y, global_if[m]), qf1(y, chan_if[m])
        f_r = qf1(y, rule_t[m])
        f_o = qf1(y, np.maximum(chan_if[m], rule_t[m]))
        best = max([("全局IF", f_g), ("通道IF", f_c), ("规则", f_r), ("OR", f_o)],
                   key=lambda t: t[1])[0]
        tag = "强" if ch in STRONG else "弱"
        print(f"{ch:<12}{tag:<9}{f_g:>9.4f}{f_c:>9.4f}{f_r:>9.4f}{f_o:>14.4f}"
              f"{best:>10}")
        if ch not in STRONG:
            gain = f_o - f_c
            flag = "  <-- 补规则层大赚" if gain > 0.1 else (
                "  <-- 补规则层有赚" if gain > 0 else "  (补规则层反而略差)")
            print(f"{'':<12}弱通道补上 max(IF, rule) = {f_o:.4f}，相对通道 IF {gain:+.4f}{flag}")

    # ---- 3. 端到端：STRONG_CHANNELS 扩到全通道 ----
    print()
    print("=" * 86)
    print("3. 端到端：把 STRONG_CHANNELS 从 3 个扩到全部 9 个通道")
    print("=" * 86)

    def scheme(strong_set, if_pred):
        p = np.zeros(len(test_df), dtype=int)
        for i, ch in enumerate(ch_te):
            if ch in strong_set:
                p[i] = max(if_pred[i], rule_t[i])
            else:
                p[i] = global_if[i]
        return p

    # 复现生产现状（口径近似：规则层用实验 k，IF 用生产 contamination）
    p_prod = scheme(STRONG, chan_if)
    f_prod = qf1(yt, p_prod)
    p_all = scheme(set(np.unique(ch_te)), chan_if)
    f_all = qf1(yt, p_all)

    # 生产原口径（nv>=4）
    rule_prod = (nv_te >= 4).astype(int)
    p_prod_orig = np.zeros(len(test_df), dtype=int)
    for i, ch in enumerate(ch_te):
        if ch in STRONG:
            p_prod_orig[i] = max(chan_if[i], rule_prod[i])
        else:
            p_prod_orig[i] = global_if[i]

    print(f"  生产现状复现（强3通道OR+ 弱通道全局IF，规则 nv>=4）= "
          f"{qf1(yt, p_prod_orig):.4f}")
    print(f"  规则层改口径后（nv>={bk}）                    = {f_prod:.4f}")
    print(f"  再把弱通道也纳入 OR（全部 9 通道）              = {f_all:.4f}")
    print(f"\n  两项纯代码改动的合计增益 = {f_all - qf1(yt, p_prod_orig):+.4f}")
    print("  两项均**不改动算法**（IF 与规则层原样，只改接线方式）：")
    print("    (1) 规则层阈值口径与实验对齐 nv>=4 -> nv>=2")
    print("    (2) 弱通道不再绕过规则层，改为 max(IF, rule)")

    # ---- 4. 配对 bootstrap ----
    print()
    print("=" * 86)
    print("4. 配对bootstrap：新接线 vs 生产原口径")
    print("=" * 86)
    rng = np.random.default_rng(42)
    n = len(yt)
    ds = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        ds.append(qf1(yt[idx], p_all[idx]) - qf1(yt[idx], p_prod_orig[idx]))
    ds = np.array(ds)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    print(f"  ΔF1 = {f_all - qf1(yt, p_prod_orig):+.4f}  "
          f"95%CI[{lo:+.4f},{hi:+.4f}]  "
          f"{'显著' if lo > 0 else '不显著(含0)'}")


if __name__ == "__main__":
    main()
