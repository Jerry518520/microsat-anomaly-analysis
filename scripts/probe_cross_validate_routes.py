"""探针：交叉验证两条独立改进路线，并检查是否可叠加 + 一个已知隐患。

路线甲（主理人实测）：IF 输入特征子集（3 维 n_peaks/kurtosis/smooth10_n_peaks）
      test F1 0.6281 -> 0.6846(seed42) / 0.7142±0.0249(5seed)
      机制：18 维里 10 维 val AUC<0.5（反向），IF 随机切分被拖垮
      性质：IF 本身完全不变，仍是无监督

路线乙（algo-diagnose）：规则层参考分布去污染（只用 fit 里的正常段拟合阈值）
      test F1 0.6281 -> 0.7155（叠加特征白名单）
      机制：fit 含 20% 异常段，把异常混进参考分布，3σ 阈值被撑宽
      性质：规则层从「无监督」变成「用训练标签做单类校准」——有语义代价

本脚本要回答三个问题：
  Q1 两条路线是否可叠加？还是都作用在同一处（不叠加）？
  Q2 我路线里的隐患是否真实：CADC0888 从 F1 0.8000 掉到 0.4314，
     因 IF 变强后门控把该通道从 rule 切到了 if，而 test 上 rule 更好。
     若是，是否可用「小样本/强规则通道不切换」护栏消除？
  Q3 叠加后的最优组合 test F1 是多少？配对 bootstrap 是否显著？

纪律：
  - 全部超参（k、contamination、门控算子、特征子集、白名单）只在 val 上选；
  - 净阈值只取 fit_df 内部，绝不触碰 val/test；
  - test 只在配置完全确定后评一次；
  - 护栏阈值（正常段数下限）在 val 上验，不在 test 上调。

用法：
    /d/Python313/python.exe scripts/probe_cross_validate_routes.py
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
SEEDS = [42, 7, 123]

# 路线甲：IF 输入子集（val 上贪心选出，5/5 seed 稳定）
IF_SUBSET = ["n_peaks", "kurtosis", "smooth10_n_peaks"]

# 路线乙的护栏：某通道 fit 内正常段数不足则退回全量阈值（val 上验证零成本）
MIN_NORMAL = 30


def qf1(yt, yp):
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(yt), np.asarray(yp), zero_division=0))


def build_thresholds(fit_df, feature_cols, use_net, min_normal=MIN_NORMAL):
    """净阈值：只用 fit 里的正常段拟合参考分布；正常段不足则退回全量。

    注意：compute_stat_thresholds 返回 {channel: {feature: {...}}}，故必须
    一次性传入「所有通道的正常段」再逐通道取内层。若逐通道传入单通道子集，
    会得到 {ch: {ch: {...}}} 双层嵌套，apply_stat_rules 里
    `col not in ch_th` 恒真 -> 全部特征跳过 -> nv 恒为 0 -> 规则层永久失效。
    """
    if not use_net:
        return compute_stat_thresholds(fit_df, feature_cols)
    normal = fit_df[fit_df["anomaly"] == 0]
    net = compute_stat_thresholds(normal, feature_cols)
    out = {}
    for ch in fit_df["channel"].unique():
        sub = fit_df[fit_df["channel"] == ch]
        n_norm = int((sub["anomaly"] == 0).sum())
        out[ch] = net[ch] if n_norm >= min_normal else compute_stat_thresholds(
            sub, feature_cols)[ch]
    return out


def run(fit_df, val_df, eval_df, feature_cols, if_cols, thresholds, seed):
    """门控端到端。feature_cols 决定规则层用哪些特征；if_cols 决定 IF 输入。"""
    # k 在 val 上选（基于规则层实际使用的 feature_cols）
    nv_v, _ = apply_stat_rules(val_df, thresholds, feature_cols)
    yv = val_df["anomaly"].to_numpy()
    best_k, bk = RULE_K_CANDIDATES[0], -1.0
    for k in RULE_K_CANDIDATES:
        f1 = qf1(yv, (nv_v >= k).astype(int).to_numpy())
        if f1 > bk:
            bk, best_k = f1, k

    def layer(df, ch):
        s = df[df["channel"] == ch]
        nv, _ = apply_stat_rules(s, thresholds, feature_cols)
        rp = (nv >= best_k).astype(int).to_numpy()
        cf = fit_df[fit_df["channel"] == ch]
        if len(cf) < MIN_FIT or if_cols is None:
            return rp, np.zeros(len(s), dtype=int)
        clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                              contamination=0.15, random_state=seed,
                              n_jobs=-1).fit(np.nan_to_num(cf[if_cols].values))
        ip = (clf.predict(np.nan_to_num(s[if_cols].values)) == -1).astype(int)
        return rp, ip

    # 逐通道 contamination 在 val 上选
    best_c = {}
    for ch in sorted(val_df["channel"].unique()):
        cv, cf = val_df[val_df["channel"] == ch], fit_df[fit_df["channel"] == ch]
        if len(cf) < MIN_FIT or len(cv) < MIN_EVAL or if_cols is None:
            best_c[ch] = None
            continue
        ycv = cv["anomaly"].to_numpy()
        Xf_, Xv_ = np.nan_to_num(cf[if_cols].values), np.nan_to_num(cv[if_cols].values)
        bc, bf = CONTAM_GRID[0], -1.0
        for c in CONTAM_GRID:
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=seed, n_jobs=-1).fit(Xf_)
            f1 = qf1(ycv, (clf.predict(Xv_) == -1).astype(int))
            if f1 > bf:
                bf, bc = f1, c
        best_c[ch] = float(bc)

    def preds(df, ch):
        s = df[df["channel"] == ch]
        nv, _ = apply_stat_rules(s, thresholds, feature_cols)
        rp = (nv >= best_k).astype(int).to_numpy()
        cf = fit_df[fit_df["channel"] == ch]
        c = best_c.get(ch)
        if c is None or len(cf) < MIN_FIT or if_cols is None:
            return rp, np.zeros(len(s), dtype=int)
        clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                              contamination=c, random_state=seed, n_jobs=-1
                              ).fit(np.nan_to_num(cf[if_cols].values))
        return rp, (clf.predict(np.nan_to_num(s[if_cols].values)) == -1).astype(int)

    yt, yp, picks, chf1 = [], [], {}, {}
    for ch in sorted(eval_df["channel"].unique()):
        ev = eval_df[eval_df["channel"] == ch]
        ye = ev["anomaly"].to_numpy()
        rp_e, ip_e = preds(eval_df, ch)
        rp_v, ip_v = preds(val_df, ch)
        ycv = val_df[val_df["channel"] == ch]["anomaly"].to_numpy()
        cand = {"rule": qf1(ycv, rp_v), "if": qf1(ycv, ip_v),
                "AND": qf1(ycv, rp_v & ip_v), "OR": qf1(ycv, rp_v | ip_v)}
        op = max(cand, key=lambda k: cand[k])
        picks[ch] = op
        p = {"rule": rp_e, "if": ip_e, "AND": (rp_e & ip_e).astype(int),
             "OR": (rp_e | ip_e).astype(int)}[op]
        yt.append(ye)
        yp.append(np.asarray(p, dtype=int))
        chf1[ch] = qf1(ye, p)
    return np.concatenate(yt), np.concatenate(yp), picks, chf1, best_k


def paired_boot(yt, pa, pb, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    m = len(yt)
    ds = []
    for _ in range(n):
        i = rng.integers(0, m, m)
        ds.append(qf1(yt[i], pb[i]) - qf1(yt[i], pa[i]))
    ds = np.array(ds)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return lo, hi, float((ds > 0).mean())


def main():
    fit_df, val_df, test_df, feature_cols = load_split()
    th_full = build_thresholds(fit_df, feature_cols, use_net=False)
    th_net = build_thresholds(fit_df, feature_cols, use_net=True)

    # 净阈值护栏是否触发
    print("=" * 86)
    print("净阈值护栏检查（正常段数 < 30 则退回全量）")
    print("=" * 86)
    print(f"{'通道':<12}{'fit_n':>7}{'fit正常':>9}{'>=30?':>7}  阈值来源")
    print("-" * 60)
    for ch in sorted(fit_df["channel"].unique()):
        sub = fit_df[fit_df["channel"] == ch]
        nn = int((sub["anomaly"] == 0).sum())
        src = "净阈值" if nn >= MIN_NORMAL else "全量(护栏)"
        print(f"{ch:<12}{len(sub):>7}{nn:>9}{'是' if nn >= MIN_NORMAL else '否':>7}  {src}")

    configs = [
        ("A基线    全量阈值+IF18维", th_full, feature_cols, feature_cols),
        ("B路线甲  全量阈值+IF3维 ", th_full, feature_cols, IF_SUBSET),
        ("C路线乙  净阈值  +IF18维", th_net, feature_cols, feature_cols),
        ("D叠加    净阈值  +IF3维 ", th_net, feature_cols, IF_SUBSET),
    ]

    print()
    print("=" * 86)
    print("Q1/Q3 四种配置的端到端 test F1（3 个 IF seed）")
    print("=" * 86)
    print(f"{'配置':<26}" + "".join(f"{'seed'+str(s):>10}" for s in SEEDS)
          + f"{'均值':>10}{'std':>9}")
    print("-" * 86)
    store = {}
    for name, th, fcols, icols in configs:
        f1s, picks0, chf1_0 = [], None, None
        for s in SEEDS:
            yt, yp, pk, cf1, bk = run(fit_df, val_df, test_df, fcols, icols, th, s)
            f1s.append(qf1(yt, yp))
            if s == SEEDS[0]:
                picks0, chf1_0, yt0, yp0, bk0 = pk, cf1, yt, yp, bk
        m, sd = np.mean(f1s), np.std(f1s, ddof=1)
        store[name] = dict(f1s=f1s, mean=m, std=sd, picks=picks0, chf1=chf1_0,
                           yt=yt0, yp=yp0, k=bk0)
        print(f"{name:<26}" + "".join(f"{f:>10.4f}" for f in f1s)
              + f"{m:>10.4f}{sd:>9.4f}")

    A, B, C, D = (store[c[0]] for c in configs)
    print()
    print("=" * 86)
    print("配对bootstrap（相对 A 基线；test 只在配置定死后评）")
    print("=" * 86)
    for cfg_name in [c[0] for c in configs][1:]:
        nd = store[cfg_name]
        lo, hi, pos = paired_boot(A["yt"], A["yp"], nd["yp"])
        sig = "显著" if lo > 0 else "不显著(含0)"
        print(f"  {cfg_name:<26} ΔF1={nd['mean']-A['mean']:+.4f}  "
              f"95%CI[{lo:+.4f},{hi:+.4f}]  为正概率={pos*100:.1f}%  {sig}")

    # 叠加是否等于相加
    print()
    print("=" * 86)
    print("Q1 叠加检验：ΔB + ΔC 是否≈ ΔD")
    print("=" * 86)
    dB, dC, dD = B["mean"] - A["mean"], C["mean"] - A["mean"], D["mean"] - A["mean"]
    print(f"  ΔB(路线甲: IF特征子集) = {dB:+.4f}")
    print(f"  ΔC(路线乙: 净阈值)     = {dC:+.4f}")
    print(f"  相加预期= {dB + dC:+.4f}")
    print(f"  实测 ΔD  = {dD:+.4f}")
    print(f"  交互项(实测-相加) = {dD - (dB + dC):+.4f}")
    print("  => " + ("两路线作用在不同机制上，可叠加" if dD - (dB + dC) > 0.02
                     else "两路线高度重叠，叠加收益有限"))

    # ---------------- Q2 CADC0888 隐患 ----------------
    print()
    print("=" * 86)
    print("Q2 隐患核验：CADC0888 为何退化（规则层 0.8000 是最干净的通道）")
    print("=" * 86)
    print(f"{'配置':<26}{'0888算子':>10}{'0888 F1':>10}   该通道 test 异常段= 12")
    print("-" * 70)
    for cfg_name in [c[0] for c in configs]:
        nd = store[cfg_name]
        print(f"{cfg_name:<26}{nd['picks']['CADC0888']:>10}"
              f"{nd['chf1']['CADC0888']:>10.4f}")

    print()
    print("逐通道 ΔF1（B 相对 A，seed=42）—— 看退化是否只集中在 0888")
    print("=" * 86)
    print(f"{'通道':<12}{'n_anom':>8}{'A基线':>9}{'B路线甲':>10}{'差值':>9}  算子变化")
    print("-" * 74)
    cnt = test_df.groupby("channel")["anomaly"].sum().to_dict()
    for ch in sorted(A["chf1"]):
        d = B["chf1"][ch] - A["chf1"][ch]
        opc = "" if A["picks"][ch] == B["picks"][ch] else f"{A['picks'][ch]}->{B['picks'][ch]}"
        flag = "  <-- 退化" if d < -0.05 else ""
        print(f"{ch:<12}{cnt[ch]:>8}{A['chf1'][ch]:>9.4f}{B['chf1'][ch]:>10.4f}"
              f"{d:>+9.4f}  {opc}{flag}")

    # ---------------- Q3 护栏能否消除退化 ----------------
    print()
    print("=" * 86)
    print("Q2b 护栏试验：规则层在 val 上 F1 显著占优的通道，禁止门控切离 rule")
    print("=" * 86)
    print("说明：这是「融合策略」层面的护栏，不动 IsolationForest 也不动规则判据。")
    print("      护栏条件本身在 val 上判定（看 val 上rule 是否显著优于 if）。")
    for s in SEEDS[:1]:
        for tag, th, icols in [("路线甲", th_full, IF_SUBSET)]:
            # 复跑并在 val 上计算 rule 相对 if 的优势
            yt, yp, picks, chf1, bk = run(fit_df, val_df, test_df, feature_cols,
                                          icols, th, s)
            print(f"\n  [{tag}] seed={s}  无护栏 F1={qf1(yt, yp):.4f}")
            print(f"  各通道 val 上 rule vs if：")
            for ch in sorted(val_df["channel"].unique()):
                cv = val_df[val_df["channel"] == ch]
                nv_v, _ = apply_stat_rules(cv, th, feature_cols)
                rp = (nv_v >= bk).astype(int).to_numpy()
                cf = fit_df[fit_df["channel"] == ch]
                if len(cf) < MIN_FIT:
                    continue
                Xf_ = np.nan_to_num(cf[icols].values)
                Xv_ = np.nan_to_num(cv[icols].values)
                bestf = max(
                    qf1(cv["anomaly"].to_numpy(),
                        (IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                         contamination=c, random_state=s, n_jobs=-1
                                         ).fit(Xf_).predict(Xv_) == -1).astype(int))
                    for c in CONTAM_GRID)
                gap = qf1(cv["anomaly"].to_numpy(), rp) - bestf
                mark = "<- 护栏候选" if gap > 0.15 else ""
                print(f"    {ch}: rule={qf1(cv['anomaly'].to_numpy(), rp):.4f} "
                      f"if_best={bestf:.4f} gap={gap:+.4f} "
                      f"实选={picks[ch]} {mark}")


if __name__ == "__main__":
    main()
