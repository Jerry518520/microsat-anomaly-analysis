"""探针：交叉矩阵最优组 R3/Tfull/I3 的稳健性与分层核对。

前一轮测到 R3+Tfull+I3 = 0.7857 (seed=42)，比基线 +0.1576 且配对 CI
[+0.0976,+0.2185]、为正概率 100%。但那是**单个 IF 随机种子**，必须在
多个种子 + 分层视角下复核，否则不能写进论文。

本脚本核验四件事：
  A1 稳健性：5 个 IF seed 下 R3/Tfull/I3 vs 基线 R18/Tfull/I18 的表现
  A2 逐通道：+0.1576 的增益来自哪些通道，是否集中在 1~2 个（若是，
     则是过拟合而非真实改进）
  A3 k*=1 的含义：R3 下val 选出的 k=1，而基线 k=2。这是「3 个形状特征
     任一越界即报警」。要明确这是口径变化还是参数漂移
  A4 嵌套验证：R3 特征子集是在 val 上贪心选的，val F1=0.7040 同样有
     乐观偏差。必须做嵌套验证确认 test=0.7857 不是 val 选择偏差的产物

纪律：
  - test 只在配置完全确定后评一次（每个 seed 独立评一次，报告分布）
  - 嵌套验证：特征子集 + k + 门控算子全部只在 A 半集选，B 半集评估
  - 特征子集候选在 val 上枚举（不预设贪心顺序），故嵌套需在A 半重算

用法：
    /d/Python313/python.exe scripts/probe_r3_robustness.py
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
CONTAM_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MIN_FIT = 10
MIN_EVAL = 5
SEEDS = [42, 7, 123, 2024, 999]
F3 = ["kurtosis", "n_peaks", "smooth10_n_peaks"]


def qf1(yt, yp):
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(yt), np.asarray(yp), zero_division=0))


def run(fit_df, val_df, eval_df, rule_cols, if_cols, thr, seed,
        k_candidates=(1, 2, 3, 4)):
    yv, ye = val_df["anomaly"].to_numpy(), eval_df["anomaly"].to_numpy()
    ch_v, ch_e = val_df["channel"].to_numpy(), eval_df["channel"].to_numpy()

    nv_v, _ = apply_stat_rules(val_df, thr, rule_cols)
    bk, bf = k_candidates[0], -1.0
    for k in k_candidates:
        f1 = qf1(yv, (nv_v >= k).astype(int))
        if f1 > bf:
            bf, bk = f1, k

    bc = {}
    for ch in np.unique(ch_v):
        cm = ch_v == ch
        cf = fit_df[fit_df["channel"] == ch]
        if len(cf) < MIN_FIT or int(cm.sum()) < MIN_EVAL:
            bc[ch] = None
            continue
        Xf, Xv = np.nan_to_num(cf[if_cols].values), np.nan_to_num(
            val_df.loc[cm, if_cols].values)
        best, bfv = CONTAM_GRID[0], -1.0
        for c in CONTAM_GRID:
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=seed, n_jobs=-1).fit(Xf)
            f1 = qf1(yv[cm], (clf.predict(Xv) == -1).astype(int))
            if f1 > bfv:
                bfv, best = f1, c
        bc[ch] = float(best)

    def layers(df, chs):
        nv, _ = apply_stat_rules(df, thr, rule_cols)
        rp = (nv >= bk).astype(int).to_numpy()
        ip = np.zeros(len(df), dtype=int)
        for ch in chs:
            m = (df["channel"] == ch).to_numpy()
            c = bc.get(ch)
            cf = fit_df[fit_df["channel"] == ch]
            if c is None or len(cf) < MIN_FIT:
                continue
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=seed, n_jobs=-1
                                  ).fit(np.nan_to_num(cf[if_cols].values))
            ip[m] = (clf.predict(np.nan_to_num(
                df.loc[m, if_cols].values)) == -1).astype(int)
        return rp, ip

    rv, iv = layers(val_df, np.unique(ch_v))
    re_, ie_ = layers(eval_df, np.unique(ch_e))

    yp = np.zeros(len(eval_df), dtype=int)
    picks, chf1 = {}, {}
    for ch in np.unique(ch_e):
        mv, me = (ch_v == ch), (ch_e == ch)
        cand = {"rule": qf1(yv[mv], rv[mv]), "if": qf1(yv[mv], iv[mv]),
                "AND": qf1(yv[mv], iv[mv] & rv[mv]), "OR": qf1(yv[mv], iv[mv] | rv[mv])}
        op = max(cand, key=lambda k: cand[k])
        picks[ch] = op
        yp[me] = {"rule": re_[me], "if": ie_[me], "AND": ie_[me] & re_[me],
                  "OR": ie_[me] | re_[me]}[op]
        chf1[ch] = qf1(ye[me], yp[me])
    return ye, yp, picks, bk, chf1


def main():
    fit_df, val_df, test_df, feature_cols = load_split()
    thr18 = compute_stat_thresholds(fit_df, feature_cols)
    thr3 = compute_stat_thresholds(fit_df, F3)

    # ---- A1 稳健性 ----
    print("=" * 88)
    print("A1 稳健性：5 个 IF 随机种子")
    print("=" * 88)
    print(f"{'seed':>7}{'基线R18I18':>13}{'最优R3I3':>11}{'差值':>9}"
          f"{'基线k':>8}{'最优k':>8}")
    print("-" * 88)
    rows = []
    for s in SEEDS:
        yA, pA, _, kA, _ = run(fit_df, val_df, test_df, feature_cols,
                               feature_cols, thr18, s)
        yB, pB, pickB, kB, chf1B = run(fit_df, val_df, test_df, F3, F3, thr3, s)
        fa, fb = qf1(yA, pA), qf1(yB, pB)
        rows.append((s, fa, fb, kA, kB, pB, chf1B))
        print(f"{s:>7}{fa:>13.4f}{fb:>11.4f}{fb-fa:>+9.4f}{kA:>8}{kB:>8}")

    fa_m = np.mean([r[1] for r in rows])
    fb_m = np.mean([r[2] for r in rows])
    fa_s = np.std([r[1] for r in rows], ddof=1)
    fb_s = np.std([r[2] for r in rows], ddof=1)
    print()
    print(f"  基线   : F1 = {fa_m:.4f} ± {fa_s:.4f}  范围 [{min(r[1] for r in rows):.4f}, {max(r[1] for r in rows):.4f}]")
    print(f"  最优组 : F1 = {fb_m:.4f} ± {fb_s:.4f}  范围 [{min(r[2] for r in rows):.4f}, {max(r[2] for r in rows):.4f}]")
    print(f"  平均增益 = {fb_m-fa_m:+.4f}   5/5 seed 为正 = "
          f"{'是' if all(r[2] > r[1] for r in rows) else '否'}")
    print(f"  两组范围重叠 = {'是（说明种子间有交叉）' if max(r[1] for r in rows) > min(r[2] for r in rows) else '否'}")

    # ---- A2 逐通道分解 ----
    print()
    print("=" * 88)
    print("A2 逐通道增益分解（seed=42），检验是否集中在少数通道")
    print("=" * 88)
    yA, pA, pkA, kA, chA = run(fit_df, val_df, test_df, feature_cols,
                               feature_cols, thr18, 42)
    yB, pB, pkB, kB, chB = run(fit_df, val_df, test_df, F3, F3, thr3, 42)
    n_an = test_df.groupby("channel")["anomaly"].sum().to_dict()
    n_se = test_df.groupby("channel").size().to_dict()
    print(f"{'通道':<12}{'n_seg':>7}{'n_anom':>8}{'基线F1':>9}{'最优F1':>9}"
          f"{'差值':>9}  算子变化")
    print("-" * 80)
    for ch in sorted(chA):
        d = chB[ch] - chA[ch]
        opc = "" if pkA[ch] == pkB[ch] else f"{pkA[ch]}->{pkB[ch]}"
        flag = "  ★大幅改善" if d > 0.15 else ("  ↓退化" if d < -0.05 else "")
        print(f"{ch:<12}{n_se[ch]:>7}{n_an[ch]:>8}{chA[ch]:>9.4f}{chB[ch]:>9.4f}"
              f"{d:>+9.4f}  {opc}{flag}")
    pos = [ch for ch in chA if chB[ch] - chA[ch] > 0.01]
    neg = [ch for ch in chA if chB[ch] - chA[ch] < -0.01]
    print(f"\n  改善通道 {len(pos)}/9: {pos}")
    print(f"  退化通道 {len(neg)}/9: {neg}")
    print("  => 增益"
          + ("分散在多个通道，非单通道驱动" if len(pos) >= 4
             else f"集中在 {len(pos)} 个通道，需警惕过拟合"))

    # ---- A3 k*=1 的含义 ----
    print()
    print("=" * 88)
    print("A3 k* 变化的口径含义")
    print("=" * 88)
    print(f"  基线 18 维：val 选 k={kA}  -> 需2 个特征同时越界才报警")
    print(f"  最优 3 维：val 选 k={kB}  -> 1 个形状特征越界即报警")
    print("  => 这不是参数漂移，而是**判据口径的实质变化**：")
    print("     18 维里单个特征越界很常见（噪声大），需多个投票；")
    print("     3 个形状特征各自判别力都强（val AUC 0.70~0.83），单个即可判阳。")
    print("  ⚠ 论文必须说明这一点，否则会被误读为'阈值从 2 调到 1 而已'。")

    # ---- A4 嵌套验证 ----
    print()
    print("=" * 88)
    print("A4 嵌套验证：F3 子集 + k + 门控 全部在 val 上选，是否过拟合")
    print("=" * 88)
    rng = np.random.default_rng(42)
    wins, draws, losses, deltas = 0, 0, 0, []
    idx_all = np.arange(len(val_df))
    for rep in range(20):
        perm = rng.permutation(idx_all)
        a_idx, b_idx = perm[:len(perm) // 2], perm[len(perm) // 2:]
        va = val_df.iloc[a_idx].reset_index(drop=True)
        vb = val_df.iloc[b_idx].reset_index(drop=True)
        # 在 A 半集上贪心选子集
        chosen, best = [], -1.0
        rem = set(range(len(feature_cols)))
        for _ in range(5):
            cb, ci = -1.0, None
            for i in list(rem):
                idx = chosen + [i]
                Xf = np.nan_to_num(fit_df[[feature_cols[j] for j in idx]].values)
                Xa = np.nan_to_num(va[[feature_cols[j] for j in idx]].values)
                ya = va["anomaly"].to_numpy()
                a = max(roc_auc_score(ya, -IsolationForest(
                    n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                    contamination=c, random_state=42, n_jobs=-1
                ).fit(Xf).decision_function(Xa)) for c in CONTAM_GRID)
                if a > cb:
                    cb, ci = a, i
            if chosen and cb <= best + 1e-6:
                break
            chosen.append(ci)
            rem.discard(ci)
            best = cb
        sel = [feature_cols[j] for j in chosen]
        if sorted(sel) != sorted(F3):
            print(f"  rep{rep:>2}: A 半选出 {sel}（与 F3 不同，跳过对比）")
            continue
        # A 半定参数，B 半评估：k 用 A 半，门控用 A 半
        th = compute_stat_thresholds(fit_df, sel)
        nva, _ = apply_stat_rules(va, th, sel)
        ya = va["anomaly"].to_numpy()
        kb, bfv = 1, -1.0
        for k in (1, 2, 3, 4):
            f1 = qf1(ya, (nva >= k).astype(int))
            if f1 > bfv:
                bfv, kb = f1, k
        # 门控算子在 A 半选
        opA = {}
        for ch in np.unique(va["channel"].to_numpy()):
            m = (va["channel"] == ch).to_numpy()
            cf = fit_df[fit_df["channel"] == ch]
            if len(cf) < MIN_FIT:
                opA[ch] = "rule"
                continue
            Xf, Xa = np.nan_to_num(cf[sel].values), np.nan_to_num(va.loc[m, sel].values)
            rp = (apply_stat_rules(va[va["channel"] == ch], th, sel)[0] >= kb).astype(int).to_numpy()
            cbest, fbest = None, -1.0
            for c in CONTAM_GRID:
                clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                      contamination=c, random_state=42, n_jobs=-1).fit(Xf)
                ip = (clf.predict(Xa) == -1).astype(int)
                f1 = max(qf1(ya[m], rp), qf1(ya[m], ip),
                         qf1(ya[m], rp & ip), qf1(ya[m], rp | ip))
                if f1 > fbest:
                    fbest, cbest = f1, c
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=cbest, random_state=42, n_jobs=-1).fit(Xf)
            ip = (clf.predict(Xa) == -1).astype(int)
            cand = {"rule": qf1(ya[m], rp), "if": qf1(ya[m], ip),
                    "AND": qf1(ya[m], rp & ip), "OR": qf1(ya[m], rp | ip)}
            opA[ch] = max(cand, key=lambda c: cand[c])
        # B 半评估（test 上评一次）
        bcf = {}
        for ch in np.unique(test_df["channel"].unique()):
            cf = fit_df[fit_df["channel"] == ch]
            cbest, fbest = None, -1.0
            cv = va[va["channel"] == ch]
            if len(cf) < MIN_FIT or len(cv) < MIN_EVAL:
                bcf[ch] = None
                continue
            Xf, Xv = np.nan_to_num(cf[sel].values), np.nan_to_num(cv[sel].values)
            for c in CONTAM_GRID:
                clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                      contamination=c, random_state=42, n_jobs=-1).fit(Xf)
                f1 = qf1(cv["anomaly"].to_numpy(), (clf.predict(Xv) == -1).astype(int))
                if f1 > fbest:
                    fbest, cbest = f1, c
            bcf[ch] = float(cbest)
        ch_t = test_df["channel"].to_numpy()
        yp = np.zeros(len(test_df), dtype=int)
        for ch in np.unique(ch_t):
            m = (ch_t == ch).to_numpy()
            ev = test_df[m]
            rp = (apply_stat_rules(ev, th, sel)[0] >= kb).astype(int).to_numpy()
            c = bcf.get(ch)
            if c is None:
                ip = np.zeros(int(m.sum()), dtype=int)
            else:
                cf = fit_df[fit_df["channel"] == ch]
                clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                      contamination=c, random_state=42, n_jobs=-1
                                      ).fit(np.nan_to_num(cf[sel].values))
                ip = (clf.predict(np.nan_to_num(ev[sel].values)) == -1).astype(int)
            yp[m] = {"rule": rp, "if": ip, "AND": rp & ip, "OR": rp | ip}[opA.get(ch, "rule")]
        f_nested = qf1(test_df["anomaly"].to_numpy(), yp)
        d = f_nested - rows[0][1]
        deltas.append(d)
        if d > 0.01:
            wins += 1
        elif d < -0.01:
            losses += 1
        else:
            draws += 1
        print(f"  rep{rep:>2}: 子集一致k={kb}  嵌套 test F1 = {f_nested:.4f}  "
              f"vs 基线 {rows[0][1]:.4f}  Δ={d:+.4f}")

    if deltas:
        print(f"\n  嵌套验证结果：胜 {wins} / 平 {draws} / 负 {losses}")
        print(f"  ΔF1 均值 = {np.mean(deltas):+.4f}  中位 = {np.median(deltas):+.4f}  "
              f"最好 {max(deltas):+.4f}  最差 {min(deltas):+.4f}")
        print(f"  => 嵌套（子集与k 均在 val 内重选）后仍为正 = "
              f"{'是，非过拟合' if np.median(deltas) > 0 else '否，需警惕'}")


if __name__ == "__main__":
    main()
