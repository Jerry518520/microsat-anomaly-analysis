"""探针：只换IF 的输入特征子集，端到端 F1 能到多少？（算法不动）

背景（scripts/probe_if_dilution.py 实测）：
    IF + 18 维全集        test AUC = 0.6557
    IF + 3 维 val 最优子集 test AUC = 0.8673   <- 算法完全没变
    18 维里有 10 维 val AUC < 0.50（反向），IF 的随机切分被拖累。

本脚本回答唯一问题：
    把门控融合里的 IF 层输入从 18 维换成「val 上贪心选出的子集」，
    端到端 test F1 能从 0.6281 提升到多少？
    **算法仍是 IsolationForest + 分段检验，不换模型。**

纪律：
  - 特征子集在 val 上贪心选（不看 test）；
  - 规则层仍用全部 18 维（规则层的 k 阈值依赖各特征统计量，不能动）；
  - IF 的 contamination 仍在 val 上逐通道选；
  - test 只在配置完全确定后评一次。
  - 同时报告 F1 与 AUC，并给 bootstrap 95% CI，与 0.6281 做配对比较。

用法：
    /d/Python313/python.exe scripts/probe_gate_feature_subset.py
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

from src.experiments.framework import evaluate, load_split  # noqa: E402
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


def qf1(yt, yp):
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(yt), np.asarray(yp), zero_division=0))


def if_auc(Xf, yf, Xe, ye, c):
    clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                          contamination=c, random_state=42, n_jobs=-1).fit(Xf)
    return roc_auc_score(ye, -clf.decision_function(Xe))


def select_subset_greedy(Xf, yf, Xv, yv, feature_cols, max_k=6):
    """贪心前向：每次加入使 val AUC 提升最多的特征。test 完全不参与。"""
    remaining = set(range(len(feature_cols)))
    chosen, best_val = [], -1.0
    traj = []
    for _ in range(max_k):
        cb, ci = -1.0, None
        for i in list(remaining):
            idx = chosen + [i]
            a = max(if_auc(Xf[:, idx], yf, Xv[:, idx], yv, c) for c in CONTAM_GRID)
            if a > cb:
                cb, ci = a, i
        if cb <= best_val + 1e-6 and chosen:
            print(f"    [贪心提前停止：加第 {len(chosen)+1} 维无提升]")
            break
        chosen.append(ci)
        remaining.discard(ci)
        best_val = cb
        traj.append((list(chosen), cb))
    return traj


def build_gate_pred(fit_df, val_df, eval_df, feature_cols, if_cols, thresholds, best_k):
    """在 val 上逐通道选算子（rule/if/AND/OR），再套到 eval_df 上。全程不看 eval 标签。"""
    # 逐通道 contamination 在 val 上选
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
                                  contamination=c, random_state=42, n_jobs=-1).fit(Xf_)
            f1 = qf1(yv, (clf.predict(Xv_) == -1).astype(int))
            if f1 > bf:
                bf, bc = f1, c
        best_c[ch] = float(bc)

    picks, y_pred, y_true = {}, [], []
    for ch in sorted(eval_df["channel"].unique()):
        ev, cv, cf = eval_df[eval_df["channel"] == ch], val_df[val_df["channel"] == ch], fit_df[fit_df["channel"] == ch]
        y_ev = ev["anomaly"].to_numpy()
        nv_ev, _ = apply_stat_rules(ev, thresholds, feature_cols)
        rule_pred = (nv_ev >= best_k).astype(int).to_numpy()

        c = best_c.get(ch)
        if c is None or len(cf) < MIN_FIT:
            if_pred = np.zeros(len(ev), dtype=int)
        else:
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=42, n_jobs=-1
                                  ).fit(np.nan_to_num(cf[if_cols].values))
            if_pred = (clf.predict(np.nan_to_num(ev[if_cols].values)) == -1).astype(int)

        nv_v, _ = apply_stat_rules(cv, thresholds, feature_cols)
        rp = (nv_v >= best_k).astype(int).to_numpy()
        yv = cv["anomaly"].to_numpy()
        if c is None or len(cf) < MIN_FIT:
            ip = np.zeros(len(cv), dtype=int)
        else:
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=42, n_jobs=-1
                                  ).fit(np.nan_to_num(cf[if_cols].values))
            ip = (clf.predict(np.nan_to_num(cv[if_cols].values)) == -1).astype(int)
        cand = {"rule": qf1(yv, rp), "if": qf1(yv, ip),
                "AND": qf1(yv, rp & ip), "OR": qf1(yv, rp | ip)}
        op = max(cand, key=lambda k: cand[k])
        picks[ch] = (op, cand[op])

        p = {"rule": rule_pred, "if": if_pred,
             "AND": (rule_pred & if_pred).astype(int),
             "OR": (rule_pred | if_pred).astype(int)}[op]
        y_pred.append(np.asarray(p, dtype=int))
        y_true.append(y_ev)
    return np.concatenate(y_true), np.concatenate(y_pred), picks


def main():
    fit_df, val_df, test_df, feature_cols = load_split()
    Xf = np.nan_to_num(fit_df[feature_cols].values)
    yf = fit_df["anomaly"].to_numpy()
    Xv = np.nan_to_num(val_df[feature_cols].values)
    yv = val_df["anomaly"].to_numpy()

    thresholds = compute_stat_thresholds(fit_df, feature_cols)
    best_k = RULE_K_CANDIDATES[0]
    bk, bf1 = best_k, -1.0
    nv_v, _ = apply_stat_rules(val_df, thresholds, feature_cols)
    for k in RULE_K_CANDIDATES:
        f1 = qf1(yv, (nv_v >= k).astype(int).to_numpy())
        if f1 > bf1:
            bf1, bk = f1, k
    best_k = bk
    print(f"[规则层] best_k = {best_k} (val F1={bf1:.4f})  —— 规则层始终用全部 18 维\n")

    results = {}

    # ============ 配置 A：IF 用全部 18 维（现状基线） ============
    print("=" * 82)
    print("配置 A（现状基线）：IF 输入 = 全部 18 维")
    print("=" * 82)
    yte, yp, picksA = build_gate_pred(fit_df, val_df, test_df, feature_cols,
                                      feature_cols, thresholds, best_k)
    resA = evaluate(yte, yp)
    results["if18"] = (yte, yp, resA, picksA)
    print(f"  test F1 = {resA['f1']:.4f}  CI{resA.get('f1_ci95')}")
    print(f"  门控配方: " + ", ".join(f"{k}={v[0]}" for k, v in picksA.items()))

    # ============ 配置 B：IF 用 val 贪心子集 ============
    print()
    print("=" * 82)
    print("配置 B：IF 输入 = val 上贪心选出的子集（算法不变）")
    print("=" * 82)
    print("  贪心轨迹（只看 val AUC）：")
    traj = select_subset_greedy(Xf, yf, Xv, yv, feature_cols, max_k=8)
    best_idx, best_val_auc = max(traj, key=lambda t: t[1])
    if_cols = [feature_cols[i] for i in best_idx]
    for idx, va in traj:
        print(f"    k={len(idx)}  val AUC={va:.4f}  {[feature_cols[i] for i in idx]}")
    print(f"\n  选定 IF 输入子集（{len(if_cols)} 维）: {if_cols}")
    print(f"  其 val AUC = {best_val_auc:.4f}")

    yte2, yp2, picksB = build_gate_pred(fit_df, val_df, test_df, feature_cols,
                                        if_cols, thresholds, best_k)
    resB = evaluate(yte2, yp2)
    results["ifsubset"] = (yte2, yp2, resB, picksB)
    print(f"\n  test F1 = {resB['f1']:.4f}  CI{resB.get('f1_ci95')}")
    print(f"  门控配方: " + ", ".join(f"{k}={v[0]}" for k, v in picksB.items()))

    # ============ 配对 bootstrap 比较 ============
    print()
    print("=" * 82)
    print("配对bootstrap：配置 B 相对配置 A 的 F1 差值")
    print("=" * 82)
    assert (yte == yte2).all(), "两配置评估集不一致，配对比较无效"
    d = yp2.astype(int) - yp.astype(int)
    rng = np.random.default_rng(42)
    n = len(yte)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        diffs.append(qf1(yte[idx], yp2[idx]) - qf1(yte[idx], yp[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  观测差值 = {resB['f1'] - resA['f1']:+.4f}")
    print(f"  配对 95% CI = [{lo:+.4f}, {hi:+.4f}]  含0? {'是（不显著）' if lo < 0 < hi else '否（显著）'}")
    print(f"  预测翻转数 = {int(d.sum())} / {n}")

    # ============ 汇总 ============
    print()
    print("=" * 82)
    print("汇总")
    print("=" * 82)
    print(f"{'配置':<40}{'IF 维数':>9}{'test F1':>10}{'Precision':>11}{'Recall':>9}")
    print("-" * 79)
    for name, (ye, yp, res, _) in results.items():
        ncol = len(feature_cols) if name == "if18" else len(if_cols)
        print(f"{name:<40}{ncol:>9}{res['f1']:>10.4f}"
              f"{res.get('precision', float('nan')):>11.4f}{res.get('recall', float('nan')):>9.4f}")


if __name__ == "__main__":
    main()
