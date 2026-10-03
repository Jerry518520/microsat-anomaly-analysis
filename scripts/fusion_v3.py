"""
v3 融合策略实验 —— 回答"规则层 + Isolation Forest 到底怎么融合 F1 最高"（主理人实做）

================================================================================
设计要点（铁律约束，违反即作废）
================================================================================
* 算法锁定：**Isolation Forest + 分段检验**，不换模型。
* 此前 Agent C 只试了二值 AND / OR（0.6038 / 0.5545）。本实验引入两类此前
  完全没用上的**连续分数**：
    - 规则层连续严重度：apply_stat_rules 返回的 nv（每段违规特征数，0~18）
    - IF 连续异常分：-decision_function（越高越异常，按通道计算）
  并系统比较以下融合策略，全部段级 SegF1：
    1. rule_only        —— 参考基线（纯规则）
    2. if_only          —— 参考基线（纯分通道 IF）
    3. if_and_rule      —— 参考（硬 AND，0.6038）
    4. if_or_rule       —— 参考（硬 OR，0.5545）
    5. soft_global      —— 全局软加权：fused = w*ifn + (1-w)*rulen，阈值 τ
    6. soft_perchannel  —— 逐通道软加权：每通道各自 (w_ch, τ_ch)
    7. gate_perchannel  —— 逐通道门控：每通道在 {rule,if,AND,OR} 中选 val F1 最高者
    8. soft_calibrated  —— 逻辑回归融合 [rulen, ifn] -> 概率，阈值调 τ

* 超参纪律（铁律 2）：
    - k（规则违规阈值）、c（IF contamination）在 val 上选（与 ablation 同口径：
      N_ESTIMATORS=100, MAX_SAMPLES=128）。
    - 连续分数归一化的 min/max 在 **fit** 上算（不泄露 val/test）。
    - 权重 w、阈值 τ、逐通道选择、逻辑回归阈值 —— 一律只在 val 上选。
    - test 全程只在"最终评估"块使用一次。
* 效率纪律：val 选参阶段用轻量 f1（无 bootstrap）；bootstrap 95% CI 只在
  最终 test/val 评估时由 framework.evaluate 给出（铁律 1.3 要求 F1 带 CI）。
* 每个 JSON 经 framework.save_result() 注入 meta 溯源块（铁律 3）。
* 只写 data/results/v3/fusion.json，不碰任何已有文件（铁律 4）。

用法：
    /d/Python313/python.exe scripts/fusion_v3.py
"""

import os
import sys

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score as _f1
from sklearn.model_selection import StratifiedKFold

np.random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments.framework import (  # noqa: E402
    evaluate,
    load_split,
    save_result,
)
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
# ⚠ 下界必须 > 0。归一化分数恒 >= 0，若 tau=0 与判定 `>= tau` 组合，
#   tau=0.0 等价于「全部判异常」——这不是阈值搜索的应有结果，而是网格边界
#   造成的退化解。实测曾有 3 个通道（0884/0886/0890）选中 tau=0.0。
#   0.05 与 TAU_GRID_CAL 的下界一致，不影响任何其他通道的既有选择。
TAU_GRID = [round(x, 3) for x in np.arange(0.05, 1.001, 0.05)]
TAU_GRID_CAL = [round(x, 3) for x in np.arange(0.1, 0.9001, 0.05)]


def quick_f1(yt, yp):
    """val 选参用轻量 F1（无 bootstrap，快）。"""
    return float(_f1(np.asarray(yt), np.asarray(yp), zero_division=0))


# ----------------------------------------------------------------- 基础组件
def rule_nv(df, thresholds, feature_cols):
    nv, _ = apply_stat_rules(df, thresholds, feature_cols)
    return nv.astype(float)


def fit_if_channel(ch_fit, feature_cols, c):
    if len(ch_fit) < MIN_FIT:
        return None
    X = np.nan_to_num(ch_fit[feature_cols].values)
    clf = IsolationForest(
        n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
        contamination=c, random_state=42, n_jobs=-1,
    ).fit(X)
    return clf


def if_score(clf, df_ch, feature_cols):
    if clf is None or len(df_ch) == 0:
        return np.zeros(len(df_ch), dtype=float)
    X = np.nan_to_num(df_ch[feature_cols].values)
    return -clf.decision_function(X)


def select_rule_k(val_df, thresholds, feature_cols):
    best_k, best_f1 = RULE_K_CANDIDATES[0], -1.0
    for k in RULE_K_CANDIDATES:
        yp = (rule_nv(val_df, thresholds, feature_cols) >= k).astype(int).to_numpy()
        f1 = quick_f1(val_df["anomaly"].to_numpy(), yp)
        if f1 > best_f1:
            best_f1, best_k = f1, k
    return int(best_k)


def select_if_contamination(val_df, fit_df, feature_cols):
    best_c = {}
    for ch in val_df["channel"].unique():
        ch_val = val_df[val_df["channel"] == ch]
        ch_fit = fit_df[fit_df["channel"] == ch]
        if len(ch_fit) < MIN_FIT or len(ch_val) < MIN_EVAL:
            best_c[ch] = None
            continue
        y_val = ch_val["anomaly"].to_numpy()
        X_fit = np.nan_to_num(ch_fit[feature_cols].values)
        X_val = np.nan_to_num(ch_val[feature_cols].values)
        best_c_ch, best_f1 = CONTAM_GRID[0], -1.0
        for c in CONTAM_GRID:
            clf = IsolationForest(
                n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                contamination=c, random_state=42, n_jobs=-1,
            ).fit(X_fit)
            p = (clf.predict(X_val) == -1).astype(int)
            f1 = quick_f1(y_val, p)
            if f1 > best_f1:
                best_f1, best_c_ch = f1, c
        best_c[ch] = float(best_c_ch)
    return best_c


# ----------------------------------------------------------------- 归一化（在 fit 上算 min/max）
def fit_normalizers(fit_df, thresholds, feature_cols, if_clfs):
    channels = fit_df["channel"].unique()
    rule_min, rule_max = np.inf, -np.inf
    if_min, if_max = {}, {}
    for ch in channels:
        ch_fit = fit_df[fit_df["channel"] == ch]
        nv = rule_nv(ch_fit, thresholds, feature_cols).to_numpy()
        if len(nv):
            rule_min = min(rule_min, nv.min())
            rule_max = max(rule_max, nv.max())
        s = if_score(if_clfs.get(ch), ch_fit, feature_cols)
        if len(s) and s.max() > s.min():
            if_min[ch], if_max[ch] = s.min(), s.max()
    if rule_max <= rule_min:
        rule_max = rule_min + 1.0
    return {"rule_min": float(rule_min), "rule_max": float(rule_max),
            "if_min": if_min, "if_max": if_max}


def norm_rule(nv, norm):
    return ((nv - norm["rule_min"]) / (norm["rule_max"] - norm["rule_min"])).clip(0, 1)


def norm_if(s, ch, norm):
    lo, hi = norm["if_min"].get(ch), norm["if_max"].get(ch)
    if lo is None or hi is None or hi <= lo:
        return np.zeros_like(s)
    return ((s - lo) / (hi - lo)).clip(0, 1)


# ----------------------------------------------------------------- 主流程
def main():
    print("=" * 78)
    print("v3 融合策略实验：规则层 + IF 的软/硬/门控/校准融合对比")
    print("=" * 78)

    fit_df, val_df, test_df, feature_cols = load_split()
    channels = sorted(fit_df["channel"].unique())
    print(f"[split] fit={len(fit_df)} val={len(val_df)} test={len(test_df)} ch={channels}")

    thresholds = compute_stat_thresholds(fit_df, feature_cols)
    best_k = select_rule_k(val_df, thresholds, feature_cols)
    best_c = select_if_contamination(val_df, fit_df, feature_cols)
    print(f"[select] best_k={best_k}  per_channel_c={best_c}")

    if_clfs = {}
    for ch in channels:
        c = best_c.get(ch)
        if_clfs[ch] = fit_if_channel(fit_df[fit_df["channel"] == ch], feature_cols, c) if c is not None else None

    norm = fit_normalizers(fit_df, thresholds, feature_cols, if_clfs)

    def scores_for(df):
        # !! 关键：所有数组按下标对齐到 df 自然行序（df.index），绝不能按通道顺序
        #    concatenate 后去和 y_true 比——test_df 非按通道排序时会错位（已踩坑）。
        nv = rule_nv(df, thresholds, feature_cols)            # Series，index=df.index
        rn = norm_rule(nv.to_numpy(), norm)                   # 已对齐 df 顺序
        yn_rule = (nv >= best_k).astype(int).to_numpy()       # 已对齐
        yn_if = np.zeros(len(df), dtype=int)
        ifn = np.zeros(len(df), dtype=float)
        for ch in channels:
            sub = df[df["channel"] == ch]
            idx = sub.index.to_numpy()
            s = if_score(if_clfs.get(ch), sub, feature_cols)
            ifn[idx] = norm_if(s, ch, norm)
            yn_if[idx] = (s > 0).astype(int) if if_clfs.get(ch) is not None else 0
        return rn, ifn, yn_rule, yn_if

    rn_va, ifn_va, yr_va, yi_va = scores_for(val_df)
    rn_te, ifn_te, yr_te, yi_te = scores_for(test_df)
    y_true_val = val_df["anomaly"].to_numpy()
    y_true_test = test_df["anomaly"].to_numpy()

    # ---- 参考四组 ----
    refs = {
        "rule_only":   (yr_te, yr_va),
        "if_only":     (yi_te, yi_va),
        "if_and_rule": (np.logical_and(yr_te, yi_te).astype(int),
                        np.logical_and(yr_va, yi_va).astype(int)),
        "if_or_rule":  (np.logical_or(yr_te, yi_te).astype(int),
                        np.logical_or(yr_va, yi_va).astype(int)),
    }

    # ---- 策略5：soft_global（全局 w + τ，val 选）----
    best_wg, best_tg, best_gf1 = 0.0, 0.5, -1.0
    for w in W_GRID:
        fused_va = w * ifn_va + (1 - w) * rn_va
        for tau in TAU_GRID:
            f1 = quick_f1(y_true_val, (fused_va >= tau).astype(int))
            if f1 > best_gf1:
                best_gf1, best_wg, best_tg = f1, w, tau
    soft_global_test = (best_wg * ifn_te + (1 - best_wg) * rn_te >= best_tg).astype(int)
    soft_global_val = (best_wg * ifn_va + (1 - best_wg) * rn_va >= best_tg).astype(int)
    print(f"[soft_global] 选 w={best_wg} τ={best_tg} (val F1={best_gf1:.4f})")

    # ---- 策略6：soft_perchannel（每通道 w_ch + τ_ch）----
    sp_test, sp_val = np.zeros(len(test_df), int), np.zeros(len(val_df), int)
    sp_ch_params = {}
    for ch in channels:
        m_te = test_df["channel"].values == ch
        m_va = val_df["channel"].values == ch
        rv_te, iv_te = rn_te[m_te], ifn_te[m_te]
        rv_va, iv_va = rn_va[m_va], ifn_va[m_va]
        yv = y_true_val[m_va]
        bw, bt, bf = 0.0, 0.5, -1.0
        for w in W_GRID:
            for tau in TAU_GRID:
                f1 = quick_f1(yv, (w * iv_va + (1 - w) * rv_va >= tau).astype(int))
                if f1 > bf:
                    bf, bw, bt = f1, w, tau
        sp_ch_params[ch] = {"w": float(bw), "tau": float(bt), "val_f1": float(bf)}
        sp_test[m_te] = (bw * iv_te + (1 - bw) * rv_te >= bt).astype(int)
        sp_val[m_va] = (bw * iv_va + (1 - bw) * rv_va >= bt).astype(int)

    # ---- 策略7：gate_perchannel（每通道选 val F1 最高算子）----
    gp_test, gp_val = np.zeros(len(test_df), int), np.zeros(len(val_df), int)
    gp_ch_choice = {}
    for ch in channels:
        m_te = test_df["channel"].values == ch
        m_va = val_df["channel"].values == ch
        yv = y_true_val[m_va]
        cand = {
            "rule": (yr_va[m_va], yr_te[m_te]),
            "if":   (yi_va[m_va], yi_te[m_te]),
            "AND":  (np.logical_and(yr_va[m_va], yi_va[m_va]).astype(int),
                     np.logical_and(yr_te[m_te], yi_te[m_te]).astype(int)),
            "OR":   (np.logical_or(yr_va[m_va], yi_va[m_va]).astype(int),
                     np.logical_or(yr_te[m_te], yi_te[m_te]).astype(int)),
        }
        best_op, best_f1 = "rule", -1.0
        for op, (ypv, _) in cand.items():
            f1 = quick_f1(yv, ypv)
            if f1 > best_f1:
                best_f1, best_op = f1, op
        gp_ch_choice[ch] = {"op": best_op, "val_f1": float(best_f1)}
        gp_val[m_va] = cand[best_op][0]
        gp_test[m_te] = cand[best_op][1]

    # ---- 策略8：soft_calibrated（LR 融合，τ 在 val 内交叉验证选出）----
    # ⚠ 不可在 val 上 fit LR 后又用同一 val 选 τ：训练集自评会让 val F1 虚高，
    #   属于方法论错误（本次因 LR 仅 2 维未暴露，但换高维特征必然虚高）。
    #   改为 5 折分层交叉验证：每折用 4/5 训 LR、在留出 1/5 上预测，
    #   拼出「全程未见过的」val 概率分布再选 τ。test 概率仍由全 val 训的 LR 给出。
    X_val2 = np.column_stack([rn_va, ifn_va])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_true_val), dtype=float)
    for tr_i, te_i in skf.split(X_val2, y_true_val):
        lr_k = LogisticRegression(C=1.0, max_iter=1000).fit(X_val2[tr_i], y_true_val[tr_i])
        oof[te_i] = lr_k.predict_proba(X_val2[te_i])[:, 1]
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(X_val2, y_true_val)
    proba_test = lr.predict_proba(np.column_stack([rn_te, ifn_te]))[:, 1]
    best_tc, best_cf1 = 0.5, -1.0
    for tau in TAU_GRID_CAL:
        f1 = quick_f1(y_true_val, (oof >= tau).astype(int))
        if f1 > best_cf1:
            best_cf1, best_tc = f1, tau
    cal_test = (proba_test >= best_tc).astype(int)
    # 报告用的 val 预测用同一批 OOF 概率，口径与选 τ 时一致，不自评
    cal_val = (oof >= best_tc).astype(int)
    print(f"[soft_calibrated] 选 τ={best_tc} (val F1={best_cf1:.4f}, 5折OOF)  coef={lr.coef_[0].round(3).tolist()}")

    # ---- 最终评估（test 仅此一次，带 bootstrap CI）----
    methods = {
        "rule_only": refs["rule_only"],
        "if_only": refs["if_only"],
        "if_and_rule": refs["if_and_rule"],
        "if_or_rule": refs["if_or_rule"],
        "soft_global": (soft_global_test, soft_global_val),
        "soft_perchannel": (sp_test, sp_val),
        "gate_perchannel": (gp_test, gp_val),
        "soft_calibrated": (cal_test, cal_val),
    }

    results = {}
    print("\n" + "=" * 78)
    print("融合策略对比（策略按 val F1 选出，test 在选定后一次性评估）")
    print("=" * 78)
    for name, (yp_test, yp_val) in methods.items():
        m_test = evaluate(y_true_test, yp_test)
        m_val = evaluate(y_true_val, yp_val)
        results[name] = {"val": m_val, "test": m_test}
        lo, hi = m_test["f1_ci95"]
        print(f"  {name:<16} test F1={m_test['f1']:.4f} CI[{lo:.4f},{hi:.4f}] "
              f"P={m_test['precision']:.3f} R={m_test['recall']:.3f} | val F1={m_val['f1']:.4f}")

    # 铁律：策略选择只看 val，test 在此之前不得进入任何决策。
    # （原实现误用 test["f1"] 选优，等于用 test 选模型 —— 已在验收 B1 中判为 P0）
    best = max(results, key=lambda k: results[k]["val"]["f1"])
    print(f"\n[结论] 按 val F1 选出的最优策略 = {best} "
          f"(val F1={results[best]['val']['f1']:.4f})")
    print(f"[test] 该策略在 test 上一次性评估：F1={results[best]['test']['f1']:.4f} "
          f"CI{results[best]['test']['f1_ci95']}")

    payload_results = {
        "setup": {
            "algorithm": "Isolation Forest + 分段检验（段级）",
            "rule_score": "nv = 每段违规特征数 (0~18)，连续严重度",
            "if_score": "-decision_function (连续异常分，按通道)",
            "normalization": "min/max 在 fit 上算，不泄露 val/test",
            "ref_repro_note": "参考四组应复现 ablation: rule .5882 / if .5612 / AND .6038 / OR .5545",
            "iron_rule": "k,c,w,τ,门控,LR阈值一律只在 val 选；test 只评估一次；CI 仅最终评估给",
        },
        "selection": {
            "best_k": best_k,
            "best_contamination_per_channel": {str(k): v for k, v in best_c.items()},
            "soft_global": {"w": best_wg, "tau": best_tg, "val_f1": best_gf1},
            "soft_perchannel_params": {str(k): v for k, v in sp_ch_params.items()},
            "gate_perchannel_choice": {str(k): v for k, v in gp_ch_choice.items()},
            "soft_calibrated": {"tau": best_tc, "val_f1": best_cf1,
                                 "lr_coef": lr.coef_[0].tolist()},
        },
        "methods": results,
        "best_method_by_val": best,
        "best_val_f1": results[best]["val"]["f1"],
        "best_method_test_f1": results[best]["test"]["f1"],
        "best_method_test_f1_ci95": results[best]["test"]["f1_ci95"],
        "selection_rule": "best_method_by_val 按 val F1 选出；test 仅在该策略确定后评估一次，不参与选择",
    }

    path = save_result(payload_results, "fusion", extra={
        "method": "Fusion strategy comparison (rule + IF)",
        "note": ("8 种融合策略对比；引入规则连续严重度 nv 与 IF 连续异常分 "
                 "-decision_function，覆盖软加权/逐通道门控/逻辑回归校准。"
                 "策略按 val F1 选出，test 在选定后一次性评估，不参与任何选择决策。"),
    })
    print(f"\n[完成] 结果: {os.path.relpath(path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
