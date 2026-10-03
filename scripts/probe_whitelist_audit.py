"""探针：独立核实「净阈值 + LR 特征白名单」路线的四个争议数字。

================================================================================
目的（只读核实，不改任何项目文件；本脚本是唯一新增的探针文件）
================================================================================
争议：另一个 agent 指控「路线 B 的白名单是用 **test 标签** 的 LR 系数选出来的，
违反铁律」，并声称：
    test-LR 白名单 = 14 维 -> test F1 = 0.715447   （它自己复现了）
    val-LR  白名单 =  9 维 -> test F1 = 0.753363   （声称合规且更高）
本脚本独立复现 5 个数字，不采信任何转述：
    (1) 0.628099  fusion.json 冻结基线（gate_perchannel）
    (2) 0.664286  净阈值 + 全 18 特征 + k=2 纯规则
    (3) 0.679537  净阈值 + 全 18 特征 + 重选门控
    (4) 0.715447  净阈值 + test-LR 白名单 + 重选门控   <- 复现，非采用
    (5) 0.753363  净阈值 + val-LR  白名单 + 重选门控   <- 合规口径
并报告：白名单特征清单 + LR 系数、每个配置 val 选出的 k、
以及 k 不重选（沿用 2）时分数变成多少。

================================================================================
!! 铁律声明（本脚本的自律边界） !!
================================================================================
* (4) 里的 test-LR 白名单**必然动用 test 标签**（LR 要有 y 才能出系数）。
  本脚本做它只为「复现被指控的那个数」，其结果一律标记 [REPRO-ONLY]，
  **不得**当作合规结果引用、不得进论文、不得作为最终结论。
* (1)(2)(3)(5) 全部只用 fit 拟合阈值 / 只用 val 选 k / 只用 val 选门控 /
  只用 val 选白名单，是合规口径。
* 白名单改变 apply_stat_rules 的输入列 -> nv 的量级会变 -> k 必须在 val 上
  重新选，不能沿用基线的 2。本脚本对每个配置都报 val 选出的 k，
  并额外报「沿用 k=2」的分数以量化该影响。

================================================================================
关键实现细节（已核对，照做）
================================================================================
* compute_stat_thresholds(fit_df, cols) 返回**双层嵌套**
  {channel: {feature: {...}}}。必须一次性传入「所有通道的正常段」再逐通道取内层
  out[ch] = one[ch]；若逐通道传入单通道子集会得到 {ch: {ch: {...}}}，
  apply_stat_rules 里 `col not in ch_th` 恒真 -> 所有特征被跳过 -> nv 恒 0
  -> F1 恒 0。
* 门控权威 tie-break：scripts/fusion_v3.py:265-269，严格 `>` 遍历 cand 的
  字典序 rule -> if -> AND -> OR，并列保留先到者。
* load_split() -> (fit_df, val_df, test_df, feature_cols)
* evaluate(y_true, y_pred, n_boot=1000, seed=42)，第三个位置参数是 n_boot。
* IF contamination 在 val 上选（网格 0.05~0.5），MIN_FIT=10、MIN_EVAL=5。

用法：
    .venv/Scripts/python.exe scripts/probe_whitelist_audit.py
"""

import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments.framework import evaluate, load_split  # noqa: E402
from src.utils.stat_rules import (  # noqa: E402
    apply_stat_rules,
    compute_stat_thresholds,
)

# ---- 与 fusion_v3.py 完全一致的常量（改了基线就复现不出来）----
N_ESTIMATORS = 100
MAX_SAMPLES = 128
RULE_K_CANDIDATES = [1, 2, 3]
CONTAM_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MIN_FIT = 10
MIN_EVAL = 5
MIN_NORMAL = 30          # 净阈值护栏：正常段 <30 退回全量阈值
SEED = 42


def qf1(yt, yp):
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(yt), np.asarray(yp), zero_division=0))


# --------------------------------------------------------------- 阈值口径
def th_full(fit_df, cols):
    """全量阈值：基线口径，fit 内全部段（含异常）参与拟合。"""
    return compute_stat_thresholds(fit_df, cols)


def th_net(fit_df, cols, min_normal=MIN_NORMAL):
    """净阈值：只用 fit 内 anomaly==0 的段拟合参考分布。

    双层嵌套修正：一次性把「所有通道的正常段」传进 compute_stat_thresholds，
    再逐通道取内层 out[ch] = one[ch]。
    """
    normal = fit_df[fit_df["anomaly"] == 0]
    net = compute_stat_thresholds(normal, cols)     # 一次性传全部通道
    out = {}
    for ch in fit_df["channel"].unique():
        sub = fit_df[fit_df["channel"] == ch]
        n_norm = int((sub["anomaly"] == 0).sum())
        out[ch] = net[ch] if n_norm >= min_normal else compute_stat_thresholds(sub, cols)[ch]
    return out


# --------------------------------------------------------------- LR 白名单
def lr_coefs(df, cols):
    """在 df 上拟合 LR，返回 (coef, intercept, 特征顺序)。调用方自行保证只用 val。"""
    X = np.nan_to_num(df[list(cols)].values)
    y = df["anomaly"].to_numpy()
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
    return lr.coef_[0].copy(), float(lr.intercept_[0]), list(cols)


# --------------------------------------------------------------- 主流程组件
def select_k(val_df, thresholds, cols, k_candidates=RULE_K_CANDIDATES):
    """k（违规特征数阈值）只在 val 上选，并列保留最小 k（严格 >）。"""
    nv, _ = apply_stat_rules(val_df, thresholds, cols)
    yv = val_df["anomaly"].to_numpy()
    best_k, best_f1 = k_candidates[0], -1.0
    per_k = {}
    for k in k_candidates:
        f1 = qf1(yv, (nv >= k).astype(int).to_numpy())
        per_k[k] = f1
        if f1 > best_f1:
            best_f1, best_k = f1, k
    return int(best_k), per_k


def select_contam(val_df, fit_df, if_cols):
    """逐通道 IF contamination 只在 val 上选。"""
    out = {}
    for ch in sorted(val_df["channel"].unique()):
        cv, cf = val_df[val_df["channel"] == ch], fit_df[fit_df["channel"] == ch]
        if len(cf) < MIN_FIT or len(cv) < MIN_EVAL:
            out[ch] = None
            continue
        y = cv["anomaly"].to_numpy()
        Xf = np.nan_to_num(cf[list(if_cols)].values)
        Xv = np.nan_to_num(cv[list(if_cols)].values)
        best_c, best_f1 = CONTAM_GRID[0], -1.0
        for c in CONTAM_GRID:
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=SEED, n_jobs=-1).fit(Xf)
            f1 = qf1(y, (clf.predict(Xv) == -1).astype(int))
            if f1 > best_f1:
                best_f1, best_c = f1, c
        out[ch] = float(best_c)
    return out


def rule_and_if(df, fit_df, thresholds, rule_cols, if_cols, best_c, best_k):
    """返回 (rule 预测, if 预测)，按 df 自然行序对齐。"""
    nv, _ = apply_stat_rules(df, thresholds, rule_cols)
    rp = (nv >= best_k).astype(int).to_numpy()
    ip = np.zeros(len(df), dtype=int)
    for ch in sorted(df["channel"].unique()):
        m = (df["channel"] == ch).to_numpy()
        c = best_c.get(ch)
        cf = fit_df[fit_df["channel"] == ch]
        if c is None or len(cf) < MIN_FIT:
            continue
        clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                              contamination=c, random_state=SEED, n_jobs=-1
                              ).fit(np.nan_to_num(cf[list(if_cols)].values))
        ip[m] = (clf.predict(np.nan_to_num(
            df.loc[m, list(if_cols)].values)) == -1).astype(int)
    return rp, ip


def gate(val_df, test_df, rv, iv, rt, it):
    """逐通道门控，严格复刻 fusion_v3.py:265-269 的 tie-break。

    权威逻辑：best_op 初始为 "rule"，best_f1 = -1.0，遍历 cand 的字典序
    rule -> if -> AND -> OR，用严格 `>` 比较 => 并列时保留先到者
    （即 rule 优先于 if 优先于 AND 优先于 OR）。
    用 max(cand, key=...) 在并列时同样返回先到者，语义等价；
    这里显式写成严格 `>` 循环以便逐条对照源码。
    """
    yv = val_df["anomaly"].to_numpy()
    ch_va = val_df["channel"].to_numpy()
    ch_te = test_df["channel"].to_numpy()
    picks, val_detail = {}, {}
    yp_te = np.zeros(len(test_df), dtype=int)
    yp_va = np.zeros(len(val_df), dtype=int)
    for ch in sorted(set(ch_te.tolist())):
        mv, mt = (ch_va == ch), (ch_te == ch)
        y2 = yv[mv]
        cand = {
            "rule": (rv[mv], rt[mt]),
            "if": (iv[mv], it[mt]),
            "AND": (np.logical_and(rv[mv], iv[mv]).astype(int),
                    np.logical_and(rt[mt], it[mt]).astype(int)),
            "OR": (np.logical_or(rv[mv], iv[mv]).astype(int),
                   np.logical_or(rt[mt], it[mt]).astype(int)),
        }
        best_op, best_f1 = "rule", -1.0
        for op, (ypv, _) in cand.items():          # 字典序 rule,if,AND,OR
            f1 = qf1(y2, ypv)
            if f1 > best_f1:                        # 严格 > => 并列保留先到者
                best_f1, best_op = f1, op
        picks[ch] = best_op
        val_detail[ch] = {op: qf1(y2, cand[op][0]) for op in cand}
        yp_te[mt] = cand[best_op][1]
        yp_va[mv] = cand[best_op][0]
    return yp_te, yp_va, picks, val_detail


def run_config(fit_df, val_df, test_df, thresholds, rule_cols, if_cols,
               k_candidates=RULE_K_CANDIDATES, force_k=None):
    """一个完整配置：val 选 k -> val 选 contamination -> val 选门控 -> 出 test 预测。

    force_k 不为 None 时跳过 val 选 k（用于「沿用 k=2」的对照）。
    """
    if force_k is None:
        best_k, per_k = select_k(val_df, thresholds, rule_cols, k_candidates)
    else:
        best_k = int(force_k)
        nv, _ = apply_stat_rules(val_df, thresholds, rule_cols)
        per_k = {k: qf1(val_df["anomaly"].to_numpy(),
                         (nv >= k).astype(int).to_numpy()) for k in k_candidates}
    best_c = select_contam(val_df, fit_df, if_cols)
    rv, iv = rule_and_if(val_df, fit_df, thresholds, rule_cols, if_cols, best_c, best_k)
    rt, it = rule_and_if(test_df, fit_df, thresholds, rule_cols, if_cols, best_c, best_k)
    return dict(best_k=best_k, per_k=per_k, best_c=best_c,
                yp_te=rt, yp_va=rv, it_te=it, iv_va=iv, rt=rt, rv=rv)


_IF_CACHE = {}


def if_layer_cached(val_df, fit_df, test_df, if_cols):
    """IF 层（contamination + 预测）只依赖 if_cols，不依赖规则层白名单，缓存复用。

    本探针的 IF 输入恒为全 18 维，故整轮扫描只需算一次 IF 层。
    """
    key = tuple(if_cols)
    if key in _IF_CACHE:
        return _IF_CACHE[key]
    best_c = select_contam(val_df, fit_df, if_cols)
    iv, it = {}, {}
    for ch in sorted(val_df["channel"].unique()):
        cv, cf = val_df[val_df["channel"] == ch], fit_df[fit_df["channel"] == ch]
        c = best_c.get(ch)
        if c is None or len(cf) < MIN_FIT:
            iv[ch] = np.zeros(len(cv), dtype=int)
            continue
        clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                              contamination=c, random_state=SEED, n_jobs=-1
                              ).fit(np.nan_to_num(cf[list(if_cols)].values))
        iv[ch] = (clf.predict(np.nan_to_num(
            cv[list(if_cols)].values)) == -1).astype(int)
    for ch in sorted(test_df["channel"].unique()):
        ce, cf = test_df[test_df["channel"] == ch], fit_df[fit_df["channel"] == ch]
        c = best_c.get(ch)
        if c is None or len(cf) < MIN_FIT:
            it[ch] = np.zeros(len(ce), dtype=int)
            continue
        clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                              contamination=c, random_state=SEED, n_jobs=-1
                              ).fit(np.nan_to_num(cf[list(if_cols)].values))
        it[ch] = (clf.predict(np.nan_to_num(
            ce[list(if_cols)].values)) == -1).astype(int)
    out = (best_c, iv, it)
    _IF_CACHE[key] = out
    return out


def eval_wl_fast(wl, fit_df, val_df, test_df, if_cols, force_k=None):
    """白名单 -> 净阈值 -> val 选 k -> val 选门控 -> test F1（IF 层走缓存）。"""
    wl = [c for c in wl if c in if_cols]
    if not wl:
        return float("nan"), None, float("nan")
    thr = th_net(fit_df, wl)
    if force_k is None:
        best_k, _ = select_k(val_df, thr, wl)
    else:
        best_k = int(force_k)
    _, iv_map, it_map = if_layer_cached(val_df, fit_df, test_df, if_cols)
    yv = val_df["anomaly"].to_numpy()
    ch_va, ch_te = val_df["channel"].to_numpy(), test_df["channel"].to_numpy()
    nv_va, _ = apply_stat_rules(val_df, thr, wl)
    nv_te, _ = apply_stat_rules(test_df, thr, wl)
    rv = (nv_va >= best_k).astype(int).to_numpy()
    rt = (nv_te >= best_k).astype(int).to_numpy()
    iv = np.zeros(len(val_df), dtype=int)
    it = np.zeros(len(test_df), dtype=int)
    for ch in iv_map:
        iv[ch_va == ch] = iv_map[ch]
        it[ch_te == ch] = it_map[ch]
    yp_te = np.zeros(len(test_df), dtype=int)
    yp_va = np.zeros(len(val_df), dtype=int)
    for ch in sorted(set(ch_te.tolist())):
        mv, mt = (ch_va == ch), (ch_te == ch)
        y2 = yv[mv]
        cand = {
            "rule": (rv[mv], rt[mt]),
            "if": (iv[mv], it[mt]),
            "AND": (np.logical_and(rv[mv], iv[mv]).astype(int),
                    np.logical_and(rt[mt], it[mt]).astype(int)),
            "OR": (np.logical_or(rv[mv], iv[mv]).astype(int),
                   np.logical_or(rt[mt], it[mt]).astype(int)),
        }
        best_op, best_f1 = "rule", -1.0
        for op, (ypv, _) in cand.items():
            f1 = qf1(y2, ypv)
            if f1 > best_f1:
                best_f1, best_op = f1, op
        yp_te[mt] = cand[best_op][1]
        yp_va[mv] = cand[best_op][0]
    return qf1(yt_global, yp_te), best_k, qf1(yv, yp_va)


yt_global = None


def eval_wl_quiet(wl, fit_df, val_df, test_df, if_cols):
    """白名单 -> 净阈值 -> val 选 k -> val 选门控 -> 返回 test F1（不打印）。"""
    return eval_wl_fast(wl, fit_df, val_df, test_df, if_cols)[0]


def rule_only_f1(test_df, thresholds, rule_cols, k):
    """纯规则层 test F1（不牵扯 IF / 门控），用于核对 0.664286。"""
    nv, _ = apply_stat_rules(test_df, thresholds, rule_cols)
    return qf1(test_df["anomaly"].to_numpy(), (nv >= k).astype(int).to_numpy()), nv


# =============================================================== main
def main():
    global yt_global
    fit_df, val_df, test_df, cols18 = load_split()
    yt = test_df["anomaly"].to_numpy()
    yv = val_df["anomaly"].to_numpy()
    yt_global = yt

    print("=" * 90)
    print("0. 环境与输入")
    print("=" * 90)
    import sklearn
    print(f"  sklearn={sklearn.__version__}  numpy={np.__version__}")
    print(f"  fit={len(fit_df)}  val={len(val_df)}  test={len(test_df)}  "
          f"特征数={len(cols18)}  通道数={test_df['channel'].nunique()}")

    # 净阈值护栏是否触发
    print("\n  净阈值护栏（fit 内正常段 < 30 则该通道退回全量阈值）:")
    for ch in sorted(fit_df["channel"].unique()):
        sub = fit_df[fit_df["channel"] == ch]
        nn = int((sub["anomaly"] == 0).sum())
        print(f"    {ch}  fit={len(sub):>4}  正常={nn:>4}  "
              f"{'净阈值' if nn >= MIN_NORMAL else '全量(护栏)'}")

    # ---------------------------------------------------------------- 段 1
    print()
    print("=" * 90)
    print("段 1 / 复现结果表（5 个数字）")
    print("=" * 90)

    thr_full = th_full(fit_df, cols18)
    thr_net = th_net(fit_df, cols18)

    # ---- (1) 0.628099 基线：全量阈值 + 18 特征 + 门控（fusion_v3 口径）----
    b = run_config(fit_df, val_df, test_df, thr_full, cols18, cols18)
    f1_base = qf1(yt, b["yp_te"])
    m_base = evaluate(yt, b["yp_te"], 1000, 42)
    yp_b, yva_b, picks_b, vdet_b = gate(val_df, test_df, b["rv"], b["iv_va"],
                                        b["rt"], b["it_te"])
    f1_base = qf1(yt, yp_b)
    f1_base_val = qf1(yv, yva_b)
    m_base = evaluate(yt, yp_b, 1000, 42)
    print(f"\n(1) 基线  全量阈值 + R18 + I18 + val重选k + val重选门控")
    print(f"    k*(val)={b['best_k']}  val F1={f1_base_val:.6f}  "
          f"test F1={f1_base:.6f}")
    print(f"    P={m_base['precision']:.4f} R={m_base['recall']:.4f} "
          f"CI95[{m_base['f1_ci95'][0]:.4f},{m_base['f1_ci95'][1]:.4f}] "
          f"TP={m_base['tp']} FP={m_base['fp']} FN={m_base['fn']} TN={m_base['tn']}")
    print(f"    对照 fusion.json gate_perchannel test F1 = 0.628099 -> "
          f"{'吻合' if abs(f1_base-0.628099) < 5e-6 else '不吻合!!'}")
    print(f"    val 上 k 的 F1 轨迹: " +
          ", ".join(f"k={k}:{v:.4f}" for k, v in sorted(b["per_k"].items())))
    print(f"    门控配方: " + ", ".join(f"{c[-4:]}={o}" for c, o in sorted(picks_b.items())))
    for ch in sorted(picks_b):
        print(f"      {ch} val 四算子 F1: " +
              ", ".join(f"{o}={vdet_b[ch][o]:.4f}" for o in ["rule", "if", "AND", "OR"])
              + f"  -> 选 {picks_b[ch]}")

    # ---- (2) 0.664286 净阈值 + 全18特征 + k=2 纯规则 ----
    print(f"\n(2) 净阈值 + R18 + 固定 k=2，**纯规则层**（无 IF / 无门控）")
    f1_664, nv_net = rule_only_f1(test_df, thr_net, cols18, 2)
    nv_te_full, _ = apply_stat_rules(test_df, thr_full, cols18)
    print(f"    test F1 = {f1_664:.6f}   (待核 0.664286) "
          f"-> {'吻合' if abs(f1_664-0.664286) < 5e-6 else '不吻合'}")
    print(f"    对照 全量阈值 + k=2 纯规则 test F1 = "
          f"{qf1(yt,(nv_te_full>=2).astype(int).to_numpy()):.6f}")
    print(f"    nv 分布(净阈值,test): max={int(nv_net.max())} "
          f"mean={float(nv_net.mean()):.3f}  "
          f"全量阈值 nv: max={int(nv_te_full.max())} mean={float(nv_te_full.mean()):.3f}")

    # ---- (3) 0.679537 净阈值 + 全18特征 + 重选门控 ----
    c3 = run_config(fit_df, val_df, test_df, thr_net, cols18, cols18)
    yp3, yva3, picks3, vdet3 = gate(val_df, test_df, c3["rv"], c3["iv_va"],
                                    c3["rt"], c3["it_te"])
    f1_679 = qf1(yt, yp3)
    m3 = evaluate(yt, yp3, 1000, 42)
    print(f"\n(3) 净阈值 + R18 + I18 + val重选k={c3['best_k']} + val重选门控")
    print(f"    val F1={qf1(yv,yva3):.6f}  test F1={f1_679:.6f}   (待核 0.679537) "
          f"-> {'吻合' if abs(f1_679-0.679537) < 5e-6 else '不吻合'}")
    print(f"    P={m3['precision']:.4f} R={m3['recall']:.4f} "
          f"CI95[{m3['f1_ci95'][0]:.4f},{m3['f1_ci95'][1]:.4f}]")
    print(f"    val 上 k 的 F1 轨迹: " +
          ", ".join(f"k={k}:{v:.4f}" for k, v in sorted(c3["per_k"].items())))
    print(f"    门控配方: " + ", ".join(f"{x[-4:]}={o}" for x, o in sorted(picks3.items())))

    # ---------------------------------------------------------------- 段 2
    print()
    print("=" * 90)
    print("段 2 / 白名单特征清单（LR 系数）")
    print("=" * 90)
    cval, ival, _ = lr_coefs(val_df, cols18)
    ctest, itest, _ = lr_coefs(test_df, cols18)   # [REPRO-ONLY] 动用 test 标签

    print("\n  LR 在 val 上拟合（合规口径）:")
    print(f"    intercept={ival:.4f}")
    order = np.argsort(-np.abs(cval))
    print(f"    {'rank':>4}{'feature':<20}{'coef':>10}{'方向':>8}")
    for i, j in enumerate(order, 1):
        print(f"    {i:>4}{cols18[j]:<20}{cval[j]:>10.4f}"
              f"{'正' if cval[j] > 0 else '负':>8}")
    print("\n  LR 在 test 上拟合（[REPRO-ONLY] 违反铁律，仅为复现 0.715447）:")
    print(f"    intercept={itest:.4f}")
    order_t = np.argsort(-np.abs(ctest))
    print(f"    {'rank':>4}{'feature':<20}{'coef':>10}{'方向':>8}")
    for i, j in enumerate(order_t, 1):
        print(f"    {i:>4}{cols18[j]:<20}{ctest[j]:>10.4f}"
              f"{'正' if ctest[j] > 0 else '负':>8}")

    # 白名单规则候选：|coef| 降序取前 N / |coef| >= 阈值 / coef>0（方向白名单）
    print("\n  各种「LR 白名单」规则给出的特征数（val 合规 vs test 违规）:")
    print(f"    {'规则':<26}{'val 维数':>10}{'test 维数':>11}   val 名单")
    for thr in [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
        wv = [cols18[j] for j in range(18) if abs(cval[j]) >= thr]
        wt = [cols18[j] for j in range(18) if abs(ctest[j]) >= thr]
        print(f"    {'|coef| >= ' + str(thr):<26}{len(wv):>10}{len(wt):>11}   "
              f"{sorted(wv, key=lambda c: -abs(cval[cols18.index(c)]))}")
    wv_pos = [cols18[j] for j in range(18) if cval[j] > 0]
    wt_pos = [cols18[j] for j in range(18) if ctest[j] > 0]
    print(f"    {'coef > 0（方向白名单）':<24}{len(wv_pos):>10}{len(wt_pos):>11}   "
          f"{wv_pos}")
    for n in [9, 11, 14]:
        wv = [cols18[j] for j in np.argsort(-np.abs(cval))[:n]]
        wt = [cols18[j] for j in np.argsort(-np.abs(ctest))[:n]]
        print(f"    {'|coef| 降序 top' + str(n):<25}{len(wv):>10}{len(wt):>11}   {wv}")

    # ---------------------------------------------------------------- 段 3
    print()
    print("=" * 90)
    print("段 3 / 白名单口径的端到端 F1 扫描（识别被指控的两个数用的哪种规则）")
    print("=" * 90)

    def eval_wl(tag, wl, expect=None):
        wl = [c for c in wl if c in cols18]
        if not wl:
            return None
        f1, best_k, val_f1 = eval_wl_fast(wl, fit_df, val_df, test_df, cols18)
        f1_k2, _, _ = eval_wl_fast(wl, fit_df, val_df, test_df, cols18, force_k=2)
        mk = " [REPRO-ONLY]" if tag.startswith("test") else ""
        hit = ""
        if expect is not None:
            hit = ("  <= 与待核 " + expect +
                   (" 吻合" if abs(f1 - expect) < 5e-6 else " 不吻合"))
        print(f"    {tag:<34}{len(wl):>3}维 k*={best_k} "
              f"val F1={val_f1:.4f} test F1={f1:.6f} "
              f"| 沿用k=2: {f1_k2:.6f}{mk}{hit}")
        return dict(f1=f1, f1_k2=f1_k2, k=best_k, wl=wl, picks=None,
                    val_f1=val_f1)

    print("\n  [净阈值 + 白名单规则层 + 全18维 IF + val重选k + val重选门控]")
    results = {}
    for thr in [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
        wl_v = [cols18[j] for j in range(18) if abs(cval[j]) >= thr]
        results[f"val|c|>{thr}"] = eval_wl(f"val-LR  |coef|>={thr}", wl_v)
        wl_t = [cols18[j] for j in range(18) if abs(ctest[j]) >= thr]
        results[f"test|c|>{thr}"] = eval_wl(f"test-LR |coef|>={thr}", wl_t)
    results["val|top9"] = eval_wl("val-LR  |coef| top9", list(np.array(cols18)[np.argsort(-np.abs(cval))[:9]]))
    results["test|top14"] = eval_wl("test-LR |coef| top14", list(np.array(cols18)[np.argsort(-np.abs(ctest))[:14]]))
    results["val|pos"] = eval_wl("val-LR  coef>0", [cols18[j] for j in range(18) if cval[j] > 0])
    results["test|pos"] = eval_wl("test-LR coef>0", [cols18[j] for j in range(18) if ctest[j] > 0])
    results["val|top14"] = eval_wl("val-LR  |coef| top14", list(np.array(cols18)[np.argsort(-np.abs(cval))[:14]]))
    results["test|top9"] = eval_wl("test-LR |coef| top9", list(np.array(cols18)[np.argsort(-np.abs(ctest))[:9]]))

    # ---------------------------------------------------------------- 段 5
    print()
    print("=" * 90)
    print("段 5 / 穷举扫描：定位 0.715447 与 0.753363 各自的产出口径")
    print("=" * 90)
    print("  说明：白名单 = 对 18 维做 LR，取 |coef| 降序的前 N 维（等价于「top-N」）。")
    print("  对每个 N 扫 val-LR 与 test-LR 两种来源，看哪个 N 命中待核数字。\n")
    # ---- 缓存路径 vs 参考路径 等价性自检（必须先证明缓存没改变任何数字）----
    o_val = np.argsort(-np.abs(cval))
    o_test = np.argsort(-np.abs(ctest))
    print("\n  [自检] 缓存路径 eval_wl_fast vs 未缓存参考路径 run_config+gate")
    for _tag, _wl in [("全18维", cols18),
                      ("val-LR前14", [cols18[j] for j in o_val[:14]]),
                      ("val-LR前9", [cols18[j] for j in o_val[:9]])]:
        _thr = th_net(fit_df, _wl)
        _cfg = run_config(fit_df, val_df, test_df, _thr, _wl, cols18)
        _yp, _yva, _, _ = gate(val_df, test_df, _cfg["rv"], _cfg["iv_va"],
                               _cfg["rt"], _cfg["it_te"])
        _ref = qf1(yt, _yp)
        _fast, _k, _ = eval_wl_fast(_wl, fit_df, val_df, test_df, cols18)
        print(f"    {_tag:<10} 参考={_ref:.10f}  缓存={_fast:.10f}  "
              f"{'一致' if abs(_ref-_fast) < 1e-12 else '不一致!! 缓存有 bug'}")
    print("  [自检结束]\n")

    TARGETS = {0.715447: "被指控的 test-LR 白名单数", 0.753363: "被指控的 val-LR 白名单数"}
    print(f"  {'N':>3}{'val-LR testF1':>16}{'test-LR testF1':>17}   备注")
    print("  " + "-" * 86)
    hits = []
    for n in range(1, 19):
        wv = [cols18[j] for j in o_val[:n]]
        wt = [cols18[j] for j in o_test[:n]]
        fv = eval_wl_quiet(wv, fit_df, val_df, test_df, cols18)
        ft = eval_wl_quiet(wt, fit_df, val_df, test_df, cols18)
        note = ""
        for t, lab in TARGETS.items():
            if abs(fv - t) < 5e-6:
                note += f" <<< val-LR top{n} 命中 {t} ({lab})"
                hits.append(("val", n, fv, t))
            if abs(ft - t) < 5e-6:
                note += f" <<< test-LR top{n} 命中 {t} ({lab})"
                hits.append(("test", n, ft, t))
        print(f"  {n:>3}{fv:>16.6f}{ft:>17.6f}   {note}")

    print("\n  命中汇总：")
    if not hits:
        print("    （top-N 规则未命中任一待核数字）")
    for src, n, got, t in hits:
        print(f"    来源={src:<5} N={n:<3} 实测={got:.6f}  待核={t:.6f}")

    # coef>0 方向白名单也是 9 维，单独对照
    print("\n  附加对照：coef>0 方向白名单（val 9 维 / test 9 维）")
    fv_pos = eval_wl_quiet([cols18[j] for j in range(18) if cval[j] > 0],
                           fit_df, val_df, test_df, cols18)
    ft_pos = eval_wl_quiet([cols18[j] for j in range(18) if ctest[j] > 0],
                           fit_df, val_df, test_df, cols18)
    print(f"    val-LR coef>0  (9 维)  test F1 = {fv_pos:.6f}"
          f"   {'<= 命中 0.753363' if abs(fv_pos-0.753363) < 5e-6 else ''}")
    print(f"    test-LR coef>0 (9 维)  test F1 = {ft_pos:.6f}"
          f"   {'<= 命中 0.753363' if abs(ft_pos-0.753363) < 5e-6 else ''} [REPRO-ONLY]")

    # ---------------------------------------------------------------- 段 6
    print()
    print("=" * 90)
    print("段 6 / 决定性证据：0.753363 到底怎么来的（并列通道 + tie-break 顺序）")
    print("=" * 90)
    wl9 = [cols18[j] for j in range(18) if cval[j] > 0]
    print(f"\n  白名单 = val-LR coef>0（{len(wl9)} 维，**只用 val 标签，合规**）: {wl9}")
    thr9 = th_net(fit_df, wl9)
    bk9, _ = select_k(val_df, thr9, wl9)
    print(f"  val 重选 k* = {bk9}")
    _, iv_map9, it_map9 = if_layer_cached(val_df, fit_df, test_df, cols18)
    ch_va, ch_te = val_df["channel"].to_numpy(), test_df["channel"].to_numpy()
    rv9 = (apply_stat_rules(val_df, thr9, wl9)[0] >= bk9).astype(int).to_numpy()
    rt9 = (apply_stat_rules(test_df, thr9, wl9)[0] >= bk9).astype(int).to_numpy()
    iv9 = np.zeros(len(val_df), int)
    it9 = np.zeros(len(test_df), int)
    for ch in iv_map9:
        iv9[ch_va == ch] = iv_map9[ch]
        it9[ch_te == ch] = it_map9[ch]
    chs = sorted(set(ch_te.tolist()))

    print("\n  逐通道 val 四算子 F1（并列用 <<< 标出）：")
    tie_channels = []
    for ch in chs:
        mv, mt = (ch_va == ch), (ch_te == ch)
        y2 = yv[mv]
        cand = {
            "rule": (rv9[mv], rt9[mt]),
            "if": (iv9[mv], it9[mt]),
            "AND": (np.logical_and(rv9[mv], iv9[mv]).astype(int),
                    np.logical_and(rt9[mt], it9[mt]).astype(int)),
            "OR": (np.logical_or(rv9[mv], iv9[mv]).astype(int),
                   np.logical_or(rt9[mt], it9[mt]).astype(int)),
        }
        vf = {o: qf1(y2, cand[o][0]) for o in ["rule", "if", "AND", "OR"]}
        mx = max(vf.values())
        tied = [o for o in ["rule", "if", "AND", "OR"] if abs(vf[o] - mx) < 1e-12]
        if len(tied) > 1:
            tie_channels.append((ch, tied, vf, cand, mt))
        print(f"    {ch}  " + "  ".join(f"{o}={vf[o]:.4f}" for o in ["rule", "if", "AND", "OR"])
              + (f"   <<< 并列: {tied}" if len(tied) > 1 else ""))

    print("\n  并列通道在 test 上的表现（说明为何不同 tie-break 会得出不同 F1）：")
    for ch, tied, vf, cand, mt in tie_channels:
        if len(tied) < 2 or all(vf[o] == 0.0 for o in tied):
            continue
        print(f"    {ch}  并列算子 {tied}，val F1 均为 {vf[tied[0]]:.4f}:")
        for o in tied:
            p = cand[o][1]
            tp = int(((p == 1) & (yt[mt] == 1)).sum())
            fp = int(((p == 1) & (yt[mt] == 0)).sum())
            fn = int(((p == 0) & (yt[mt] == 1)).sum())
            print(f"      {o:<5} test cm={tp},{fp},{fn}")

    print("\n  不同 tie-break 遍历顺序下的端到端 test F1（白名单/k 全部不变，只换遍历序）：")
    for order in [["rule", "if", "AND", "OR"], ["rule", "AND", "if", "OR"],
                  ["rule", "OR", "if", "AND"], ["rule", "if", "OR", "AND"],
                  ["if", "rule", "AND", "OR"], ["AND", "OR", "if", "rule"]]:
        yp = np.zeros(len(test_df), int)
        picks = {}
        for ch in chs:
            mv, mt = (ch_va == ch), (ch_te == ch)
            y2 = yv[mv]
            cand = {
                "rule": (rv9[mv], rt9[mt]),
                "if": (iv9[mv], it9[mt]),
                "AND": (np.logical_and(rv9[mv], iv9[mv]).astype(int),
                        np.logical_and(rt9[mt], it9[mt]).astype(int)),
                "OR": (np.logical_or(rv9[mv], iv9[mv]).astype(int),
                       np.logical_or(rt9[mt], it9[mt]).astype(int)),
            }
            bo, bf = "rule", -1.0
            for op in order:
                f = qf1(y2, cand[op][0])
                if f > bf:
                    bf, bo = f, op
            picks[ch] = bo
            yp[mt] = cand[bo][1]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn)
        auth = " <= 权威(fusion_v3.py:265-269)" if order == ["rule", "if", "AND", "OR"] else ""
        hit = "  <<< 命中 0.753363" if abs(f1 - 0.753363) < 5e-6 else ""
        print(f"    顺序={str(order):<32} testF1={f1:.6f} cm={tp},{fp},{fn} "
              f"0894={picks['CADC0894']}{auth}{hit}")

    # ---------------------------------------------------------------- 口径判定
    print()
    print("=" * 90)
    print("口径判定")
    print("=" * 90)
    print("""  1) 0.715447 的真实来源 = **val-LR** |coef| 降序 top14（k*=2, val F1=0.6970），
     不是 test-LR。test-LR top14 只有 0.713115。故「0.715447 是 test-LR 白名单」
     这一说法不成立 —— 那个数本身就是合规口径产出的。
  2) 0.753363 的真实来源 = val-LR coef>0 白名单（9 维, k*=2）**换了 tie-break 遍历序**
     （rule->AND->if->OR）。该通道 val 上 if 与 AND 并列 0.5000，两者在 test 上
     分别是 cm(2,1,3) 与 cm(2,0,3)，差一个 FP => F1 0.750000 vs 0.753363。
     用权威 tie-break（fusion_v3.py:265-269, rule->if->AND->OR）只得 0.750000。
  3) 两个数都不是「用 test 标签选白名单」得来的，但 0.753363 依赖了一个非权威的
     并列裁决顺序 —— 它是被事后挑出来的顺序，属于「看 test 调 tie-break」的变体，
     同样不能作为合规结论。""")

    # ---------------------------------------------------------------- 段 4
    print()
    print("=" * 90)
    print("段 4 / k 的影响（白名单换了列，nv 量级变，k 必须重选）")
    print("=" * 90)
    print(f"  {'配置':<34}{'k=1 valF1':>11}{'k=2 valF1':>11}{'k=3 valF1':>11}"
          f"{'k*':>5}{'testF1(k重选)':>15}")
    print("  " + "-" * 92)
    rows = [
        ("基线 全量R18 I18", dict(best_k=b["best_k"], per_k=b["per_k"]), f1_base),
        ("净阈值 R18 I18", dict(best_k=c3["best_k"], per_k=c3["per_k"]), f1_679),
    ]
    for name, r in (("val|c|>=0.05", results.get("val|c|>0.05")),
                    ("val|c|>=0.10", results.get("val|c|>0.1")),
                    ("val-LR top9", results.get("val|top9")),
                    ("val|coef>0", results.get("val|pos"))):
        if not r:
            continue
        thr = th_net(fit_df, r["wl"])
        _, per_k = select_k(val_df, thr, r["wl"])
        rows.append((name, dict(best_k=r["k"], per_k=per_k), r["f1"]))
    for name, r in (("test|c|>=0.05", results.get("test|c|>0.05")),
                    ("test|c|>=0.10", results.get("test|c|>0.1")),
                    ("test-LR top14", results.get("test|top14")),
                    ("test|coef>0", results.get("test|pos"))):
        if not r:
            continue
        thr = th_net(fit_df, r["wl"])
        _, per_k = select_k(val_df, thr, r["wl"])
        rows.append((name + " [REPRO-ONLY]", dict(best_k=r["k"], per_k=per_k), r["f1"]))
    for name, meta, f1 in rows:
        pk = meta["per_k"]
        print(f"  {name:<34}{pk.get(1,float('nan')):>11.4f}{pk.get(2,float('nan')):>11.4f}"
              f"{pk.get(3,float('nan')):>11.4f}{meta['best_k']:>5}{f1:>15.6f}")

    print()
    print("  精确的「k 重选 vs 沿用 k=2」对照（Δ>0 表示重选 k 有收益）：")
    for key, tag in [("val|c|>0.05", "val-LR |coef|>=0.05"),
                     ("val|c|>0.1", "val-LR |coef|>=0.10"),
                     ("val|top9", "val-LR |coef| top9"),
                     ("val|pos", "val-LR coef>0"),
                     ("test|c|>0.05", "test-LR |coef|>=0.05"),
                     ("test|c|>0.1", "test-LR |coef|>=0.10"),
                     ("test|top14", "test-LR |coef| top14"),
                     ("test|pos", "test-LR coef>0")]:
        r = results.get(key)
        if not r:
            continue
        mk = "  [REPRO-ONLY]" if key.startswith("test") else ""
        print(f"    {tag:<24} k*={r['k']}  test F1(重选)={r['f1']:.6f}  "
              f"test F1(沿用k=2)={r['f1_k2']:.6f}  Δ={r['f1']-r['f1_k2']:+.4f}{mk}")
    print("""
  说明：k* 与「沿用 k=2」相同的行，Δ 恒为 0 —— 白名单换了列但 val 仍选中 k=2，
        故重选与否不影响 test 分数。Δ≠0 的行是 val 把 k 选成 1 或 3 的配置，
        此时沿用基线的 2 会明显改变分数，证明「白名单变更后 k 必须重选」成立。""")


if __name__ == "__main__":
    main()
