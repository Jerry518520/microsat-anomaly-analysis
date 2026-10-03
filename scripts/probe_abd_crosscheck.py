"""
A/B/D 档「穷举 tie-break 敏感性」独立交叉验证探针
================================================================================

目的：用**完全独立的第二实现**复现 B 档（净阈值 + 18 特征）与 D 档（净阈值 +
11 特征白名单）的穷举 tie-break 区间 / 极差 / 中位数 / 点估计，用于交叉验证现有结论。

纪律（本脚本自身必须遵守）：
  * 本脚本**不 import、不 copy、不运行** scripts/probe_abd_median.py /
    probe_tiebreak_full.py / probe_tiebreak_enumerate.py，全部逻辑自写。
  * 只读：只 import 权威划分 load_split 与权威规则函数 compute_stat_thresholds /
    apply_stat_rules，不改src/ 任何文件，不写 data/results/ 任何文件。
  * 不重跑 fusion.json。

复现口径（与任务书一致）：
  * 通道 9 个；MIN_FIT=10, MIN_EVAL=5, MIN_NORMAL=30
  * IsolationForest: n_estimators=100, max_samples=128, random_state=42, n_jobs=-1
  * contamination 候选: [0.05..0.50] step 0.05
  * 算子语义: AND = rule & if, OR = rule | if   （OR 必须是 |）
  * 算子遍历顺序: ["rule", "if", "AND", "OR"]
  * k∈ {1,2,3,4} 在 val 上选，严格 > 取最大（并列保留先到者）
  * 逐通道算子：val F1 最大者入选，并列集合= 所有达到最大值的算子
  * 穷举：只对「并列集合非单元素」的通道做笛卡尔积（单元素通道固定），
          组合数 = 各通道并列集合大小之积
  * 权威点估计：按算子字典序遍历全部组合 + 严格 > 保留先到者 => 并列时取字典序最小
  * 净阈值：某通道 fit 内正常段数 < MIN_NORMAL 时，该通道退回用全量 fit_df 的阈值
  * D 档白名单：|roc_auc(val_y, feat) - 0.5| >= 0.10，按 feature_cols 原始列序输出；
          白名单只作用于**规则层阈值**，IsolationForest 仍用全部 18 特征

用法：
    /d/Python313/python.exe scripts/probe_abd_crosscheck.py
"""

import itertools
import os
import sys
from fractions import Fraction

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score as _f1
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments.framework import load_split  # noqa: E402
from src.utils.stat_rules import (  # noqa: E402
    apply_stat_rules,
    compute_stat_thresholds,
)

# ----------------------------------------------------------------- 固定超参
MIN_FIT = 10
MIN_EVAL = 5
MIN_NORMAL = 30
N_ESTIMATORS = 100
MAX_SAMPLES = 128
IF_SEED = 42
CONTAM_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
K_CANDIDATES = [1, 2, 3, 4]
OPS = ["rule", "if", "AND", "OR"]          # 关键：遍历/字典序顺序
AUC_ABS_THRESH = 0.10# 白名单判据 |auc-0.5| >= 0.10

CHANNELS_EXPECTED = [
    "CADC0872", "CADC0873", "CADC0874", "CADC0884", "CADC0886",
    "CADC0888", "CADC0890", "CADC0892", "CADC0894",
]

# 目标数值（任务书给定，用于自动比对；不做任何「凑数」调整）
TARGETS = {
    "B": {
        # 精确端点 43/64 = 0.671875（恰为六位小数）、11/16 = 0.6875（精确值仅 4 位，
        # 表中写作 0.687500 属补零对齐六位显示列，非多出有效数字），
        # 未舍入相减 = 0.015625 -> 六位 0.015625，与 PAPER_TABLE.md 表格一致
        "interval": (0.671875, 0.687500),
        "range": 0.015625,
        "median": 0.679689,
        "n_combo": 256,
        "n_unique": 4,
        "point": 0.6795,
        "ties": {
            "CADC0872": ["rule"], "CADC0873": ["AND"], "CADC0874": ["rule"],
            "CADC0884": ["rule", "if", "AND", "OR"],
            "CADC0886": ["rule", "OR"], "CADC0888": ["rule", "AND"],
            "CADC0890": ["rule", "if", "AND", "OR"],
            "CADC0892": ["rule", "OR"], "CADC0894": ["if", "AND"],
        },
    },
    "D": {
        # 精确端点：162/229 = 0.7074235807860262、83/114 = 0.7280701754385965
        # 未舍入相减 = 0.02064659465257029
        # ⚠ 口径说明：0.020646594... 六位舍入应为 0.020647，但 PAPER_TABLE.md
        #   表格取 0.020646，判据是「上界 − 极差 = 下界」的表内自洽
        #   （0.728070 − 0.020646 = 0.707424 ✓；用 0.020647 则得 0.707423 ✗）。
        #   表格自洽优先于单值精度，故此处 "range" 同样取 0.020646。
        # 注：PAPER_TABLE.md 曾把 D 档下界原写成 0.707368，与同组的上界
        #     0.728070、极差 0.020646 不自洽（0.728070 - 0.707368 = 0.020702）。
        #     更正依据是精确有理数 + 规模判据，不是全空间枚举：
        #       0.707368 = 88421/125000（既约），F1 = 2tp/(2tp+fp+fn)
        #       最小总段数 = 2Q - P = 250000 - 88421 = 161579 段
        #     而 test 只有 529 段 -> 该值在本test 规模下不可能产生。
        #     ⚠ 不得写「经全空间枚举证实不可达」：全空间诊断报出的 min/max
        #       不是子空间上下界（F1 非 tp/fp/fn 的线性函数），不能作此对证。
        "interval": (0.707424, 0.728070),
        "interval_as_written": (0.707368, 0.728070),
        "range": 0.020646,
        "range_unrounded": 0.02064659465257029,
        "median": 0.717714,
        "n_combo": 128,
        "n_unique": 8,
        "point": 0.7155,
        "ties": {
            "CADC0872": ["rule"], "CADC0873": ["AND"], "CADC0874": ["rule"],
            "CADC0884": ["rule", "if", "AND", "OR"],
            "CADC0886": ["rule", "OR"], "CADC0888": ["rule", "AND"],
            "CADC0890": ["rule", "if", "AND", "OR"],
            "CADC0892": ["if"],
            "CADC0894": ["if", "AND"],
        },
    },
}


def f1_of(y_true, y_pred):
    return float(_f1(np.asarray(y_true).astype(int),
                     np.asarray(y_pred).astype(int), zero_division=0))


# ----------------------------------------------------------------- 净阈值
def build_net_thresholds(fit_df, feature_cols):
    """净阈值：只用 fit 内正常段估计阈值；某通道正常段 < MIN_NORMAL 则退回全量 fit 阈值。

    返回 (thresholds, 诊断 dict)
    """
    th_all = compute_stat_thresholds(fit_df, feature_cols)
    normal_df = fit_df[fit_df["anomaly"] == 0]
    th_norm = compute_stat_thresholds(normal_df, feature_cols)

    thresholds, diag = {}, {}
    for ch in fit_df["channel"].unique():
        n_normal = int((normal_df["channel"] == ch).sum())
        n_fit = int((fit_df["channel"] == ch).sum())
        fallback = n_normal < MIN_NORMAL
        thresholds[ch] = th_all[ch] if fallback else th_norm[ch]
        diag[ch] = {"n_fit": n_fit, "n_normal": n_normal, "fallback": fallback}
    return thresholds, diag


# ----------------------------------------------------------------- 规则层
def rule_nv(df, thresholds, rule_cols):
    """每段违规特征数（Series，index 与 df.index 对齐）。"""
    nv, _ = apply_stat_rules(df, thresholds, rule_cols)
    return nv.astype(float)


def select_k(val_df, thresholds, rule_cols):
    """全局 k：在 val 上遍历 K_CANDIDATES，严格 > 取最大（并列保留先到者）。"""
    y_val = val_df["anomaly"].to_numpy()
    nv = rule_nv(val_df, thresholds, rule_cols).to_numpy()
    best_k, best_f1, trace = K_CANDIDATES[0], -1.0, []
    for k in K_CANDIDATES:
        f1 = f1_of(y_val, (nv >= k).astype(int))
        trace.append((k, f1))
        if f1 > best_f1:                      # 严格 >：并列保留先到者
            best_f1, best_k = f1, k
    return int(best_k), best_f1, trace


# ----------------------------------------------------------------- IF 层
def fit_if_predict(ch_fit, evals, feature_cols, c):
    """在 ch_fit 上 fit IF（全部 18 特征），对 evals=[(name, df)] 预测 1=异常。"""
    X_fit = np.nan_to_num(ch_fit[feature_cols].values)
    clf = IsolationForest(
        n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
        contamination=c, random_state=IF_SEED, n_jobs=-1,
    ).fit(X_fit)
    return {name: (clf.predict(np.nan_to_num(df[feature_cols].values)) == -1).astype(int)
            for name, df in evals}


def select_contamination(ch_fit, ch_val, feature_cols):
    """在 val 上选 contamination；样本不足返回 None（=>该通道 if 预测恒为 0）。"""
    if len(ch_fit) < MIN_FIT or len(ch_val) < MIN_EVAL:
        return None, "skipped_low_data"
    X_fit = np.nan_to_num(ch_fit[feature_cols].values)
    X_val = np.nan_to_num(ch_val[feature_cols].values)
    y_val = ch_val["anomaly"].to_numpy()
    best_c, best_f1 = CONTAM_GRID[0], -1.0
    for c in CONTAM_GRID:
        clf = IsolationForest(
            n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
            contamination=c, random_state=IF_SEED, n_jobs=-1,
        ).fit(X_fit)
        f1 = f1_of(y_val, (clf.predict(X_val) == -1).astype(int))
        if f1 > best_f1:                      # 严格 >：并列保留先到者
            best_f1, best_c = f1, c
    return float(best_c), "ok"


# ----------------------------------------------------------------- 逐通道 bank
def build_bank(fit_df, val_df, test_df, feature_cols, rule_cols, tag):
    """构造逐通道 4 算子预测 bank（val / test 各一份，按通道内位置对齐）。"""
    thresholds, thr_diag = build_net_thresholds(fit_df, feature_cols)
    k_star, k_val_f1, k_trace = select_k(val_df, thresholds, rule_cols)

    chans = sorted(fit_df["channel"].unique())

    # 规则层预测（整表一次，按 df.index 对齐）
    rule_pred = {"val": {}, "test": {}}
    nv_val = rule_nv(val_df, thresholds, rule_cols)
    nv_test = rule_nv(test_df, thresholds, rule_cols)
    rule_pred["val"]["all"] = (nv_val >= k_star).astype(int).to_numpy()
    rule_pred["test"]["all"] = (nv_test >= k_star).astype(int).to_numpy()

    bank = {"val": {}, "test": {}, "rows": {}, "meta": {}}

    for ch in chans:
        ch_fit = fit_df[fit_df["channel"] == ch]
        ch_val = val_df[val_df["channel"] == ch]
        ch_test = test_df[test_df["channel"] == ch]
        # 通道内位置（用于按通道取 y /拼回）
        rows_val = np.flatnonzero((val_df["channel"] == ch).to_numpy())
        rows_test = np.flatnonzero((test_df["channel"] == ch).to_numpy())
        bank["rows"][ch] = {"val": rows_val, "test": rows_test}

        c_star, c_status = select_contamination(ch_fit, ch_val, feature_cols)
        if c_star is None:
            ifp = {"val": np.zeros(len(ch_val), dtype=int),
                   "test": np.zeros(len(ch_test), dtype=int)}
        else:
            ifp = fit_if_predict(ch_fit, [("val", ch_val), ("test", ch_test)],
                                 feature_cols, c_star)

        for split, rows_len in (("val", len(ch_val)), ("test", len(ch_test))):
            rp = rule_pred[split]["all"][bank["rows"][ch][split]]
            ip = ifp[split]
            r, i = rp.astype(bool), ip.astype(bool)
            op_pred = {
                "rule": rp,
                "if": ip,
                "AND": (r & i).astype(int),      # AND = rule & if
                "OR": (r | i).astype(int),       # OR  = rule | if   （必须是 |）
            }
            # --- 自查断言：OR 是rule/if 的超集，AND 是子集 ---
            if split == "test":
                assert np.all(op_pred["OR"] >= rp) and np.all(op_pred["OR"] >= ip), \
                    f"[{tag}] {ch}: OR>=rule/if 断言失败"
                assert np.all(op_pred["AND"] <= rp) and np.all(op_pred["AND"] <= ip), \
                    f"[{tag}] {ch}: AND<=rule/if 断言失败"
                assert len(op_pred["OR"]) == rows_len
            bank[split][ch] = op_pred

        bank["meta"][ch] = {
            "c_star": c_star, "c_status": c_status,
            "n_fit": len(ch_fit), "n_val": len(ch_val), "n_test": len(ch_test),
            "n_val_pos": int(ch_val["anomaly"].sum()),
            "n_test_pos": int(ch_test["anomaly"].sum()),
            "thresholds": thr_diag[ch],
            "if_all_zero": bool(c_star is None),
        }

    bank["meta"]["_k_star"] = k_star
    bank["meta"]["_k_val_f1"] = k_val_f1
    bank["meta"]["_k_trace"] = k_trace
    return bank


# ----------------------------------------------------------------- 组装 &穷举
def assemble(combo, bank, split, df, chans):
    """按组合逐通道拼回整表预测。

    !! 关键：必须用 df 的**真实行号**写回，绝不能用掩码位置序号。
    """
    pred = np.zeros(len(df), dtype=int)
    idx_all = df.index.to_numpy()
    for pos, ch in enumerate(chans):
        m = (df["channel"] == ch).to_numpy()
        pred[idx_all[m]] = bank[split][ch][combo[pos]]
    return pred


def analyze(tag, bank, val_df, test_df, chans, rule_cols):
    y_val = val_df["anomaly"].to_numpy()
    y_test = test_df["anomaly"].to_numpy()

    # ---- 逐通道 val 最优算子 & 并列集合 ----
    ties, rows_out = {}, []
    for ch in chans:
        r = bank["rows"][ch]["val"]
        yv = y_val[r]
        f1s = {op: f1_of(yv, bank["val"][ch][op]) for op in OPS}
        best = max(f1s.values())
        tie = [op for op in OPS if f1s[op] == best]        # 保持 OPS 顺序
        ties[ch] = tie
        m = bank["meta"][ch]
        rows_out.append({
            "ch": ch, "n_val": m["n_val"], "n_val_pos": m["n_val_pos"],
            "n_fit": m["n_fit"], "n_normal": m["thresholds"]["n_normal"],
            "fallback": m["thresholds"]["fallback"],
            "c": m["c_star"], "if_all_zero": m["if_all_zero"],
            "f1s": f1s, "tie": tie,
        })

    # ---- 穷举（仅并列集合非单元素的通道参与组合）----
    n_combo = 1
    for ch in chans:
        n_combo *= len(ties[ch])
    combos = list(itertools.product(*[ties[ch] for ch in chans]))   # 字典序

    scores = np.empty(len(combos), dtype=float)
    for i, combo in enumerate(combos):
        scores[i] = f1_of(y_test, assemble(combo, bank, "test", test_df, chans))

    uniq = np.unique(scores)
    lo, hi = float(scores.min()), float(scores.max())
    rng = hi - lo
    med = float(np.median(scores))

    # ---- 权威点估计 ----
    # 口径A：逐通道各自独立取「OPS 顺序里第一个达到val 最优的算子」=并列时字典序第一个
    #        这就是 fusion_v3 gate_perchannel 的确定性选择（每个通道独立 argmax）
    gate_combo = tuple(ties[ch][0] for ch in chans)
    gate_point = f1_of(y_test, assemble(gate_combo, bank, "test", test_df, chans))

    # 口径B：穷举 256/128 组合里取 test F1 最大者，并列时取字典序第一个达到的
    best_val, best_combo = -1.0, None
    for combo, s in zip(combos, scores):
        if s > best_val:                     # 严格 >：保留先到者
            best_val, best_combo = float(s), combo
    max_point = float(best_val)

    # ---- 全空间4^9 诊断（分解式：F1 只依赖逐通道 (tp,fp,fn) 之和）----
    tp_t, fp_t, fn_t = [], [], []
    for ch in chans:
        r = bank["rows"][ch]["test"]
        yt = y_test[r]
        tp_t.append([int(((yt == 1) & (bank["test"][ch][op] == 1)).sum()) for op in OPS])
        fp_t.append([int(((yt == 0) & (bank["test"][ch][op] == 1)).sum()) for op in OPS])
        fn_t.append([int(((yt == 1) & (bank["test"][ch][op] == 0)).sum()) for op in OPS])
    TP = np.zeros(1, dtype=np.int64)
    FP = np.zeros(1, dtype=np.int64)
    FN = np.zeros(1, dtype=np.int64)
    for i in range(len(chans)):
        TP = (TP[:, None] + np.array(tp_t[i])[None, :]).ravel()
        FP = (FP[:, None] + np.array(fp_t[i])[None, :]).ravel()
        FN = (FN[:, None] + np.array(fn_t[i])[None, :]).ravel()
    den = 2 * TP + FP + FN
    full_f1 = np.where(den > 0, 2 * TP / np.maximum(den, 1), 0.0)
    full_uniq = np.unique(np.round(full_f1, 12))

    return {
        "tag": tag, "ties": ties, "rows": rows_out, "rule_cols": rule_cols,
        "k_star": bank["meta"]["_k_star"], "k_val_f1": bank["meta"]["_k_val_f1"],
        "k_trace": bank["meta"]["_k_trace"],
        "n_combo": n_combo, "n_unique": int(len(uniq)),
        "interval": (lo, hi), "range": rng, "median": med,
        "point": gate_point, "point_combo": gate_combo,
        "max_point": max_point, "max_combo": best_combo,
        "unique_values": [float(x) for x in uniq],
        "scores": scores,
        "test_pos": int(y_test.sum()), "test_n": int(len(y_test)),
        "tp_t": tp_t, "fp_t": fp_t, "fn_t": fn_t,
        "full_uniq": [float(x) for x in full_uniq],
        "full_min": float(full_f1.min()), "full_max": float(full_f1.max()),
    }


# ----------------------------------------------------------------- 报告
def report(res, targets, diff_channels):
    t = targets
    print("=" * 96)
    print(f"【{res['tag']} 档】净阈值 + {len(res['rule_cols'])} 特征规则层  |  规则层特征: {res['rule_cols']}")
    print("=" * 96)
    print(f"  k* (val 全局, 严格>) = {res['k_star']}   val F1 = {res['k_val_f1']:.6f}"
          f"   轨迹 " + ", ".join(f"k={k}:{f:.6f}" for k, f in res["k_trace"]))
    print()
    print("  逐通道 val 表")
    print(f"  {'通道':<10}{'val段数':>8}{'val正例':>9}{'fit段数':>8}{'正常段':>7}"
          f"{'退回':>6}{'contamination':>15}{'if恒0':>7}   "
          f"{'rule':>8}{'if':>8}{'AND':>8}{'OR':>8}   并列集合")
    for r in res["rows"]:
        c = "None" if r["c"] is None else f"{r['c']:.2f}"
        print(f"  {r['ch']:<10}{r['n_val']:>8}{r['n_val_pos']:>9}{r['n_fit']:>8}"
              f"{r['n_normal']:>7}{'是' if r['fallback'] else '否':>6}{c:>15}"
              f"{'是' if r['if_all_zero'] else '否':>7}   "
              + "".join(f"{r['f1s'][op]:>8.4f}" for op in OPS)
              + f"   {r['tie']}")

    print()
    print(f"  组合数        = {res['n_combo']}   (目标 {t['n_combo']})"
          f"   {'OK' if res['n_combo'] == t['n_combo'] else '*** 不符 ***'}")
    print(f"  唯一值个数    = {res['n_unique']}   (目标 {t['n_unique']})"
          f"   {'OK' if res['n_unique'] == t['n_unique'] else '*** 不符 ***'}")
    print(f"  test F1 区间  = [{res['interval'][0]:.6f}, {res['interval'][1]:.6f}]"
          f"   (目标 [{t['interval'][0]}, {t['interval'][1]}])")
    print(f"  极差          = {res['range']:.6f}   (目标 {t['range']})")
    print(f"  中位数        = {res['median']:.6f}   (目标 {t['median']})")
    print(f"  点估计(口径A 逐通道 val 最优+ 并列取字典序首) = {res['point']:.6f}   (目标 {t['point']})")
    print(f"    A 组合 = {dict(zip(CHANNELS_EXPECTED, res['point_combo']))}")
    print(f"  参考  口径B 穷举 256/128 组合取 test 最大      = {res['max_point']:.6f}")
    print(f"    B 组合 = {dict(zip(CHANNELS_EXPECTED, res['max_combo']))}")
    print(f"  唯一值明细    = {[round(v, 6) for v in res['unique_values']]}")
    print(f"  k* 是否= 2    = {'OK' if res['k_star'] == 2 else '*** 不符 ***'}")

    # 全空间 4^9 诊断（不限并列集合，任意算子组合）
    lo_t, hi_t, rg_t = t["interval"][0], t["interval"][1], t["range"]
    print()
    print("  全空间 4^9 诊断（不限并列集合，任意算子组合）")
    print(f"    test n={res['test_n']}  正例={res['test_pos']}")
    print(f"    全空间 F1 min={res['full_min']:.6f}  max={res['full_max']:.6f}")
    print(f"    全空间唯一值个数 = {len(res['full_uniq'])}"
          f"   不低于目标下界的值 = {[round(v, 6) for v in res['full_uniq'] if v >= lo_t - 1e-9][:12]}")
    tgt_lo, tgt_hi = t["interval"]
    print()
    print("    ⚠ 可达性判定的适用范围（重要，勿误读）：")
    print("      · 下界/上界：子空间穷举已直接算出，端点必然属于子空间，")
    print("        「全空间可达」只是旁证，不构成对证逻辑。")
    print("      · 中位数/点估计：**本来就不要求在全空间可达** —— 它们是")
    print("        512/256/128 组子空间的统计量，全空间可达与否与之无关。")
    print("        上一版把这两项也报成「否」并据此称『经全空间枚举证实』，")
    print("        是逻辑错误：可达性测试根本不是中位数的对证手段。")
    print("      · 全空间 F1 的 min/max 也**不是**子空间的上下界：F1 不是")
    print("        tp/fp/fn 的线性函数，同时优化各通道时最优点不等于各通道")
    print("        独立最优的组合。故 full_min <子区间下界 属正常，不代表有矛盾。")
    # 容差说明：目标值有六位与四位两种写法（如点估计 0.7155 实际是
    # 0.715517），故用 1e-4 判定「是否属于该集合」。
    # 这与前面 checks 里的逐位比对是两种不同用途，不可混用。
    TOL = 1e-4
    sub = [float(x) for x in res["scores"]]
    print(f"    诊断性检查（唯一有效的那项）：目标下界 {tgt_lo} 是否 ∈ 子空间穷举集 = "
          f"{'是' if any(abs(u - tgt_lo) < TOL for u in sub) else '否'}")
    for name, v in (("目标上界", tgt_hi), ("目标中位数", t["median"]),
                    ("目标点估计", t["point"])):
        hit = any(abs(v - u) < TOL for u in sub)
        print(f"    {name} {v} 是否 ∈ 子空间穷举集: {'是' if hit else '否'}"
              f"（子空间口径，非全空间；容差 {TOL}）")
    print(f"    逐通道 test (tp,fp,fn) 矩阵（行=通道, 列={OPS}）")
    for i, ch in enumerate(CHANNELS_EXPECTED):
        print(f"      {ch}: tp={res['tp_t'][i]} fp={res['fp_t'][i]} fn={res['fn_t'][i]}")

    # 并列集合比对
    print()
    print("  并列集合比对（vs 目标）")
    tie_ok = True
    for ch in CHANNELS_EXPECTED:
        got, exp = res["ties"][ch], t["ties"][ch]
        ok = got == exp
        tie_ok &= ok
        mark = "OK" if ok else "*** 不符 ***"
        note = ""
        if not ok:
            note = f"  <-- 目标 {exp}"
        print(f"    {ch}: {got}{note}   {mark}")
    # 与另一档的差异通道
    other = "D" if res["tag"] == "B" else "B"
    print(f"  与 {other} 档并列集合不同的通道: {diff_channels}")

    print()
    print("  目标逐项判定")
    # B 档目标按 4 位小数给出，D 档按 6 位小数给出 —— 按各自精度做舍入比较
    # ⚠ 目标值（EXPECTED）的区间端点与极差一律为六位（与 PAPER_TABLE.md 表格
    #   同口径），故比对也必须用六位。此前 B 档按 nd=4 比对，是在我把 B 档
    #   目标值从四位改成六位之后忘了同步，导致假报警「*** 不符 ***」。
    nd = 6
    checks = [
        ("组合数", res["n_combo"], t["n_combo"]),
        ("唯一值个数", res["n_unique"], t["n_unique"]),
        (f"区间下界(舍入到{nd}位)", round(res["interval"][0], nd), t["interval"][0]),
        (f"区间上界(舍入到{nd}位)", round(res["interval"][1], nd), t["interval"][1]),
        # 极差口径说明：PAPER_TABLE.md 表格取「六位端点相减」（自洽优先），
        # 本实现 res["range"] 是「未舍入端点相减」的真值。两者在 D 档差 1e-6
        # （0.020646 vs 0.020647），是**口径差而非实现差**，故不判成败。
        ("极差(未舍入真值)", res["range"],
         t.get("range_unrounded", t["range"])),
        ("中位数(6位)", round(res["median"], 6), t["median"]),
        ("点估计(舍入到4位)", round(res["point"], 4), t["point"]),
        ("k*", res["k_star"], 2),
        ("并列集合全部一致", tie_ok, True),
    ]
    all_ok = True
    # ⚠ 比对容差：涉及未舍入浮点（如极差真值 0.02064659465257035 vs
    #   0.02064659465257029）时，末位差属 IEEE-754 表示误差，不是实现差异。
    #   故浮点项一律用绝对容差 1e-12 判定，整数/列表项仍用精确相等。
    FTOL = 1e-12
    for name, got, exp in checks:
        if isinstance(got, float) and isinstance(exp, float):
            ok = abs(got - exp) < FTOL
        else:
            ok = got == exp
        all_ok &= ok
        print(f"    {name:<22} 实得={got!s:<12} 目标={exp!s:<12} {'OK' if ok else '*** 不符 ***'}")

    # 目标三数（上下界+ 极差）自洽性
    consistent = abs((hi_t - rg_t) - lo_t) < 1e-9
    print(f"    目标自洽性: 上界-极差={hi_t - rg_t:.6f} vs 下界={lo_t:.6f}-> "
          f"{'自洽 OK' if consistent else '*** 目标三数互相矛盾 ***'}")
    if "interval_as_written" in t:
        lo_w = t["interval_as_written"][0]
        print(f"    原文下界={lo_w:.6f}：上界-该下界={hi_t - lo_w:.6f} "
              f"!= 目标极差 {rg_t:.6f} -> 三数不自洽")
        # 可达性检查：规模判据只能【证成】端点可信，不能【证伪】某个字面量。
        # F1 = 2tp/(2tp+fpn) = P/Q既约 -> 2tp = P*m, 2tp+fpn = Q*m
        #   P 偶取 m=1（P/2 = tp，Q-P = fpn），P 奇取 m=2（tp = P，fpn = 2Q-2P）
        n_test = res["test_n"]          # 从本次运行的真实 test 取，不写字面常量
        n_pos = res["test_pos"]
        n_neg = n_test - n_pos

        def feasible(lit):
            # 返回 (tp, fp+fn)，总段数 = 两者之和
            # ⚠ 必须用字符串构造 Fraction。若传 float，Fraction() 取的是double 的
            #   精确二进制值（分母可达 2^52），规模会算错十几个数量级。
            x = lit if isinstance(lit, Fraction) else Fraction(str(lit))
            p, q = x.numerator, x.denominator
            if p % 2 == 0:          # m=1: 2tp = p
                return p // 2, q - p
            return p, 2 * q - 2 * p  # m=2: 2tp = 2p

        def display_solutions(lit):
            """显示语义穷举：有多少完整混淆矩阵满足 round(F1, 6) == lit。

            ⚠ 必须施加【完整】混淆矩阵约束：
                 tp + fn = n_pos 且 fp + tn = n_neg  （两条等式缺一不可）
               若只约束 tp <= n_pos 与 tp+fp+fn <= n_test，会允许 fp 为负的假解，
               使 B 档解数虚增（0.687500 会由 5 解误报为 10 解）。
               属静默错误：A/D 档因真值恰为 1 解而侥幸不受影响，故务必断言。"""
            hit = []
            for tp_ in range(0, n_pos + 1):
                fn_ = n_pos - tp_                      # 第一条等式
                for fp_ in range(0, n_neg + 1):        # tn_ = n_neg - fp_
                    den = 2 * tp_ + fp_ + fn_
                    if den and round(2 * tp_ / den, 6) == round(lit, 6):
                        tn_ = n_neg - fp_              # 第二条等式
                        assert tp_ + fp_ + fn_ + tn_ == n_test, "混淆矩阵总和应等于 test 段数"
                        assert fn_ <= n_pos and fp_ <= n_neg, "fp/fn 越界"
                        hit.append((tp_, fp_, fn_, tn_))
            return hit

        tp_l, fpn_l = feasible(lo_w)
        lo_exact = Fraction(res["interval"][0]).limit_denominator(10 ** 6)
        tp_r, fpn_r = feasible(lo_exact)
        # 六位显示字面量（显示值 != 精确值，须单独算一次）
        tp_d, fpn_d = feasible(round(res["interval"][0], nd))
        print(f"      规模判据(排除 0.707368): 原文下界 {lo_w:.6f} = "
              f"{Fraction(str(lo_w))} 需{tp_l + fpn_l} 段/tp={tp_l}")
        print(f"      规模判据(证成 穷举实得): 端点 {res['interval'][0]:.16f} = "
              f"{lo_exact} 需 {tp_r + fpn_r} 段/tp={tp_r}")
        print(f"      => 两候选得到不同判决 ({tp_r + fpn_r} <= {n_test} 存活 vs "
              f"{tp_l + fpn_l} > {n_test} 排除)，判据能区分")
        sol_ok = display_solutions(res["interval"][0])
        sol_bad = display_solutions(lo_w)
        print(f"      显示语义穷举(最强, 完整混淆矩阵约束): "
              f"round(F1,{nd})=={round(res['interval'][0], nd):.{nd}f} "
              f"解数={len(sol_ok)} {sol_ok[:2]}")
        if sol_ok:
            t_, p_, f_, n_ = sol_ok[0]
            print(f"        ->该解最简 F1 = {Fraction(2 * t_, 2 * t_ + p_ + f_)}"
                  f"  (应等于穷举端点 {lo_exact})")
        print(f"      显示语义穷举: round(F1,{nd})=={lo_w:.6f} 解数={len(sol_bad)}"
              f"  => {'该字面量在本 test 规模下不可能产生' if not sol_bad else '可产生'}")

        def window(p, q, parity_both=False):
            """可行窗口 + 整性约束 -> (lo, hi, m列表)。用于机理诊断。
            ⚠ 返回三元组，取解数须用返回值[2] 的 len()，直接 len() 得到的是元组长度。
            ⚠ 整性约束【只】施加于 tp = Pm/2，即仅要求 Pm 为偶。
              fpn = m(Q-P) 中 m/Q/P 皆整数 => fpn 自动为整数，【不可】再要求它为偶。
              parity_both=True 保留旧错误口径，仅用于复现该静默错误：
              在 P 偶、Q-P 奇时会排除全部 m，把真解 1 个误报为 0 个。"""
            lo = 2 * n_pos / (2 * q - p)
            hi = min(2 * n_pos / p, 2 * n_test / (2 * q - p))
            # ⚠ 这里必须写成 `not parity_both or ...`。
            #   若写成 `and (parity_both and ...)`，当 parity_both=False 时该子表达式
            #   求值为 False，and 链把【整个条件】判为 False -> 解数恒为 0（静默错误）。
            #   这类 bug 的特征是「程序正常退出、格式正确、数字有说服力」，
            #   故下方加了 assert_parity_equiv 断言做 sanity check。
            ok = [m for m in range(1, 4 * n_test + 2)
                  if lo <= m <= hi
                  and (p * m) % 2 == 0
                  and (not parity_both or (m * (q - p)) % 2 == 0)]
            # 不变量自检：窗口内每个 m 还原出的混淆矩阵须合法且 F1 恰为 p/q
            for m in ok:
                tp_ = p * m // 2
                fpn_ = m * (q - p)
                fn_ = n_pos - tp_
                fp_ = fpn_ - fn_
                tn_ = n_neg - fp_
                assert fn_ >= 0 and fp_ >= 0 and tn_ >= 0, f"m={m} 混淆矩阵越界"
                assert tp_ + fpn_ <= n_test, f"m={m} 总段数超限"
                assert Fraction(2 * tp_, 2 * tp_ + fpn_) == Fraction(p, q), \
                    f"m={m} 还原F1 != {p}/{q}"
            return lo, hi, ok

        def assert_parity_equiv():
            """sanity check：parity_both 开关本身不得改变正确口径的解数。
            曾因写成 `and (parity_both and ...)` 导致解数恒为 0。"""
            for p_, q_ in [(5, 6), (11, 16), (43, 64), (162, 229), (3, 7)]:
                a = len(window(p_, q_, parity_both=False)[2])
                b = len(window(p_, q_, parity_both=True)[2])
                assert a > 0, f"parity_both=False 时 {p_}/{q_} 解数为 0，疑似布尔链写错"
                assert a >= b, f"{p_}/{q_} 旧口径解数反而更多，异常"
            print(f"        [sanity] parity 开关自检通过（正确口径解数均 > 0）")

        def brute_exact(p, q):
            """暴力穷举：F1【恰等于】既约 P/Q 的完整混淆矩阵个数。用于裁决窗口口径。"""
            cnt, ex = 0, []
            for tp_ in range(n_pos + 1):
                fn_ = n_pos - tp_
                for fp_ in range(n_neg + 1):
                    den = 2 * tp_ + fp_ + fn_
                    if den and Fraction(2 * tp_, den) == Fraction(p, q):
                        cnt += 1
                        ex.append((tp_, fp_, fn_, n_neg - fp_))
            return cnt, ex

        def branch_pick(p, q):
            """生效上界由判别式决定：2Q-P <= (N/n_pos)*P ?"""
            return "P项" if (2 * q - p) <= (n_test / n_pos) * p else "2Q-P项"

        # 判别性实验：P 与 Q 各自都能大幅改变解数，单看任一皆不成立。
        # ⚠ fpn 公式曾三次推错：写成 (2Q-P)m/2漏了 2tp 项中已含 m，与暴力枚举差 1。
        # ⚠ 早期版本用「固定 2Q-P」当对照组，但该组 Q 随 P 一起变，不是真正的
        #   单变量对照。改用严格控变量：组B固定 Q 只变P，组A固定 P 只变Q。
        p0 = 15
        ok0 = window(p0, 26)[2]      # 2Q-P = 37；⚠ 取 [2] 才是 m 列表，len() 直接用会算成元组长度
        print(f"      机理: 窗口 [2·n_pos/(2Q-P), min(2·n_pos/P, 2·N/(2Q-P))] "
              f"+ m 整性约束(仅 Pm 偶)；解数 = 窗口内合格 m 的个数")
        grpA = {q_: len(window(5, q_)[2]) for q_ in (6, 7, 8, 12, 14)}
        grpB = {p_: len(window(p_, 16)[2]) for p_ in range(1, 16, 2)}
        assert_parity_equiv()
        print(f"      判别性实验(固定 P=5, 严格只变Q): "
              + " ".join(f"5/{q_}={n_}解" for q_, n_ in grpA.items())
              + f"  => {min(grpA.values())}..{max(grpA.values())} 解, "
                f"{max(grpA.values()) / min(grpA.values()):.0f} 倍差")
        print(f"      判别性实验(固定 Q=16, 严格只变P): "
              + " ".join(f"{p_}/16={n_}解" for p_, n_ in grpB.items())
              + f"  => {min(grpB.values())}..{max(grpB.values())} 解, "
                f"{max(grpB.values()) / min(grpB.values()):.0f} 倍差")
        print(f"        => P、Q 各自都能大幅改变解数，故「分子决定」「分母决定」"
              f"「2Q-P 决定」均不成立，须合并看窗口宽度")

        # 第三维度交叉印证：锁死 2Q-P=37 只变 P。
        # ⚠ 此组 Q 随 P 同变，【不是单变量对照】，仅用于验证 2Q-P 不变时解数仍变。
        grpC = {}
        for p_ in range(1, 16, 2):
            q_ = (37 + p_) // 2
            assert 2 * q_ - p_ == 37, "第三组未锁死 2Q-P"
            grpC[f"{p_}/{q_}"] = len(window(p_, q_)[2])
        s_win = s_enum = 0
        for p_ in range(1, 16, 2):
            q_ = (37 + p_) // 2
            s_win += len(window(p_, q_)[2])
            s_enum += brute_exact(p_, q_)[0]
        assert s_win == s_enum, f"第三组闭式{s_win} != 枚举{s_enum}"
        print(f"      第三维度(锁死 2Q-P=37, 只变P; 非单变量): "
              + " ".join(f"{k}={v}解" for k, v in grpC.items()))
        print(f"        => 2Q-P 钉死时解数仍 {max(grpC.values())}→{min(grpC.values())} 变化，"
              f"进一步排除「2Q-P 单独决定」；窗口/枚举合计均为 {s_win}，逐个一致")

        # 0 解伪差异陷阱普查：len(set(解数))>1 可能只来自 {0,1}，
        # 那只说明「一个不可达、一个可达」，不是判别性证据。
        n_trap = n_all = 0
        for g in range(3, 400, 2):
            cnts = [len(window(p_, (g + p_) // 2)[2])
                    for p_ in range(1, g, 2) if (g + p_) // 2 > p_]
            if not cnts or len(set(cnts)) < 2:
                continue
            n_all += 1
            if set(cnts) <= {0, 1}:
                n_trap += 1
        print(f"      ⚠ 0解伪差异普查: 扫 2Q-P=3..399 全部奇数，"
              f"解数有差异的组={n_all}，其中仅由{{0,1}}构成的伪差异组={n_trap}")
        print(f"        => 筛选差异组须加 set(解数)-{{0}} 非空，否则「不可达 vs 可达」"
              f"会被误当作判别性证据")

        for nm, p_, q_ in [("43/64", 43, 64), ("11/16", 11, 16)]:
            a_, b_, ok_ = window(p_, q_)
            print(f"        B 档 {nm}: 2Q-P={2 * q_ - p_}, 窗口宽={b_ - a_:.1f}, "
                  f"生效上界={branch_pick(p_, q_)} "
                  f"(阈值{(n_test / n_pos) * p_:.1f}), 可行 m={ok_} -> {len(ok_)} 解")

        # 判别式：两分支是否都真实存在（固定 Q=16 只变 P）
        print(f"      判别式: 2Q-P <= (N/n_pos)·P = {n_test / n_pos:.4f}·P ? "
              f"满足取P项，否则取2Q-P项")
        for q_ in (16,):
            row = []
            for p_ in range(1, q_, 2):
                row.append(f"{p_}/{q_}:{len(window(p_, q_)[2])}解"
                           f"({branch_pick(p_, q_)})")
            print(f"        Q={q_} 恒定只变P -> " + "  ".join(row))
        print(f"        => 两分支都真实存在；本test n_pos偏小使 B 档两值均走 P 项")

        # 整性约束口径裁决：只施加 Pm 偶 vs 旧口径(再要求 m(Q-P) 偶)，以暴力穷举为准
        print(f"      ⚠ 整性约束口径裁决（只 Pm 偶 = 正确）:")
        cases = [(150, 241), (156, 241), (162, 229), (43, 64), (11, 16),
                 (83, 114), (2, 3), (2, 5), (4, 9), (8, 9), (1, 16),
                 (15, 16), (5, 21), (15, 26)]
        nbad = 0
        for p_, q_ in cases:
            b_cnt, _ = brute_exact(p_, q_)
            good = len(window(p_, q_, parity_both=False)[2])
            bad = len(window(p_, q_, parity_both=True)[2])
            if b_cnt != good:
                nbad += 1
            flag = "" if bad == good else f"  <-旧口径错报{bad}解"
            print(f"        {p_}/{q_:<4} 穷举={b_cnt:<4} 正确口径={good:<4} "
                  f"旧口径={bad:<4}{flag}")
        assert nbad == 0, f"正确口径与穷举有 {nbad} 处分歧"
        print(f"        => 14 组零分歧，证实「只 Pm 偶」为唯一正确口径；"
              f"旧口径在 P 偶、Q-P 奇时把真解 1 个误报为 0 个")

        # 受影响需三条件同时成立：P偶 & Q-P奇 & 窗口内含奇m
        # ⚠ 扫描须从偶数起步(range(2,...,2))；曾误用 range(1,...,2) 全扫到奇数，
        #   导致已知答案 2/3 都不命中却未察觉。此即纪律 9。
        def has_odd_m(p_, q_):
            return any(m % 2 == 1 for m in window(p_, q_, parity_both=False)[2])

        assert has_odd_m(2, 3), "扫描器自检失败：已知答案 2/3 应判为受影响"
        imm = aff = 0
        for p_ in range(2, 120, 2):
            for q_ in range(p_ + 1, 200):
                if (q_ - p_) % 2 == 1 and has_odd_m(p_, q_):
                    aff += 1
                else:
                    imm += 1
        print(f"        受影响需三条件同时成立(P偶 & Q-P奇 & 窗口含奇m): "
              f"实测 {aff}/{aff + imm} = {100 * aff / (aff + imm):.1f}% 受影响")
        lo_, hi_, ok_ = window(18, 19)
        print(f"        反例 18/19 满足前两条件但窗口[{lo_:.1f},{hi_:.1f}]不含奇m => 免疫; "
              f"本表 B/D 档因 P 为奇数而免疫")
        print(f"      ⚠ 易错：被检验对象是精确端点 {lo_exact}（分母 "
              f"{lo_exact.denominator}），非其六位显示 {round(res['interval'][0], nd):.{nd}f}"
              f" = {Fraction(str(round(res['interval'][0], nd)))}（规模判据下需 "
              f"{tp_d + fpn_d} 段）—— 显示值是舍入产物、从来不是被检验对象，"
              f"且舍入会破坏既约性")
        print(f"      ⚠ 不得写「经全空间枚举证实不可达」：全空间诊断的 min/max "
              f"不是子空间上下界（F1 非 tp/fp/fn 的线性函数），不能作此对证。")

    # 极差口径核对：目标的极差是否等于「舍入后端点相减」
    rng_rounded = round(res["interval"][1], nd) - round(res["interval"][0], nd)
    print(f"    极差两种口径: 本实现(未舍入端点相减)={res['range']:.8f} -> "
          f"舍入{nd}位={round(res['range'], nd)}")
    print(f"                目标口径(舍入端点相减)={rng_rounded:.8f} -> "
          f"与目标 {rg_t} {'一致 OK' if abs(rng_rounded - rg_t) < 1e-9 else '不一致'}")
    print(f"    实得区间端点精确值: lo={res['interval'][0]!r}  hi={res['interval'][1]!r}")
    return all_ok


def main():
    np.random.seed(42)
    fit_df, val_df, test_df, feature_cols = load_split()
    chans = sorted(fit_df["channel"].unique())

    print("#" * 96)
    print("A/B/D穷举 tie-break 敏感性 —— 独立交叉验证探针（第二实现）")
    print("#" * 96)
    print(f"[split] fit={len(fit_df)}  val={len(val_df)}  test={len(test_df)}  "
          f"feature_cols={len(feature_cols)}")
    print(f"[channel] {chans}")
    print(f"[期望通道] {CHANNELS_EXPECTED}")
    print(f"[通道一致] {'OK' if chans == CHANNELS_EXPECTED else '*** 不符 ***'}")
    print(f"[特征列] {feature_cols}")

    # ---- 白名单（按 feature_cols 原始列序）----
    y_val = val_df["anomaly"].to_numpy()
    wl, aucs = [], {}
    for c in feature_cols:
        v = val_df[c]
        ok = ~v.isna().to_numpy()
        a = float(roc_auc_score(y_val[ok], v.to_numpy()[ok].astype(float)))
        aucs[c] = a
        if abs(a - 0.5) >= AUC_ABS_THRESH:
            wl.append(c)                     # 追加即天然保持原始列序
    print()
    print("=" * 96)
    print("D 档白名单：|roc_auc(val_y, feat) - 0.5| >= 0.10，按 feature_cols 原始列序")
    print("=" * 96)
    for c in feature_cols:
        hit = c in wl
        print(f"  {'[IN ]' if hit else '[out]'} {c:<20} auc={aucs[c]:.6f}  |auc-0.5|={abs(aucs[c]-0.5):.6f}")
    print(f"  白名单个数 = {len(wl)}（目标 11）   {'OK' if len(wl) == 11 else '*** 不符 ***'}")
    print(f"  白名单（原始列序）= {wl}")

    # ---- 跑两档 ----
    print()
    print("#" * 96)
    bankB = build_bank(fit_df, val_df, test_df, feature_cols, feature_cols, "B")
    resB = analyze("B", bankB, val_df, test_df, chans, feature_cols)

    bankD = build_bank(fit_df, val_df, test_df, feature_cols, wl, "D")
    resD = analyze("D", bankD, val_df, test_df, chans, wl)

    diff_channels = [ch for ch in CHANNELS_EXPECTED if resB["ties"][ch] != resD["ties"][ch]]

    okB = report(resB, TARGETS["B"], diff_channels)
    print("\n\n")
    okD = report(resD, TARGETS["D"], diff_channels)

    # ---- 档间差异核对 ----
    print()
    print("=" * 96)
    print("B / D 档间差异核对")
    print("=" * 96)
    exp_diff = ["CADC0892"]
    print(f"  实得差异通道 = {diff_channels}   目标 ['CADC0892']"
          f"   {'OK' if diff_channels == exp_diff else '*** 不符 ***'}")
    if diff_channels == ["CADC0892"]:
        print(f"    CADC0892: B={resB['ties']['CADC0892']}  ->  D={resD['ties']['CADC0892']}")
    print(f"  B 档 IF 是否用 18 特征 =是（rule 层用 {len(feature_cols)}）")
    print(f"  D 档 IF 是否用 18 特征 = 是（rule 层用 {len(wl)}）")

    print()
    print("#" * 96)
    print(f"总结：B 档{'完全复现' if okB else '存在不符项'}   D 档{'完全复现' if okD else '存在不符项'}")
    print("#" * 96)


if __name__ == "__main__":
    main()
