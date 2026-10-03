"""探针：核实 D 档「IF 输入维度 × contamination 选法」2×2 口径矩阵。

起因（待裁决）：
  某路复算报出 D 档存在一个更高口径 0.7288（IF 用 11 维训练、contamination
  仍按 18 维选 = 不对称混合）。另一路主张按 val F1 最高者选，应维持 0.7155。
  两者都声称 val 数值，但报的val 值不同：
     一路: 权威 val=0.720000 / 0.7288 的 val=0.714286
     另一路: 权威 val=0.7159  / 0.7288 的 val=0.7001
  本脚本用fusion_v3.py 的**权威函数**（非重实现）逐格实算，裁决哪个val 对。

纪律：
  全部超参（k、contamination、白名单、门控算子）只在 val 上选，test 只评一次。
  本脚本只读，不修改任何生产代码或结果文件。

用法：
    /d/Python313/python.exe scripts/probe_d2x2_calibrate.py
"""

import os
import sys
import warnings
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments.framework import load_split  # noqa: E402
from src.utils.stat_rules import compute_stat_thresholds  # noqa: E402
from scripts.fusion_v3 import (  # noqa: E402
    fit_if_channel,
    if_score,
    quick_f1,
    rule_nv,
    select_if_contamination,
    select_rule_k,
)

OPS = ["rule", "if", "AND", "OR"]
MIN_NORMAL = 30  # 净阈值护栏：正常段不足则退回全量阈值


def net_thresholds(fit_df, cols):
    """净阈值：仅用 fit 内正常段估计参考分布，附MIN_NORMAL 护栏。

    注意compute_stat_thresholds 返回**双层嵌套** {channel: {feature: {...}}}。
    必须一次性传入所有通道的正常段再逐通道取内层；若逐通道传单通道子集再
    update 合并，结构仍合法（nv 正常），但把内层当外层用会直接 KeyError。
    """
    normal = fit_df[fit_df["anomaly"] == 0]
    n_normal = normal.groupby("channel").size()
    th = compute_stat_thresholds(normal, cols)
    full = compute_stat_thresholds(fit_df, cols)
    for ch in n_normal.index:
        if int(n_normal[ch]) < MIN_NORMAL:
            th[ch] = full[ch]  # 正常段不足 -> 退回全量
    return th, {ch: int(n) for ch, n in n_normal.items()}


def main():
    fit_df, val_df, test_df, feature_cols = load_split()
    F3 = ["kurtosis", "n_peaks", "smooth10_n_peaks"]
    print(f"[split] fit={len(fit_df)} val={len(val_df)} test={len(test_df)} "
          f"n_feat={len(feature_cols)}")

    # WL 11 特征：以 val |rank_auc - 0.5| >= 0.10 筛（仅用 val）
    from sklearn.metrics import roc_auc_score
    yv = val_df["anomaly"].to_numpy()
    scores = {}
    for c in feature_cols:
        v = val_df[c].values
        if np.all(~np.isfinite(v)) or len(np.unique(v)) < 2:
            continue
        a = roc_auc_score(yv, np.nan_to_num(v))
        scores[c] = abs(a - 0.5)
    WL = sorted([c for c, s in scores.items() if s >= 0.10])
    print(f"[WL] {len(WL)} 特征（判据 |val rank_auc-0.5| >= 0.10）: {WL}")

    cols18 = list(feature_cols)
    # 锁定原始列序（IF 按列索引取特征，列序变化会改变预测）
    WL_ordered = [c for c in cols18 if c in set(WL)]

    print()
    print("=" * 96)
    print("逐格明细（val / test 均为逐通道门控后的段级 F1）")
    print("=" * 96)
    print(f"{'IF输入':>8}{'c选法':>8}{'k*':>4}{'val F1':>10}{'test F1':>10}"
          f"{'test下界':>11}{'test上界':>11}{'极差':>9}  性质")
    print("-" * 96)

    results = {}
    for if_cols, c_cols, ifname, cname in (
            (cols18, cols18, "18维", "18维"),
            (cols18, WL_ordered, "18维", "11维"),
            (WL_ordered, cols18, "11维", "18维"),
            (WL_ordered, WL_ordered, "11维", "11维")):
        row = cell_full(fit_df, val_df, test_df, WL_ordered, if_cols, c_cols)
        kind = "自洽" if ifname == cname else "★不对称"
        results[(ifname, cname)] = row
        print(f"{ifname:>8}{cname:>8}{row['k']:>4}{row['val']:>10.6f}"
              f"{row['test']:>10.6f}{row['lo']:>11.4f}{row['hi']:>11.4f}"
              f"{row['hi'] - row['lo']:>9.4f}  {kind}")

    print()
    print("=" * 96)
    print("裁决")
    print("=" * 96)
    auth = results[("18维", "18维")]
    dec = results[("11维", "11维")]
    mix = results[("11维", "18维")]
    mix2 = results[("18维", "11维")]
    print(f"  权威自洽(IF18+c18)   val={auth['val']:.6f}  test={auth['test']:.6f}")
    print(f"  解耦自洽(IF11+c11)   val={dec['val']:.6f}  test={dec['test']:.6f}"
          f"   Δtest = {dec['test'] - auth['test']:+.4f}")
    print(f"  不对称  (IF11+c18)   val={mix['val']:.6f}  test={mix['test']:.6f}"
          f"   Δtest = {mix['test'] - auth['test']:+.4f}")
    print(f"  不对称  (IF18+c11)   val={mix2['val']:.6f}  test={mix2['test']:.6f}"
          f"   Δtest = {mix2['test'] - auth['test']:+.4f}")
    print()
    hi_val = max(results, key=lambda k: results[k]["val"])
    hi_test = max(results, key=lambda k: results[k]["test"])
    print(f"  按 val 选  -> IF{hi_val[0]}+c{hi_val[1]}  "
          f"val={results[hi_val]['val']:.6f}  test={results[hi_val]['test']:.6f}")
    print(f"  按 test 选 -> IF{hi_test[0]}+c{hi_test[1]}  "
          f"val={results[hi_test]['val']:.6f}  test={results[hi_test]['test']:.6f}")
    print(f"  同解? {'是' if hi_val == hi_test else '否'}")
    print()
    if dec["test"] < auth["test"]:
        print("  => 解耦方向**无提升**（自洽解耦反而更差），"
              "不应为 fusion_v3.py 四处共用 feature_cols 立项改代码")
    else:
        print("  => 解耦方向有提升，需进一步讨论")


def cell_full(fit_df, val_df, test_df, rule_cols, if_cols, c_src_cols):
    """完整一格：含 tie-break 全枚举区间。"""
    thresholds, _ = net_thresholds(fit_df, rule_cols)
    nv_v_all = rule_nv(val_df, thresholds, rule_cols).to_numpy()
    nv_t_all = rule_nv(test_df, thresholds, rule_cols).to_numpy()

    best_k, bk = 1, -1.0
    for k in (1, 2, 3, 4, 5):
        f1 = quick_f1(val_df["anomaly"].to_numpy(), (nv_v_all >= k).astype(int))
        if f1 > bk:
            bk, best_k = f1, k

    best_c = select_if_contamination(val_df, fit_df, c_src_cols)
    yv, yt = val_df["anomaly"].to_numpy(), test_df["anomaly"].to_numpy()
    chv, cht = val_df["channel"].to_numpy(), test_df["channel"].to_numpy()
    chans = sorted(set(cht))

    opt_sets, test_preds, val_preds = {}, {}, {}
    for ch in chans:
        cf = fit_df[fit_df["channel"] == ch]
        c = best_c.get(ch)
        clf = (fit_if_channel(cf, if_cols, c)
               if (c is not None and len(cf) >= 10) else None)
        mv, mt = chv == ch, cht == ch
        rv = (nv_v_all[mv] >= best_k).astype(int)
        rt = (nv_t_all[mt] >= best_k).astype(int)
        if clf is None:
            iv = np.zeros(int(mv.sum()), dtype=int)
            it = np.zeros(int(mt.sum()), dtype=int)
        else:
            iv = (if_score(clf, val_df[mv], if_cols) > 0).astype(int)
            it = (if_score(clf, test_df[mt], if_cols) > 0).astype(int)
        vp = {"rule": rv, "if": iv, "AND": rv & iv, "OR": rv | iv}
        tp = {"rule": rt, "if": it, "AND": rt & it, "OR": rt | it}
        vf = {op: quick_f1(yv[mv], p) for op, p in vp.items()}
        mx = max(vf.values())
        opt_sets[ch] = [op for op in OPS if abs(vf[op] - mx) < 1e-12]
        test_preds[ch], val_preds[ch] = tp, vp

    y_all = np.concatenate([yt[cht == ch] for ch in chans])
    # 权威点估计（严格 > + 字典序）。
    # ⚠ 必须按下标对齐到 df 的自然行序后再与 y_true 比——绝不能按通道顺序
    #   concatenate 后去和 y_all 比。val_df 非按通道排序，逐通道拼接会错位
    #   （首版探针即因此把 val F1 算成 0.256，test 侧因顺序恰好一致而未暴露）。
    #   fusion_v3.py:190 的注释已警告过此坑。
    auth_t = np.zeros(len(test_df), dtype=int)
    auth_v = np.zeros(len(val_df), dtype=int)
    for ch in chans:
        op = opt_sets[ch][0]
        mt, mv = cht == ch, chv == ch
        auth_t[test_df.index.to_numpy()[mt]] = test_preds[ch][op]
        auth_v[val_df.index.to_numpy()[mv]] = val_preds[ch][op]

    # 全枚举区间（test 侧；按通道拼接后与 y_all 同序，此处 y_all 亦按通道构造故一致）
    allf1 = []
    for combo in product(*[opt_sets[ch] for ch in chans]):
        allf1.append(quick_f1(y_all, np.concatenate(
            [test_preds[ch][op] for ch, op in zip(chans, combo)])))
    allf1 = np.array(allf1)
    return {"k": best_k,
            "val": quick_f1(yv, auth_v),
            "test": quick_f1(yt, auth_t),
            "lo": float(allf1.min()), "hi": float(allf1.max()),
            "n_combo": len(allf1)}


if __name__ == "__main__":
    main()
