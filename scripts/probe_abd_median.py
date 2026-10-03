"""探针：核实 A/B/D 三档的 tie-break 完整分布 + 部署变体 + 全局优先级规则。

起因（自查）：
  本文PAPER_TABLE.md 4.2a 表中的「中位数」一列来自 agent 转述，未经本机核实。
  外部更正称正确值为 A 0.634864 / B 0.679689 / D 0.717714，
  与文档现值（A 0.6349 / B 0.6802 / D 0.7159）不一致。必须实测定谳。

纪律：
  全部超参（k、contamination、白名单、门控算子）只在 val 上选；
  test 仅在组合确定后评估一次。本脚本只读。

易错点（踩过，勿删）：
  1) 数组必须按下标对齐 df 自然行序，不能按通道顺序 concatenate 后
     直接与 y_true 比 —— 会错位算出失真值。fusion_v3.py:190 有同款警告。
     test 侧即便 y_all 也按通道构造，仍统一走下标写回以免埋雷。
  2) compute_stat_thresholds 返回双层嵌套 {channel: {feature: {...}}}；
     必须一次性传入所有通道的正常段再逐通道取内层 out[ch] = one[ch]，
     不可把内层当外层（会 KeyError）。
  3) IsolationForest 按列索引取特征，特征子集必须锁定原始列序。

用法：
    /d/Python313/python.exe scripts/probe_abd_median.py
"""

import os
import sys
import warnings
from itertools import product

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments.framework import load_split  # noqa: E402
from src.utils.stat_rules import (  # noqa: E402
    apply_stat_rules,
    compute_stat_thresholds,
)

N_EST = 100
MAX_SAMP = 128
CONTAM = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MIN_FIT, MIN_EVAL, MIN_NORMAL = 10, 5, 30
OPS = ["rule", "if", "AND", "OR"]
WL_GUARD = 0.10  # 先验常数，选定前未看 test


def qf1(yt, yp):
    return float(f1_score(np.asarray(yt), np.asarray(yp), zero_division=0))


def net_thr(fit_df, cols):
    """净阈值：仅用 fit 内正常段估计，附 MIN_NORMAL 护栏。"""
    normal = fit_df[fit_df["anomaly"] == 0]
    n_norm = normal.groupby("channel").size().to_dict()
    one = compute_stat_thresholds(normal, cols)
    full = compute_stat_thresholds(fit_df, cols)
    th = {}
    for ch in sorted(set(fit_df["channel"])):
        th[ch] = one[ch] if int(n_norm.get(ch, 0)) >= MIN_NORMAL else full[ch]
    return th


def build(fit_df, val_df, test_df, rule_cols, if_cols, thr, seed=42):
    """为每通道算 4 算子的 val F1 与 test 预测。"""
    yv = val_df["anomaly"].to_numpy()
    yt = test_df["anomaly"].to_numpy()
    chv = val_df["channel"].to_numpy()
    cht = test_df["channel"].to_numpy()

    nv_v = apply_stat_rules(val_df, thr, rule_cols)[0]
    nv_t = apply_stat_rules(test_df, thr, rule_cols)[0]

    bk, bf = 1, -1.0
    for k in (1, 2, 3, 4):
        f1 = qf1(yv, (nv_v >= k).astype(int))
        if f1 > bf:
            bf, bk = f1, k

    best_c = {}
    for ch in np.unique(chv):
        cm = chv == ch
        cf = fit_df[fit_df["channel"] == ch]
        if len(cf) < MIN_FIT or int(cm.sum()) < MIN_EVAL:
            best_c[ch] = None
            continue
        Xf = np.nan_to_num(cf[if_cols].values)
        bc, bcf = CONTAM[0], -1.0
        for c in CONTAM:
            m = IsolationForest(n_estimators=N_EST, max_samples=MAX_SAMP,
                                contamination=c, random_state=seed,
                                n_jobs=-1).fit(Xf)
            s = qf1(yv[cm], (m.predict(np.nan_to_num(
                val_df.loc[cm, if_cols].values)) == -1).astype(int))
            if s > bcf:
                bcf, bc = s, c
        best_c[ch] = float(bc)

    bank = {}
    for ch in np.unique(cht):
        mv, mt = chv == ch, cht == ch
        cf = fit_df[fit_df["channel"] == ch]
        rv = (nv_v[mv] >= bk).astype(int)
        rt = (nv_t[mt] >= bk).astype(int)
        c = best_c.get(ch)
        if c is None or len(cf) < MIN_FIT:
            iv = np.zeros(int(mv.sum()), dtype=int)
            it = np.zeros(int(mt.sum()), dtype=int)
        else:
            m = IsolationForest(n_estimators=N_EST, max_samples=MAX_SAMP,
                                contamination=c, random_state=seed,
                                n_jobs=-1).fit(np.nan_to_num(cf[if_cols].values))
            iv = (m.predict(np.nan_to_num(
                val_df.loc[mv, if_cols].values)) == -1).astype(int)
            it = (m.predict(np.nan_to_num(
                test_df.loc[mt, if_cols].values)) == -1).astype(int)
        vp = {"rule": rv, "if": iv, "AND": rv & iv, "OR": rv | iv}
        # ⚠ OR 必须是 | 而非 &。首版 tp 里误写成 rt & it（vp 是对的），
        # 导致 OR≡AND，解除了退化通道的合并约束、唯一值数 14/16/16、
        # A 档区间虚增为 [0.5939, 0.6383]—— 与另一实现的 OR bug 同型。
        tp = {"rule": rt, "if": it, "AND": rt & it, "OR": rt | it}
        # 自检：AND 逐位 <= rule 且 <= if；OR 逐位 >= 两者。
        # 若 OR 误写为&，第二条对 rule 失效（OR 不再是超集）而被第一条掩盖，
        # 故两条必须同检。退化通道另检：IF 侧全0 -> OR≡rule、AND≡if。
        for nm, t in (("val", vp), ("test", tp)):
            assert np.all(t["AND"] <= t["rule"]) and np.all(t["AND"] <= t["if"]), \
                f"{nm}/{ch}: AND 不是子集"
            assert np.all(t["OR"] >= t["rule"]) and np.all(t["OR"] >= t["if"]), \
                f"{nm}/{ch}: OR 不是超集（疑似误写为 &）"
        for nm, t, base in (("val", vp, iv), ("test", tp, it)):
            if base.sum() == 0:
                assert np.array_equal(t["OR"], t["rule"]), \
                    f"{nm}/{ch}: IF 侧全 0 时 OR 应恒等于 rule"
                assert np.array_equal(t["AND"], base), \
                    f"{nm}/{ch}: IF 侧全 0 时 AND 应恒等于 if(全 0)"
        vf = {op: qf1(yv[mv], p) for op, p in vp.items()}
        mx = max(vf.values())
        bank[ch] = {
            "val_f1": vf, "test_pred": tp, "val_pred": vp,
            "opt": [op for op in OPS if abs(vf[op] - mx) < 1e-12],
            "n_val": int(mv.sum()), "n_pos_val": int(yv[mv].sum()),
            "n_test": int(mt.sum()), "k": bk,
            "test_pos": (int(yt[mt].sum()),),
        }
    return bank, yv, yt, chv, cht, bk, test_df.index.to_numpy(), val_df.index.to_numpy()


def eval_combo(bank, cht, yt, test_idx, cols_ch, choice):
    """按 choice 逐通道拼回 test，算段级 F1。

    ⚠ 必须用 df 的真实行号（test_idx）写回，不能用 np.flatnonzero(mask)。
    布尔掩码的位置序号与 df 自然行序只在 df 恰好按通道排序时才相等；
    test_df 非按通道排序时两者错位，会算出失真区间。
    首版即因此把 A 档区间算成 [0.5939, 0.6383]（OR bug 的错误值）。
    fusion_v3.py:190 已有同款警告注释。
    """
    pred = np.zeros(len(yt), dtype=int)
    for ch in cols_ch:
        m = cht == ch
        pred[test_idx[m]] = bank[ch]["test_pred"][choice[ch]]
    return qf1(yt, pred)


def eval_combo_val(bank, chv, yv, val_idx):
    """val 侧门控 F1。同样必须用 df 真实行号写回，理由同 eval_combo。"""
    pred = np.zeros(len(yv), dtype=int)
    for ch in sorted(bank):
        m = chv == ch
        vf = bank[ch]["val_f1"]
        mx = max(vf.values())
        op = [o for o in OPS if abs(vf[o] - mx) < 1e-12][0]
        pred[val_idx[m]] = bank[ch]["val_pred"][op]
    return qf1(yv, pred)


def report(bank, yv, yt, chv, cht, tag, k, test_idx, val_idx):
    chs = sorted(bank)
    print()
    print("=" * 92)
    print(f"{tag}   （k*={k}，选k 口径：val）")
    print("=" * 92)
    print(f"{'通道':<12}{'val段':>6}{'val正':>6}{'并列数':>7}  可选算子")
    print("-" * 92)
    for ch in chs:
        b = bank[ch]
        print(f"{ch:<12}{b['n_val']:>6}{b['n_pos_val']:>6}"
              f"{len(b['opt']):>7}  {b['opt']}")

    n_full = sum(1 for c in chs if len(bank[c]["opt"]) == 4)
    n_part = sum(1 for c in chs if 1 < len(bank[c]["opt"]) < 4)
    n_uniq = sum(1 for c in chs if len(bank[c]["opt"]) == 1)
    print(f"\n  全并列 {n_full}/9   部分并列 {n_part}/9   唯一最优 {n_uniq}/9")

    total = 1
    for c in chs:
        total *= len(bank[c]["opt"])
    combos = list(product(*[bank[c]["opt"] for c in chs]))
    assert len(combos) == total, (len(combos), total)

    # 按通道拼接 y（与 pred 构造顺序无关，eval 内已用下标对齐）
    f1s = np.array([eval_combo(bank, cht, yt, test_idx, chs,
                                {c: o for c, o in zip(chs, cb)})
                    for cb in combos])
    uniq = np.unique(np.round(f1s, 10))

    auth = {c: bank[c]["opt"][0] for c in chs}  # 严格 > + 字典序
    a_t = eval_combo(bank, cht, yt, test_idx, chs, auth)
    a_v = eval_combo_val(bank, chv, yv, val_idx)

    assert f1s.min() - 1e-12 <= a_t <= f1s.max() + 1e-12, (
        f"权威值 {a_t} 不在枚举区间 [{f1s.min()}, {f1s.max()}] 内——"
        f"穷举边界定义有误")
    assert len(uniq) == len(set(np.round(uniq, 10))), "唯一值重复"

    med = float(np.median(f1s))
    med_u = float(np.median(uniq))
    lt = float((f1s < a_t - 1e-12).mean() * 100)
    le = float((f1s <= a_t + 1e-12).mean() * 100)

    print(f"\n  val F1（门控后）      = {a_v:.6f}")
    print(f"  组合数                = {total}")
    print(f"  test 唯一值个数       = {len(uniq)}")
    print(f"  test F1 区间          = [{f1s.min():.4f}, {f1s.max():.4f}]"
          f"   极差 {f1s.max()-f1s.min():.4f}")
    print(f"  中位数 np.median(全部) = {med:.6f}")
    print(f"  中位数 np.median(唯一) = {med_u:.6f}")
    print(f"  第{np.argmax(np.sort(uniq))+1} 低唯一值          = "
          f"{np.sort(uniq)[np.argmax(np.sort(uniq))]:.6f}")
    print(f"  权威点估计            = {a_t:.6f}   "
          f"(严格< {lt:.1f}%  |  <= {le:.1f}%)")
    print(f"  sorted(uniq)[len//2]  = {sorted(uniq)[len(uniq)//2]:.6f}"
          f"   <-- 易误当中位数")
    return {"k": k, "val": a_v, "test": a_t, "lo": float(f1s.min()),
            "hi": float(f1s.max()), "med": med, "med_u": med_u,
            "n_combo": total, "n_uniq": len(uniq), "lt": lt, "le": le,
            "n_full": n_full, "n_part": n_part, "n_uniq_ch": n_uniq,
            "f1s": f1s, "bank": bank}


def deployment_variants(bank, cht, yt, chs, tag, test_idx, chv, yv):
    """部署变体对照：各单一全局算子 + 文档所述的保守回退策略。"""
    print()
    print("-" * 92)
    print(f"部署变体对照（{tag}）：test F1")
    print("-" * 92)
    res = {}
    for op in OPS:
        res[f"全9通道纯 {op}"] = eval_combo(
            bank, cht, yt, test_idx, chs, {c: op for c in chs})
    uniq_ch = [c for c in chs if len(bank[c]["opt"]) == 1]
    res["无并列通道用最优+并列通道回退 rule"] = eval_combo(
        bank, cht, yt, test_idx, chs,
        {c: (bank[c]["opt"][0] if c in uniq_ch else "rule") for c in chs})
    res["权威（逐通道门控·字典序）"] = eval_combo(
        bank, cht, yt, test_idx, chs, {c: bank[c]["opt"][0] for c in chs})
    for k2, v in res.items():
        print(f"  {k2:<42} {v:.4f}")
    print(f"  （无并列通道：{uniq_ch}）")

    # --- 文档 4.2 第 5 条所述策略：「val 异常段数 < 5 或 val 段数 < 30
    # 的通道强制回退纯规则」。此处逐条实测，不接受"其规则 F1 已不差"的推断。
    print()
    print("  文档所述保守回退策略的逐档实测（A/B/D 通用判据，仅用 val 计数）：")
    variants = {}
    for name, thr_npos, thr_nval in (
        ("val正<5 或 val段<30 -> rule", 5, 30),
        ("val正<5 -> rule", 5, 0),
        ("val段<30 -> rule", 0, 30),
        ("val正<10 或 val段<30 -> rule", 10, 30),
    ):
        fb = [c for c in chs
              if bank[c]["n_pos_val"] < thr_npos or bank[c]["n_val"] < thr_nval]
        v_rule = eval_combo(
            bank, cht, yt, test_idx, chs,
            {c: ("rule" if c in fb else bank[c]["opt"][0]) for c in chs})
        v_and = eval_combo(
            bank, cht, yt, test_idx, chs,
            {c: ("AND" if c in fb else bank[c]["opt"][0]) for c in chs})
        variants[name] = (fb, v_rule, v_and)
        print(f"    {name:<32} 通道{len(fb)}个{str(fb)}")
        print(f"{'':>36}回退 rule = {v_rule:.4f}   回退 AND = {v_and:.4f}")
    print(f"    {'（参照）不强制回退 = 权威门控':<32} "
          f"{res['权威（逐通道门控·字典序）']:.4f}")
    best = max(
        [("全 AND", res["全9通道纯 AND"]),
         ("权威门控", res["权威（逐通道门控·字典序）"])]
        + [(n + f"/rule", v[1]) for n, v in variants.items()]
        + [(n + f"/AND", v[2]) for n, v in variants.items()],
        key=lambda x: x[1])
    print(f"    -> 本档最优部署变体：{best[0]} = {best[1]:.4f}")

    # 四种全局优先级 tie-break
    print()
    print("  四种全局 tie-break 优先级规则：")
    for op in OPS:
        v = eval_combo(bank, cht, yt, test_idx, chs, {c: op for c in chs})
        note = "  <- 现行字典序(rule优先)" if op == "rule" else ""
        print(f"    {op:>4} 优先{'':<2} test F1 = {v:.4f}{note}")
    return res, variants


def main():
    fit_df, val_df, test_df, cols18 = load_split()
    print(f"[split] fit={len(fit_df)} val={len(val_df)} test={len(test_df)}"
          f"  n_feat={len(cols18)}")

    # 白名单：val |rank_auc - 0.5| >= 0.10（仅用 val）
    yv0 = val_df["anomaly"].to_numpy()
    WL = []
    for c in cols18:
        v = val_df[c].values
        if len(np.unique(v)) < 2 or not np.all(np.isfinite(v)):
            continue
        if abs(roc_auc_score(yv0, np.nan_to_num(v)) - 0.5) >= WL_GUARD:
            WL.append(c)
    WL = [c for c in cols18 if c in set(WL)]  # 锁定原始列序
    print(f"[WL] {len(WL)} 特征（|val rank_auc-0.5| >= {WL_GUARD}，仅用 val）")

    thr_full = compute_stat_thresholds(fit_df, cols18)
    thr_net18 = net_thr(fit_df, cols18)
    thr_net11 = net_thr(fit_df, WL)

    rows = []
    # A：全量阈值 + 18 特征（本文基线）
    bA, yv, yt, chv, cht, kA, ti, vi = build(
        fit_df, val_df, test_df, cols18, cols18, thr_full)
    rA = report(bA, yv, yt, chv, cht,
                "A档：全量阈值 + 18 特征（本文基线）", kA, ti, vi)
    deployment_variants(bA, cht, yt, sorted(bA), "A 档", ti, chv, yv)
    rows.append(("A 全量阈值+18特征", rA))

    # B：净阈值 + 18 特征
    bB, yv, yt, chv, cht, kB, ti, vi = build(
        fit_df, val_df, test_df, cols18, cols18, thr_net18)
    rB = report(bB, yv, yt, chv, cht, "B 档：净阈值 + 18 特征", kB, ti, vi)
    rows.append(("B 净阈值+18特征", rB))

    # D：净阈值 + 11 特征白名单（规则层 WL，IF 层 18 维 = 权威口径）
    bD, yv, yt, chv, cht, kD, ti, vi = build(
        fit_df, val_df, test_df, WL, cols18, thr_net11)
    rD = report(bD, yv, yt, chv, cht,
                "D 档：净阈值 + 11 特征白名单（推荐改进）", kD, ti, vi)
    rows.append(("D 净阈值+11特征", rD))
    # D 档才是交付文档的推荐口径，部署变体必须在 D 上核实（A 档结论不可套用）
    deployment_variants(bD, cht, yt, sorted(bD), "D 档（推荐口径）", ti, chv, yv)

    print()
    print("=" * 100)
    print("三档汇总（本文可写入交付文档的数值）")
    print("=" * 100)
    print(f"{'档':<24}{'k':>3}{'组合数':>7}{'唯一值':>7}{'val F1':>10}"
          f"{'test下界':>10}{'test上界':>10}{'极差':>9}{'中位数':>10}"
          f"{'权威值':>10}")
    print("-" * 100)
    for name, r in rows:
        print(f"{name:<24}{r['k']:>3}{r['n_combo']:>7}{r['n_uniq']:>7}"
              f"{r['val']:>10.4f}{r['lo']:>10.4f}{r['hi']:>10.4f}"
              f"{r['hi']-r['lo']:>9.4f}{r['med']:>10.4f}{r['test']:>10.4f}")
    print()
    a, b, d = rA, rB, rD
    print(f"  A->B  净阈值区间贡献 = B.lo - A.hi = {b['lo']-a['hi']:+.4f}")
    print(f"  B->D  白名单区间贡献 = D.lo - B.hi = {d['lo']-b['hi']:+.4f}")
    print(f"  A->D  合计（最保守）  = D.lo - A.hi = {d['lo']-a['hi']:+.4f}")
    print(f"  三档两两不重叠: "
          f"A|B={b['lo'] > a['hi']}  B|D={d['lo'] > b['hi']}  A|D={d['lo'] > a['hi']}")
    print()
    print("  中位数口径核对（三种算法）：")
    for name, r in rows:
        print(f"    {name:<24} median(全部)={r['med']:.6f}  "
              f"median(唯一值)={r['med_u']:.6f}")
    print()
    print("  分位（两种口径）：")
    for name, r in rows:
        print(f"    {name:<24} 严格< ={r['lt']:5.1f}%   <= = {r['le']:5.1f}%")


if __name__ == "__main__":
    main()
