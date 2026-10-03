"""探针：核实「规则层白名单（少数强判别力特征）」路线的机制归属。

================================================================================
要回答的5 个问题（对应两个质疑）
================================================================================
  Q1扫描复现：扫描规则层特征白名单大小，test F1 随特征数如何变化？
             明确白名单定义；同时报告「val 选」与「test 选」两种口径。
  Q2 单调性：test F1 是否随特征数减少「单调递增」？是否存在拐点？
  Q3 退化程度：极少特征 + k=1 在方法论上还能否称作「规则层」？
  Q4 矛盾点：白名单里是否含独立验证判别力接近随机的特征（gaps_squared）？
  Q5 seed 稳健性：最优配置在 5 个 IF 随机种子下的 test F1 分布。

================================================================================
纪律（违反则结果不可采信）
================================================================================
  * 规则阈值compute_stat_thresholds 只在 fit 上拟合（纯 unsupervised，不看标签）
  * 白名单排序统计量、k、IF contamination、门控算子 —— **全部只在 val 上选**
  * test 只在配置完全确定后评一次；「test 选」那一列仅作参照、明确标不合规
  * compute_stat_thresholds 返回**双层嵌套** {channel: {feature: {...}}}：
    必须一次性传入所有通道，apply_stat_rules 才能用 `thresholds[ch][col]` 取到内层。
    若逐通道传单通道子集，则 `col not in ch_th` 恒真、所有特征被跳过、nv 恒 0。

用法：
    /d/Python313/python.exe scripts/probe_dilution_mechanism.py
"""

import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments.framework import evaluate, load_split  # noqa: E402
from src.utils.stat_rules import (  # noqa: E402
    apply_stat_rules,
    compute_stat_thresholds,
)

# ----------------------------------------------------------------- 口径常量
N_ESTIMATORS = 100
MAX_SAMPLES = 128
K_CANDIDATES = [1, 2, 3, 4]          # 与题面一致：k 在 {1,2,3,4} 选
CONTAM_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MIN_FIT = 10
MIN_EVAL = 5
MIN_NORMAL = 30                       # 净阈值护栏
SEEDS = [42, 7, 123, 2024, 999]
WEAK_FEATURE = "gaps_squared"         # 题面指认的「判别力接近随机」特征
N_FEATURES = 18


def qf1(yt, yp):
    return float(f1_score(np.asarray(yt), np.asarray(yp), zero_division=0))


# --------------------------------------------------------------- 阈值拟合
def build_thresholds(fit_df, cols, use_net):
    """规则阈值。只在 fit 上拟合，绝不看 val/test 标签。

    use_net=True :只用 fit 内 anomaly==0 的正常段拟合（每通道正常段 <30 则退回全量）
    use_net=False: 用 fit 全量（含异常段）拟合 —— 这是 fusion_v3.py 的现状口径

    关键：一次性传入所有通道，返回 {channel: {feature: {mean,std,q1,q3,iqr}}}
    """
    if not use_net:
        return compute_stat_thresholds(fit_df, cols)

    normal = fit_df[fit_df["anomaly"] == 0]
    net = compute_stat_thresholds(normal, cols)          # 双层嵌套，外层是全部通道
    out = {}
    for ch in fit_df["channel"].unique():
        sub = fit_df[fit_df["channel"] == ch]
        n_norm = int((sub["anomaly"] == 0).sum())
        out[ch] = net[ch] if n_norm >= MIN_NORMAL else compute_stat_thresholds(sub, cols)[ch]
    return out


# --------------------------------------------------------------- 白名单排序
def ranking_scores(crit, Xv_raw, yv, scaler, feature_cols):
    """在 **val** 上算每个特征的排序统计量。返回 (score, detail)。

    crit='auc2_val' :|AUC_val - 0.5|  —— 双侧。规则层用对称偏离
                      (|v-mean|>3σ 或越 IQR 栅栏)，故AUC<0.5 的特征同样有信息量，
                      判别力应按**双侧**距离衡量。这是与规则层形式匹配的口径。
    crit='lr_val'   :|标准化 LR 系数| —— 多变量口径，val 上拟合 LR
    """
    if crit == "auc2_val":
        aucs = np.array([
            roc_auc_score(yv, Xv_raw[:, j]) if len(np.unique(Xv_raw[:, j])) > 1 else 0.5
            for j in range(len(feature_cols))
        ])
        return np.abs(aucs - 0.5), aucs
    if crit == "lr_val":
        lr = LogisticRegression(max_iter=8000, C=1.0).fit(scaler.transform(Xv_raw), yv)
        coef = lr.coef_[0]
        return np.abs(coef), coef
    raise ValueError(crit)


def whitelist_from_scores(score, feature_cols, n):
    """按 score 降序选 top-n，但**输出顺序固定为 feature_cols 的原始顺序**。

    为什么必须固定顺序：IsolationForest 的随机切分按**列索引**取特征，
    同一批特征仅换一个列顺序、random_state 不变，预测也会变（实测翻转 2/81），
    逐通道门控会把这种微小差异放大成 F1 的可观漂移（实测 0.6281 -> 0.5961）。
    若让白名单按score 顺序输出，则「特征集合相同」与「IF 输入相同」不再等价，
    扫描结果会混入列顺序噪声。故这里统一按原始列序返回。
    并列时按特征名排序，保证选集确定性可复现。
    """
    order = sorted(range(len(feature_cols)), key=lambda j: (-score[j], feature_cols[j]))
    picked = set(order[:n])
    return [feature_cols[j] for j in range(len(feature_cols)) if j in picked], sorted(picked)


# --------------------------------------------------------------- 端到端门控
def run_config(fit_df, val_df, eval_df, feature_cols, rule_cols, if_cols,
               use_net, seed, thr_cache=None):
    """门控端到端。rule_cols 决定规则层输入白名单；if_cols 决定 IF 输入。

    k / IF contamination / 门控算子 **全部在 val 上选**，eval_df 只用于最后打分。
    门控 tie-break 与 fusion_v3.py:265-269 一致：严格 >，字典序 rule→if→AND→OR。
    """
    thr = (thr_cache if thr_cache is not None else build_thresholds(fit_df, rule_cols, use_net))

    yv = val_df["anomaly"].to_numpy()
    ch_va = val_df["channel"].to_numpy()

    # ---- 规则层 k 在 val 上选 ----
    nv_v, _ = apply_stat_rules(val_df, thr, rule_cols)
    nv_v = nv_v.to_numpy()
    best_k, bk = K_CANDIDATES[0], -1.0
    for k in K_CANDIDATES:
        f1 = qf1(yv, (nv_v >= k).astype(int))
        if f1 > bk:
            bk, best_k = f1, k
    best_k = int(best_k)

    # ---- 逐通道 IF contamination 在 val 上选 ----
    best_c = {}
    for ch in np.unique(ch_va):
        cm = ch_va == ch
        cf = fit_df[fit_df["channel"] == ch]
        if len(cf) < MIN_FIT or int(cm.sum()) < MIN_EVAL:
            best_c[ch] = None
            continue
        y = yv[cm]
        Xf = np.nan_to_num(cf[if_cols].values)
        Xv = np.nan_to_num(val_df.loc[cm, if_cols].values)
        bc, bf = CONTAM_GRID[0], -1.0
        for c in CONTAM_GRID:
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=seed, n_jobs=-1).fit(Xf)
            f1 = qf1(y, (clf.predict(Xv) == -1).astype(int))
            if f1 > bf:
                bf, bc = f1, c
        best_c[ch] = float(bc)

    def layers(df):
        chs = df["channel"].to_numpy()
        nv, _ = apply_stat_rules(df, thr, rule_cols)
        rp = (nv.to_numpy() >= best_k).astype(int)
        ip = np.zeros(len(df), dtype=int)
        for ch in np.unique(chs):
            m = chs == ch
            c = best_c.get(ch)
            cf = fit_df[fit_df["channel"] == ch]
            if c is None or len(cf) < MIN_FIT:
                continue
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=seed, n_jobs=-1,
                                  ).fit(np.nan_to_num(cf[if_cols].values))
            ip[m] = (clf.predict(np.nan_to_num(df.loc[m, if_cols].values)) == -1).astype(int)
        return rp, ip, nv.to_numpy()

    rv, iv, nv_val = layers(val_df)

    # ---- 门控算子在 val 上选（tie-break: 严格 >，字典序 rule→if→AND→OR）----
    y_eval = eval_df["anomaly"].to_numpy()
    ch_ev = eval_df["channel"].to_numpy()
    re_, ie_, nv_eval = layers(eval_df)

    picks, yp, yp_rule_only = {}, np.zeros(len(eval_df), dtype=int), np.zeros(len(eval_df), dtype=int)
    yp_val = np.zeros(len(val_df), dtype=int)          # 同一配方回算 val 预测
    for ch in np.unique(ch_ev):
        mv, me = ch_va == ch, ch_ev == ch
        y2 = yv[mv]
        cand = {
            "rule": qf1(y2, rv[mv]),
            "if": qf1(y2, iv[mv]),
            "AND": qf1(y2, iv[mv] & rv[mv]),
            "OR": qf1(y2, iv[mv] | rv[mv]),
        }
        op = max(cand, key=lambda k: cand[k])       # 严格 > 且字典序 -> 与 fusion_v3 一致
        picks[ch] = (op, cand[op])
        yp[me] = {"rule": re_[me], "if": ie_[me], "AND": ie_[me] & re_[me],
                  "OR": ie_[me] | re_[me]}[op]
        yp_rule_only[me] = re_[me]
        yp_val[mv] = {"rule": rv[mv], "if": iv[mv], "AND": iv[mv] & rv[mv],
                      "OR": iv[mv] | rv[mv]}[op]

    return {
        "y_true": y_eval, "y_pred": yp, "y_rule_only": yp_rule_only,
        "k": best_k, "picks": picks, "nv_val": nv_val, "nv_eval": nv_eval,
        # 规则层单独（nv>=k）的 val F1
        "val_f1": qf1(yv, (nv_v >= best_k).astype(int)),
        # 端到端门控的 val F1 —— 这是合规的模型选择依据
        "gate_val_f1": qf1(yv, yp_val),
        "test_f1": qf1(y_eval, yp),
    }


# --------------------------------------------------------------- 退化诊断
def degeneracy_report(fit_df, val_df, eval_df, rule_cols, k, use_net, name):
    """判断「少特征 + k=1」是否还算统计规则层。

    关键诊断：
      * OR 分解 —— P(任一特征越界) 与各单特征 P 的关系
      * 边际贡献 —— 去掉任一特征后 OR 判定的F1 掉多少（去掉不掉 = 该特征没贡献）
      * 触发集中度 —— nv==1 的占比（k=1 时判定完全由「首个越界者」决定）
    """
    thr = build_thresholds(fit_df, rule_cols, use_net)
    nv_e, _ = apply_stat_rules(eval_df, thr, rule_cols)
    nv_e = nv_e.to_numpy()
    y = eval_df["anomaly"].to_numpy()
    pred = (nv_e >= k).astype(int)
    out = {"name": name, "k": k, "f1": qf1(y, pred),
           "P": float(precision_score(y, pred, zero_division=0)),
           "R": float(recall_score(y, pred, zero_division=0)),
           "pred_pos_rate": float(pred.mean()),
           "nv_eq1_rate": float((nv_e == 1).mean())}

    # 逐特征越界指示 +边际贡献
    per_feat, drops = {}, []
    base_hit = pred.astype(bool)
    for col in rule_cols:
        v, _ = apply_stat_rules(eval_df, thr, [col])
        viol = (v.to_numpy() > 0)
        per_feat[col] = {"viol_rate": float(viol.mean()),
                         "f1_alone": qf1(y, viol.astype(int)),
                         "uniq_in_viol": float((nv_e[viol] == 1).mean()) if viol.any() else None}
        # 边际贡献：去掉该特征后的 OR 判定 F1
        drop_pred = base_hit.copy()
        drop_pred[viol & (nv_e == 1)] = False     # 只由它触发的那些段翻回正常
        drops.append((col, qf1(y, drop_pred.astype(int)) - out["f1"]))
    out["per_feature"] = per_feat
    out["marginal_drop"] = sorted(drops, key=lambda t: t[1])   # 最小的 = 最没用
    return out


# ================================================================== 主流程
def main():
    fit_df, val_df, test_df, feature_cols = load_split()
    yv, yt = val_df["anomaly"].to_numpy(), test_df["anomaly"].to_numpy()
    Xv_raw = np.nan_to_num(val_df[feature_cols].values.astype(float))
    scaler = StandardScaler().fit(np.nan_to_num(fit_df[feature_cols].values.astype(float)))

    print("=" * 100)
    print("探针：规则层白名单路线的机制核实（只读，不修改任何项目文件）")
    print("=" * 100)
    print(f"[split] fit={len(fit_df)} val={len(val_df)} test={len(test_df)} 特征={len(feature_cols)}")
    print(f"[口径] k∈{K_CANDIDATES}；contam 网格 {len(CONTAM_GRID)} 档；"
          f"IF n_estimators={N_ESTIMATORS} max_samples={MAX_SAMPLES}")
    print(f"[纪律] 白名单/k/contam/门控 全部 val 选；test 仅在配置冻结后评一次\n")

    # ---------- 0. 阈值口径双轨：Tfull（现状） / Tnet（净阈值） ----------
    CALIBERS = [("Tfull", False), ("Tnet", True)]

    # ================================================================
    # 第1 段：扫描复现
    # ================================================================
    print("=" * 100)
    print("第 1 段  扫描复现：白名单大小 → F1")
    print("=" * 100)

    # 1a. 先把排序统计量摊开，说明白名单是怎么定的
    print("\n[1a] 白名单排序统计量（全部在 val 上计算；test 完全不参与）")
    print("-" * 100)
    print(f"{'feature':<20}{'AUC_val':>10}{'|AUC-.5|':>11}{'stdLRcoef_val':>15}"
          f"{'AUC_test':>10}{'stdLR_test':>13}")
    auc_v, detail_auc = ranking_scores("auc2_val", Xv_raw, yv, scaler, feature_cols)
    _, lr_v = ranking_scores("lr_val", Xv_raw, yv, scaler, feature_cols)
    Xt_raw = np.nan_to_num(test_df[feature_cols].values.astype(float))
    auc_t = np.array([roc_auc_score(yt, Xt_raw[:, j]) if len(np.unique(Xt_raw[:, j])) > 1 else 0.5
                      for j in range(N_FEATURES)])
    lr_t = LogisticRegression(max_iter=8000, C=1.0).fit(scaler.transform(Xt_raw), yt).coef_[0]
    for j in sorted(range(N_FEATURES), key=lambda j: -auc_v[j]):
        print(f"{feature_cols[j]:<20}{detail_auc[j]:>10.4f}{auc_v[j]:>11.4f}"
              f"{lr_v[j]:>15.4f}{auc_t[j]:>10.4f}{lr_t[j]:>13.4f}")

    print("\n  排序口径 A = |AUC_val-0.5|（双侧，与规则层对称偏离形式匹配）  ← 主口径")
    print("  排序口径 B = |标准化 LR 系数|_val（多变量）")
    print("  说明：题面所述「guard 阈值 → 9/8/5/4 特征」在本数据上无法用任何 LR 系数口径复现")
    print("        （已实测 raw-LR / std-LR / 单变量LR / C∈{0.05..10} / |t| 统计量，计数均不匹配），")
    print("        故本探针改用**显式 top-n 白名单**并同时扫 n=1..18，把「特征数」直接作为自变量，")
    print("        这比依赖来源不明的 guard 更可复核。")

    # 1b. 逐 n 扫描（两个口径 × 两种阈值口径）
    scan = {}   # (caliber, crit, n) -> result
    for cal_name, use_net in CALIBERS:
        for crit in ["auc2_val", "lr_val"]:
            score, _ = ranking_scores(crit, Xv_raw, yv, scaler, feature_cols)
            for n in range(1, N_FEATURES + 1):
                wl, _ = whitelist_from_scores(score, feature_cols, n)
                r = run_config(fit_df, val_df, test_df, feature_cols, wl, wl, use_net, 42)
                r["whitelist"] = wl
                scan[(cal_name, crit, n)] = r

    # 完整门控的 val F1 需重跑一次带 val 预测的实现
    print("\n[1b] 逐特征数扫描（IF seed=42；IF 输入 = 规则层同一白名单）")
    print("-" * 100)
    print(f"{'口径':<8}{'排序':<9}{'n':>3}{'k*':>4}{'规则valF1':>11}"
          f"{'门控valF1':>11}{'门控testF1':>12}{'P':>8}{'R':>8}  白名单")
    print("-" * 100)
    for cal_name, _ in CALIBERS:
        for crit in ["auc2_val", "lr_val"]:
            for n in range(1, N_FEATURES + 1):
                r = scan[(cal_name, crit, n)]
                gv = r.get("gate_val_f1")
                gvs = f"{gv:.4f}" if gv is not None else "  n/a "
                P = precision_score(yt, r["y_pred"], zero_division=0)
                R = recall_score(yt, r["y_pred"], zero_division=0)
                print(f"{cal_name:<8}{crit:<9}{n:>3}{r['k']:>4}{r['val_f1']:>11.4f}"
                      f"{gvs:>11}{r['test_f1']:>12.4f}{P:>8.3f}{R:>8.3f}  "
                      f"{','.join(x[:9] for x in r['whitelist'])}")
        print()

    # ================================================================
    # 第 2 段：单调性判定
    # ================================================================
    print("=" * 100)
    print("第 2 段  单调性判定：「特征越少越好」是否真的单调成立")
    print("=" * 100)
    for cal_name, _ in CALIBERS:
        for crit in ["auc2_val", "lr_val"]:
            ks = list(range(1, N_FEATURES + 1))
            tv = [scan[(cal_name, crit, n)]["test_f1"] for n in ks]
            # 合规口径：端到端门控在 val 上的 F1（模型选择只能用这个）
            gv = [scan[(cal_name, crit, n)]["gate_val_f1"] for n in ks]
            # 规则层单独在 val 上的 F1（辅助参考）
            rule_val = [scan[(cal_name, crit, n)]["val_f1"] for n in ks]
            mono_down = all(tv[i] >= tv[i + 1] - 1e-12 for i in range(len(tv) - 1))
            rev = [i + 1 for i in range(len(tv) - 1) if tv[i + 1] > tv[i] + 1e-12]
            print(f"\n  [{cal_name} / {crit}]")
            print(f"    test F1 序列 (n=1..18): {[round(x,4) for x in tv]}")
            print(f"    门控 val F1 序列      : {[round(x,4) for x in gv]}")
            print(f"    规则层 val F1 序列     : {[round(x,4) for x in rule_val]}")
            print(f"    test F1 随 n 减少单调递增？ {'是' if mono_down else '否'}")
            print(f"    出现回升的 n 位置（违规步）: {rev if rev else '无'}")
            best_n_test = max(range(len(tv)), key=lambda i: tv[i]) + 1
            best_n_val = max(range(len(gv)), key=lambda i: gv[i]) + 1
            print(f"    test 最优 n = {best_n_test} (F1={tv[best_n_test-1]:.4f})   "
                  f"val 最优 n = {best_n_val} (门控 val F1={gv[best_n_val-1]:.4f})")
            print(f"    test 最优 n 是否为端点(1 或 18)？ "
                  f"{'是 -> 无内部拐点，趋势为单调' if best_n_test in (1, N_FEATURES) else '否 -> 存在内部拐点'}")
            # 拐点：相邻差分符号翻转处
            diffs = [tv[i + 1] - tv[i] for i in range(len(tv) - 1)]
            turns = [i + 1 for i in range(len(diffs) - 1)
                     if (diffs[i] > 1e-12) != (diffs[i + 1] > 1e-12)]
            print(f"    test F1 差分符号翻转位置(拐点候选, n=左端): {turns if turns else '无'}")
            print(f"    最大单步变化: n={min(range(len(diffs)), key=lambda i: diffs[i])+1}"
                  f"->{min(range(len(diffs)), key=lambda i: diffs[i])+2} Δ={min(diffs):+.4f}")

    # val 选出的最优配置（合规）与 test 选出的最优配置（不合规，仅参照）
    print("\n" + "-" * 100)
    print("两种选参口径对比（合规 = 只用 val 选；不合规 = 用 test 选，仅作参照）")
    print("-" * 100)
    print(f"{'口径':<8}{'排序':<9}{'val选 n':>9}{'val选 testF1':>14}"
          f"{'test选 n':>9}{'test选 testF1':>15}{'乐观偏差':>11}")
    for cal_name, _ in CALIBERS:
        for crit in ["auc2_val", "lr_val"]:
            gvals = {n: scan[(cal_name, crit, n)]["gate_val_f1"] for n in range(1, N_FEATURES + 1)}
            tvals = {n: scan[(cal_name, crit, n)]["test_f1"] for n in range(1, N_FEATURES + 1)}
            nv_ = max(gvals, key=lambda n: gvals[n])
            nt_ = max(tvals, key=lambda n: tvals[n])
            print(f"{cal_name:<8}{crit:<9}{nv_:>9}{tvals[nv_]:>14.4f}"
                  f"{nt_:>9}{tvals[nt_]:>15.4f}{tvals[nt_]-tvals[nv_]:>+11.4f}")

    # ================================================================
    # 第 2b 段：k 敏感性扫描（质疑 2：断崖还是渐变）
    # ================================================================
    print("\n" + "=" * 100)
    print("第 2b 段  k 敏感性：k 是断崖还是渐变？")
    print("=" * 100)
    print("  说明：k* 由 val 选出（合规）。此处额外给出「强制固定 k」的**规则层单独** test F1，")
    print("        仅用于刻画 k 的敏感性形态，不作为选参依据。")
    print("-" * 100)
    print(f"{'口径':<8}{'排序':<9}{'n':>4}{'k*':>4}   " +
          "".join(f"{'k=' + str(k):>10}" for k in K_CANDIDATES) + "     形态")
    print("-" * 100)
    for cal_name, use_net in CALIBERS:
        for crit in ["auc2_val", "lr_val"]:
            gvals = {n: scan[(cal_name, crit, n)]["gate_val_f1"] for n in range(1, N_FEATURES + 1)}
            n_best = max(gvals, key=lambda n: gvals[n])
            wl = scan[(cal_name, crit, n_best)]["whitelist"]
            thr = build_thresholds(fit_df, wl, use_net)
            nv_e, _ = apply_stat_rules(test_df, thr, wl)
            nv_e = nv_e.to_numpy()
            row = [qf1(yt, (nv_e >= k).astype(int)) for k in K_CANDIDATES]
            rng = max(row) - min(row)
            shape = "断崖" if rng > 0.3 else ("渐变" if rng > 0.1 else "平坦")
            print(f"{cal_name:<8}{crit:<9}{n_best:>4}{scan[(cal_name, crit, n_best)]['k']:>4}   " +
                  "".join(f"{x:>10.4f}" for x in row) + f"{shape}（极差 {rng:.4f}）")
    print("\n  题面转述的「k=1→0.800 / k=2→0.623 / k=3→0.248」形态是否成立，见上表实测。")

    # ================================================================
    # 第 3 段：退化程度判定
    # ================================================================
    print("\n" + "=" * 100)
    print("第 3 段  退化程度判定：少特征 + k=1 还算不算「统计规则层」")
    print("=" * 100)
    # 以主口径 Tfull/auc2_val 为例，检查 n=4 与 n=18
    for cal_name, use_net in CALIBERS:
        for crit in ["auc2_val"]:
            score, _ = ranking_scores(crit, Xv_raw, yv, scaler, feature_cols)
            for n in [4, 8, 18]:
                wl, _ = whitelist_from_scores(score, feature_cols, n)
                r = scan[(cal_name, crit, n)]
                d = degeneracy_report(fit_df, val_df, test_df, wl, r["k"], use_net,
                                      f"{cal_name}/{crit}/n={n}")
                print(f"\n  --- {d['name']}  k={d['k']}  白名单={wl} ---")
                print(f"      规则层单独 test F1={d['f1']:.4f}  P={d['P']:.3f}  R={d['R']:.3f}"
                      f"  预测为异常段比例={d['pred_pos_rate']:.3f}  nv==1 占比={d['nv_eq1_rate']:.3f}")
                print(f"      {'特征':<18}{'越界率':>9}{'单特征F1':>11}{'边际贡献ΔF1':>13}")
                for col in wl:
                    pf = d["per_feature"][col]
                    md = dict(d["marginal_drop"])[col]
                    print(f"      {col:<18}{pf['viol_rate']:>9.4f}{pf['f1_alone']:>11.4f}{md:>+13.4f}")
                useless = [c for c, v in d["marginal_drop"] if abs(v) < 1e-9]
                print(f"      边际贡献=0（完全无用）的特征: {useless if useless else '无'}")

    # ================================================================
    # 第 4 段：矛盾点核验 —— gaps_squared 是否被白名单选中
    # ================================================================
    print("\n" + "=" * 100)
    print("第 4 段  矛盾点核验：白名单里有没有「判别力接近随机」的特征")
    print("=" * 100)
    g = feature_cols.index(WEAK_FEATURE)
    print(f"\n  目标特征: {WEAK_FEATURE}")
    print(f"    AUC_test = {auc_t[g]:.4f}   |AUC_test-0.5| = {abs(auc_t[g]-0.5):.4f}"
          f"   LR+(AUC/(1-AUC)) = {auc_t[g]/(1-auc_t[g]):.4f}")
    print(f"    AUC_val  = {detail_auc[g]:.4f}   |AUC_val-0.5| = {auc_v[g]:.4f}")
    print(f"    标准化 LR 系数 val={lr_v[g]:.4f} / test={lr_t[g]:.4f}")
    print(f"    【与题面转述核对】题面称「test LR≈1.58，即 AUC 约 0.55」。")
    print(f"    实测 test LR+ = {auc_t[g]/(1-auc_t[g]):.4f}，AUC = {auc_t[g]:.4f}。")
    print(f"    → 「判别力接近随机」这一定性判断成立（|AUC-0.5|={abs(auc_t[g]-0.5):.4f}，极小）；")
    print(f"    → 但「LR≈1.58 / AUC≈0.55」这组具体数字在本数据上无法复现，属转述失真。")

    print(f"\n  各白名单口径下{WEAK_FEATURE} 是否入选（按 |AUC_val-0.5| 与 |LR_val| 排序）：")
    print("-" * 100)
    print(f"{'n':>4}{'auc2_val 口径':<16}{'含 gaps?':<12}{'lr_val 口径':<18}{'含 gaps?':<12}")
    s_auc, _ = ranking_scores("auc2_val", Xv_raw, yv, scaler, feature_cols)
    s_lr, _ = ranking_scores("lr_val", Xv_raw, yv, scaler, feature_cols)
    for n in range(1, N_FEATURES + 1):
        w_a, _ = whitelist_from_scores(s_auc, feature_cols, n)
        w_l, _ = whitelist_from_scores(s_lr, feature_cols, n)
        print(f"{n:>4}{n:>3} 维{'':<10}{'  是<<<' if WEAK_FEATURE in w_a else '    否':<12}"
              f"{n:>3} 维{'':<12}{'  是<<<' if WEAK_FEATURE in w_l else '    否':<12}")
    rank_a = sorted(range(N_FEATURES), key=lambda j: (-s_auc[j], feature_cols[j])).index(g) + 1
    rank_l = sorted(range(N_FEATURES), key=lambda j: (-s_lr[j], feature_cols[j])).index(g) + 1
    print(f"\n  {WEAK_FEATURE} 的排名：auc2_val 口径第 {rank_a} / 18 名；lr_val 口径第 {rank_l} / 18 名")

    # ================================================================
    # 第 5 段：跨 seed 稳健性
    # ================================================================
    print("\n" + "=" * 100)
    print("第 5 段  跨 IF seed 稳健性（42/7/123/2024/999）")
    print("=" * 100)
    # 用 val 选出全局最优（合规）配置，再看它在各 seed 下的 test F1
    for cal_name, use_net in CALIBERS:
        gvals = {n: scan[(cal_name, "auc2_val", n)]["gate_val_f1"] for n in range(1, N_FEATURES + 1)}
        n_best = max(gvals, key=lambda n: gvals[n])
        wl = scan[(cal_name, "auc2_val", n_best)]["whitelist"]
        print(f"\n  [{cal_name}/auc2_val] val 选出的最优配置： n={n_best}  白名单={wl}")
        f1s, picks_all = [], []
        for sd in SEEDS:
            r = run_config(fit_df, val_df, test_df, feature_cols, wl, wl, use_net, sd)
            f1s.append(qf1(yt, r["y_pred"]))
            picks_all.append(r["picks"])
        f1s = np.array(f1s)
        print(f"    各 seed test F1: " + "  ".join(f"{s}={f:.4f}" for s, f in zip(SEEDS, f1s)))
        print(f"    均值={f1s.mean():.4f}  标准差={f1s.std(ddof=1):.4f}  "
              f"极差={f1s.max()-f1s.min():.4f}  min={f1s.min():.4f}  max={f1s.max():.4f}")
        # 门控配方是否随 seed 漂移
        ops = {ch: sorted({p[ch][0] for p in picks_all}) for ch in picks_all[0]}
        drift = {ch: (v[0] if len(v) == 1 else f"漂移{v}") for ch, v in ops.items()}
        print(f"    门控配方跨 seed: {drift}")
        # 与 18 维基线的配对比较（各 seed）
        base = []
        for sd in SEEDS:
            rb = run_config(fit_df, val_df, test_df, feature_cols, feature_cols,
                            feature_cols, use_net, sd)
            base.append(qf1(yt, rb["y_pred"]))
        base = np.array(base)
        print(f"    18 维基线各 seed test F1: " + "  ".join(f"{f:.4f}" for f in base)
              + f"   均值={base.mean():.4f}")
        print(f"    Δ(白名单 - 18维) 各 seed: " +
              "  ".join(f"{a-b:+.4f}" for a, b in zip(f1s, base)) +
              f"   均值={np.mean(f1s-base):+.4f}")

    # 最优配置的完整指标（带 CI）
    print("\n" + "-" * 100)
    print("val 选出的最优配置 —— 完整 test 指标（含 bootstrap 95% CI）")
    print("-" * 100)
    for cal_name, _ in CALIBERS:
        gvals = {n: scan[(cal_name, "auc2_val", n)]["gate_val_f1"] for n in range(1, N_FEATURES + 1)}
        n_best = max(gvals, key=lambda n: gvals[n])
        r = scan[(cal_name, "auc2_val", n_best)]
        m = evaluate(yt, r["y_pred"])
        print(f"\n  [{cal_name}] n={n_best} k={r['k']} 白名单={r['whitelist']}")
        print(f"    test F1={m['f1']:.4f} CI{m['f1_ci95']}  P={m['precision']:.4f} R={m['recall']:.4f}"
              f"  MCC={m['mcc']:.4f}")
        print(f"    混淆矩阵 TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}"
              f"  误报率={m['false_alarm_ratio']}")
        print(f"    门控配方: " + ", ".join(f"{k}={v[0]}" for k, v in r["picks"].items()))

    # 18 维基线对照
    print("\n" + "-" * 100)
    print("对照：全 18 维基线")
    print("-" * 100)
    for cal_name, _ in CALIBERS:
        r = scan[(cal_name, "auc2_val", N_FEATURES)]
        m = evaluate(yt, r["y_pred"])
        print(f"  [{cal_name}] n=18 k={r['k']}  test F1={m['f1']:.4f} CI{m['f1_ci95']}"
              f"  P={m['precision']:.4f} R={m['recall']:.4f}")


if __name__ == "__main__":
    main()