"""探针：定位 IsolationForest 表达力瓶颈的归属。

要回答的唯一问题：
    IF 的 AUC 只有 0.6557，是「算法本身上限低」，
    还是「输入没归一化 / 量纲没处理」导致的？

判据设计：
  A. 18 维原始特征 -> IF              （现状）
  B. 18 维 RobustScaler -> IF          （仅改输入，算法不变）
  C. 18 维 StandardScaler -> IF
  D. 18 维 RobustScaler -> IF 污染率按真实异常率设定（而非网格选）
  E. 18 维 RobustScaler -> RandomForest（对照，验证瓶颈不在特征）
  F. 只用单特征 n_peaks -> IF（验证「单特征已达多少」，判断是否 IF 天生吃不下多维）

铁律：所有 scaler 只在 fit 上 fit，val/test 只做 transform。
      超参（contamination）只在 val 上选；test 只在选定后评一次。

用法：
    /d/Python313/python.exe scripts/probe_if_scaling.py
"""

import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments.framework import load_split  # noqa: E402

N_ESTIMATORS = 100
MAX_SAMPLES = 128
CONTAM_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def if_auc(X_fit, y_fit, X_eval, y_eval, contam):
    clf = IsolationForest(
        n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
        contamination=contam, random_state=42, n_jobs=-1,
    ).fit(X_fit)
    s = -clf.decision_function(X_eval)
    return roc_auc_score(y_eval, s)


def main():
    fit_df, val_df, test_df, feature_cols = load_split()
    X_fit_raw = np.nan_to_num(fit_df[feature_cols].values)
    y_fit = fit_df["anomaly"].to_numpy()
    X_val_raw = np.nan_to_num(val_df[feature_cols].values)
    y_val = val_df["anomaly"].to_numpy()
    X_test_raw = np.nan_to_num(test_df[feature_cols].values)
    y_test = test_df["anomaly"].to_numpy()

    # ---- 1. 先量一下量纲差异到底有多大 -------------------------------
    print("=" * 78)
    print("1. 特征量纲审计（IF 的输入长什么样）")
    print("=" * 78)
    stats = []
    for i, c in enumerate(feature_cols):
        col = X_fit_raw[:, i]
        lo, hi = np.percentile(col, [1, 99])
        stats.append((c, col.mean(), col.std(), lo, hi))
    print(f"{'feature':<20}{'mean':>13}{'std':>13}{'p1':>13}{'p99':>13}")
    print("-" * 72)
    for c, m, s, lo, hi in stats:
        print(f"{c:<20}{m:>13.4g}{s:>13.4g}{lo:>13.4g}{hi:>13.4g}")
    nz = [s[3] for s in stats if s[3] != 0]
    if nz:
        print(f"\n非零 p1 的 max/min 倍数 = {max(nz) / min(nz):.3g}"
              f"   <- 若达1e3 以上，IF 的随机切分几乎必然被大尺度轴主导")
    print(f"真实 fit 异常率 = {y_fit.mean():.4f}")

    # ---- 2. 缩放器只在 fit 上 fit -------------------------------------
    rs = RobustScaler().fit(X_fit_raw)
    ss = StandardScaler().fit(X_fit_raw)
    scal = {
        "raw": (X_fit_raw, X_val_raw, X_test_raw),
        "robust": (rs.transform(X_fit_raw), rs.transform(X_val_raw), rs.transform(X_test_raw)),
        "standard": (ss.transform(X_fit_raw), ss.transform(X_val_raw), ss.transform(X_test_raw)),
    }

    # ---- 3. 各配置：污染率在 val 上选，test 只评一次 --------------------
    print()
    print("=" * 78)
    print("2. 核心实验：IF 在不同输入下的判别力")
    print("=" * 78)
    print(f"{'配置':<34}{'c*(val选)':>11}{'val AUC':>10}{'test AUC':>10}{'vs raw':>9}")
    print("-" * 74)
    base_test_auc = None
    rows = {}
    for name, (Xf, Xv, Xt) in scal.items():
        best_c, best_auc = CONTAM_GRID[0], -1.0
        for c in CONTAM_GRID:
            a = if_auc(Xf, y_fit, Xv, y_val, c)
            if a > best_auc:
                best_auc, best_c = a, c
        clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                              contamination=best_c, random_state=42, n_jobs=-1).fit(Xf)
        t_auc = roc_auc_score(y_test, -clf.decision_function(Xt))
        if base_test_auc is None:
            base_test_auc = t_auc
        rows[name] = (best_c, best_auc, t_auc)
        print(f"IF + {name:<28}{best_c:>11.2f}{best_auc:>10.4f}{t_auc:>10.4f}"
              f"{t_auc - base_test_auc:>+9.4f}")

    # ---- 4. 对照：同样 RobustScaler 输入下的有监督模型 ------------------
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(
        scal["robust"][0], y_fit)
    rf_val = roc_auc_score(y_val, rf.predict_proba(scal["robust"][1])[:, 1])
    rf_test = roc_auc_score(y_test, rf.predict_proba(scal["robust"][2])[:, 1])
    print(f"{'RandomForest + robust（对照）':<34}{'-':>11}{rf_val:>10.4f}{rf_test:>10.4f}"
          f"{rf_test - base_test_auc:>+9.4f}")

    # ---- 5. 单特征对照：IF 天生吃不下多维吗 ----------------------------
    print()
    print("=" * 78)
    print("3. 单特征 vs 全特征（判断瓶颈在「维数」还是「算法」）")
    print("=" * 78)
    for c in ["n_peaks", "diff2_var", "var_div_duration", "kurtosis"]:
        i = feature_cols.index(c)
        Xf = X_fit_raw[:, [i]]
        Xv = X_val_raw[:, [i]]
        Xt = X_test_raw[:, [i]]
        best_c, best_auc = CONTAM_GRID[0], -1.0
        for cc in CONTAM_GRID:
            a = if_auc(Xf, y_fit, Xv, y_val, cc)
            if a > best_auc:
                best_auc, best_c = a, cc
        clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                              contamination=best_c, random_state=42, n_jobs=-1).fit(Xf)
        ta = roc_auc_score(y_test, -clf.decision_function(Xt))
        print(f"  IF 仅用 {c:<18} c*={best_c:.2f}  val AUC={best_auc:.4f}  test AUC={ta:.4f}")

    # ---- 6. 结论判据 ---------------------------------------------------
    print()
    print("=" * 78)
    print("4. 归属判据")
    print("=" * 78)
    raw_auc = rows["raw"][2]
    rob_auc = rows["robust"][2]
    gain = rob_auc - raw_auc
    rf_auc = rf_test
    print(f"  IF + 原始输入test AUC        = {raw_auc:.4f}")
    print(f"  IF + RobustScaler test AUC   = {rob_auc:.4f}   (增益 {gain:+.4f})")
    print(f"  RF + RobustScaler test AUC   = {rf_auc:.4f}   (IF 仍差 {rf_auc - rob_auc:.4f})")
    print()
    if gain < 0.02 and (rf_auc - rob_auc) > 0.15:
        print("  => 归因：**输入归一化不是主因，IF 的表达力上限是主因**")
        print("     缩放几乎无增益；同样输入下有监督模型仍高 0.15+。")
        print("     修复路径只能是「增强 IF 的用法」或「引入有监督层」。")
    elif gain > 0.05:
        print("  => 归因：**输入量纲是主因**，IF 本身未必不行。")
        print("     修复路径：加 RobustScaler 即可，算法不动。")
    else:
        print("  => 归因：混合，需进一步拆分")


if __name__ == "__main__":
    main()
