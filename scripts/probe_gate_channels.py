"""探针：复核 PAPER_TABLE.md 4.2 表中逐通道门控的样本量与算子选择。

只读，不改任何交付文件。目的：
  1. 打印 fit/val/test 的逐通道段数与异常段数，核实表中 n_anomaly 声明；
  2. 用与 fusion_v3.py 完全相同的口径重跑门控算子选择，逐位对照表中的 val F1；
  3. 复算 CADC0884 / 0886 / 0890 三通道 val 是否真的过小（结论应写进论文）。

用法：
    /d/Python313/python.exe scripts/probe_gate_channels.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score as _f1

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


def quick_f1(yt, yp):
    return float(_f1(np.asarray(yt), np.asarray(yp), zero_division=0))


def main():
    fit_df, val_df, test_df, feature_cols = load_split()

    print("=" * 78)
    print("A. 逐通道样本量（段级）")
    print("=" * 78)
    hdr = f"{'channel':<12}{'fit_n':>7}{'fit_anom':>10}{'val_n':>7}{'val_anom':>10}{'test_n':>8}{'test_anom':>11}"
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for ch in sorted(fit_df["channel"].unique()):
        f = fit_df[fit_df["channel"] == ch]
        v = val_df[val_df["channel"] == ch]
        t = test_df[test_df["channel"] == ch]
        rows[ch] = (len(f), int(f["anomaly"].sum()),
                    len(v), int(v["anomaly"].sum()),
                    len(t), int(t["anomaly"].sum()))
        print(f"{ch:<12}{rows[ch][0]:>7}{rows[ch][1]:>10}{rows[ch][2]:>7}"
              f"{rows[ch][3]:>10}{rows[ch][4]:>8}{rows[ch][5]:>11}")
    print(f"{'TOTAL':<12}{len(fit_df):>7}{int(fit_df['anomaly'].sum()):>10}"
          f"{len(val_df):>7}{int(val_df['anomaly'].sum()):>10}"
          f"{len(test_df):>8}{int(test_df['anomaly'].sum()):>11}")

    print()
    print("=" * 78)
    print("B. 门控算子选择复算（口径与 fusion_v3.py 一致）")
    print("=" * 78)
    thresholds = compute_stat_thresholds(fit_df, feature_cols)

    # 全局 best_k
    best_k, best_f1 = RULE_K_CANDIDATES[0], -1.0
    for k in RULE_K_CANDIDATES:
        nv, _ = apply_stat_rules(val_df, thresholds, feature_cols)
        f1 = quick_f1(val_df["anomaly"].to_numpy(), (nv >= k).astype(int).to_numpy())
        if f1 > best_f1:
            best_f1, best_k = f1, k
    print(f"[global] best_k = {best_k} (val F1={best_f1:.4f})")

    # 逐通道 best_c（在 val 上选）
    best_c = {}
    for ch in sorted(val_df["channel"].unique()):
        ch_val = val_df[val_df["channel"] == ch]
        ch_fit = fit_df[fit_df["channel"] == ch]
        if len(ch_fit) < MIN_FIT or len(ch_val) < MIN_EVAL:
            best_c[ch] = None
            continue
        y_val = ch_val["anomaly"].to_numpy()
        X_fit = np.nan_to_num(ch_fit[feature_cols].values)
        X_val = np.nan_to_num(ch_val[feature_cols].values)
        bc, bf = CONTAM_GRID[0], -1.0
        for c in CONTAM_GRID:
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=42, n_jobs=-1).fit(X_fit)
            f1 = quick_f1(y_val, (clf.predict(X_val) == -1).astype(int))
            if f1 > bf:
                bf, bc = f1, c
        best_c[ch] = float(bc)

    print()
    hdr2 = (f"{'channel':<12}{'op':>6}{'val_f1':>9}{'val_n':>7}{'val_an':>8}"
            f"{'n_anom<5?':>10}  备注")
    print(hdr2)
    print("-" * (len(hdr2) + 12))
    picks = {}
    for ch in sorted(val_df["channel"].unique()):
        ch_val = val_df[val_df["channel"] == ch]
        ch_fit = fit_df[fit_df["channel"] == ch]
        y_val = ch_val["anomaly"].to_numpy()
        n_an = int(y_val.sum())

        nv_v, _ = apply_stat_rules(ch_val, thresholds, feature_cols)
        rule_pred = (nv_v >= best_k).astype(int).to_numpy()
        rule_f1 = quick_f1(y_val, rule_pred)

        c = best_c.get(ch)
        if c is None or len(ch_fit) < MIN_FIT:
            if_pred = np.zeros(len(ch_val), dtype=int)
            if_f1 = 0.0
        else:
            clf = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                                  contamination=c, random_state=42, n_jobs=-1
                                  ).fit(np.nan_to_num(ch_fit[feature_cols].values))
            if_pred = (clf.predict(np.nan_to_num(ch_val[feature_cols].values)) == -1).astype(int)
            if_f1 = quick_f1(y_val, if_pred)

        cands = {"rule": rule_f1, "if": if_f1,
                 "AND": quick_f1(y_val, (rule_pred & if_pred).astype(int)),
                 "OR": quick_f1(y_val, (rule_pred | if_pred).astype(int))}
        op = max(cands, key=lambda k: cands[k])
        picks[ch] = (op, cands[op], cands)

        flag = "YES" if n_an < 5 else ""
        note = []
        if n_an == 0:
            note.append("val无异常段->选算子无意义")
        elif n_an < 5:
            note.append("val异常段过少->F1不可靠")
        if len(ch_val) < 30:
            note.append(f"val总段数{len(ch_val)}<30")
        print(f"{ch:<12}{op:>6}{cands[op]:>9.4f}{len(ch_val):>7}{n_an:>8}"
              f"{flag:>10}  {'; '.join(note)}")

    print()
    print("四算子 val F1 全量对照：")
    for ch in sorted(picks):
        c = picks[ch][2]
        print(f"  {ch}: " + "  ".join(f"{k}={v:.4f}" for k, v in c.items())
              + f"   -> 选 {picks[ch][0]}")

    print()
    print("=" * 78)
    print("C. 小样本通道的 train+val 合并异常段数（更宽松口径下是否仍过小）")
    print("=" * 78)
    for ch in sorted(picks):
        n_an_val = rows[ch][3]
        n_an_both = rows[ch][1] + rows[ch][3]
        if n_an_val < 5:
            print(f"  {ch}: val_anom={n_an_val}, fit_anom={rows[ch][1]}, "
                  f"fit+val_anom={n_an_both}, val_n={rows[ch][2]}")


if __name__ == "__main__":
    main()
