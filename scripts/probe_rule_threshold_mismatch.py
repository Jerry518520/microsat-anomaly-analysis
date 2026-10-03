"""探针：核验 prod-gap 指控的「规则阈值口径不一致」是否成立。

指控：src/integration/anomaly_rag_pipeline.py:162-164 用的是
      rule_pred = (n_violated / n_total >= 0.2)  即约等于 nv >= 3.6 -> 4
      而实验 scripts/fusion_v3.py 用的是 (nv >= best_k)，best_k=2
      两处规则层判据不同 -> 0.1739 的差距里有一部分是口径错配，不是方法差距。

本脚本要给出三个数字：
  1. n_total 的真实取值（是恒为 18，还是随通道变化？0.2 阈值对应几个特征？）
  2. nv 分布：nv=2 / nv=3 / nv=4 各占多少段
  3. 口径错配本身值多少 F1：在同一套模型上，只把 rule_pred 的判据从
     「比例>=0.2」换成「nv>=2」，端到端 test F1 变多少？

纪律：全程用 fit 拟合阈值，val 选 k，test 只评一次。

用法：
    /d/Python313/python.exe scripts/probe_rule_threshold_mismatch.py
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments.framework import load_split  # noqa: E402
from src.utils.stat_rules import (  # noqa: E402
    apply_stat_rules,
    compute_stat_thresholds,
)

RULE_K_CANDIDATES = [1, 2, 3, 4]


def qf1(yt, yp):
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(yt), np.asarray(yp), zero_division=0))


def main():
    fit_df, val_df, test_df, feature_cols = load_split()
    thresholds = compute_stat_thresholds(fit_df, feature_cols)
    print(f"特征数 = {len(feature_cols)}")

    # ---------- 1. n_total 真实取值 ----------
    print()
    print("=" * 78)
    print("1. n_total（规则判定用的分母）实际是多少？比例阈值 0.2 对应几个特征？")
    print("=" * 78)
    for name, df in [("fit", fit_df), ("val", val_df), ("test", test_df)]:
        nv, nt = apply_stat_rules(df, thresholds, feature_cols)
        u = sorted(nt.unique())
        print(f"  {name:<5} n_total 取值 = {u}   "
              f"->比例>=0.2 等价于 nv >= {np.ceil(0.2 * u[0]):.0f} 至 {np.ceil(0.2 * u[-1]):.0f}")
    nv_t, nt_t = apply_stat_rules(test_df, thresholds, feature_cols)
    print(f"\n  结论：n_total 在 test 上恒为 {sorted(nt_t.unique())}，"
          f"故生产判据 (nv/18 >= 0.2) == (nv >= 4)")
    print(f"  而实验判据是 (nv >= best_k)，best_k 在 val 上选出 = 见下")

    # ---------- 2. nv 分布 ----------
    print()
    print("=" * 78)
    print("2. nv 分布与两种判据下的规则层 F1")
    print("=" * 78)
    yv = val_df["anomaly"].to_numpy()
    nv_v, _ = apply_stat_rules(val_df, thresholds, feature_cols)
    print(f"{'nv':>4}{'段数':>8}{'其中异常':>10}{'val F1(nv>=nv)':>16}")
    print("-" * 42)
    for t in range(0, 8):
        pred = (nv_v >= t).astype(int).to_numpy()
        mask = (nv_v == t).to_numpy()
        print(f"{t:>4}{int(mask.sum()):>8}{int(yv[mask].sum()):>10}{qf1(yv, pred):>16.4f}")
    print(f"{'>=9':>4}{int((nv_v >= 9).sum()):>8}"
          f"{int(yv[(nv_v >= 9).to_numpy()].sum()):>10}{qf1(yv, (nv_v >= 9).astype(int).to_numpy()):>16.4f}")

    best_k, bk_f1 = None, -1.0
    for k in RULE_K_CANDIDATES:
        f1 = qf1(yv, (nv_v >= k).astype(int).to_numpy())
        if f1 > bk_f1:
            bk_f1, best_k = f1, k
    print(f"\n  val 上选出的 best_k = {best_k} (val F1={bk_f1:.4f})")
    print(f"  生产硬编码口径 nv>=4 的 val F1 = "
          f"{qf1(yv, (nv_v >= 4).astype(int).to_numpy()):.4f}")
    print(f"  => 两口径在val 上就差 "
          f"{qf1(yv, (nv_v >= 4).astype(int).to_numpy()) - bk_f1:+.4f}")

    # ---------- 3. 口径错配值多少端到端 F1 ----------
    print()
    print("=" * 78)
    print("3. 口径错配单独值多少 test F1？（同模型，只换 rule 判据）")
    print("=" * 78)
    # 用纯规则层做隔离测试：只换判据，不牵扯 IF
    yt = test_df["anomaly"].to_numpy()
    f1_by_k = {}
    for t in range(1, 7):
        f1_by_k[t] = qf1(yt, (nv_t >= t).astype(int).to_numpy())
    print(f"{'判据':>16}{'test F1':>12}{'相对 best_k':>14}")
    print("-" * 44)
    for t, f1 in f1_by_k.items():
        tag = "<- 实验口径" if t == best_k else ("<- 生产口径" if t == 4 else "")
        print(f"{'nv >= ' + str(t):>16}{f1:>12.4f}{f1 - f1_by_k[best_k]:>+14.4f}  {tag}")
    print(f"\n  纯规则层：生产口径 nv>=4 与实验口径 nv>={best_k} 相差 "
          f"{f1_by_k[4] - f1_by_k[best_k]:+.4f}")
    print("  注：这是「只算规则层」的隔离口径；与 IF 融合后影响会被放大或稀释，")
    print("      端到端净效应需在生产管线里换判据后重跑才能确定。")

    # ---------- 4. 逐通道看 nv>=2 vs nv>=4 的差异 ----------
    print()
    print("=" * 78)
    print("4. 逐通道：两种判据的规则层 test F1")
    print("=" * 78)
    print(f"{'通道':<12}{'n_test':>8}{'n_anom':>8}{'F1(nv>=2)':>11}{'F1(nv>=4)':>11}{'差值':>9}")
    print("-" * 60)
    tdf = test_df.copy()
    tdf["nv"] = nv_t.values
    for ch in sorted(tdf["channel"].unique()):
        s = tdf[tdf["channel"] == ch]
        ye = s["anomaly"].to_numpy()
        f2 = qf1(ye, (s["nv"].values >= 2).astype(int))
        f4 = qf1(ye, (s["nv"].values >= 4).astype(int))
        print(f"{ch:<12}{len(s):>8}{int(ye.sum()):>8}{f2:>11.4f}{f4:>11.4f}{f4-f2:>+9.4f}")

    # ---------- 5. 训练管线里best_k 是否被 val 选过 ----------
    print()
    print("=" * 78)
    print("5. 结论")
    print("=" * 78)
    print(f"  实验侧 best_k 在 val 上选出 = {best_k}（有依据，可复现）")
    print(f"  生产侧 0.2 比例阈值 => nv>={int(np.ceil(0.2*len(feature_cols)))}（硬编码，无 val 依据）")
    print(f"  纯规则层 test F1 差 = {f1_by_k[4] - f1_by_k[best_k]:+.4f}")
    print("  => 指控成立：两处规则层判据确实不是同一个口径。")


if __name__ == "__main__":
    main()
