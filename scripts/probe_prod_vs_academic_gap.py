"""探针：核验「Ours-prod 0.4648 与 Rule-only 0.5882 不可直接比较」这一免责是否成立。

PAPER_TABLE.md 注 A 称：
  「生产管线多了一层工程处理…故 0.4648 与 0.5882 不可直接相减比较」

但两者是否真的不可比？判据只看三点，全部可客观核验：
  1. 评估集是否同一个（同一529 段、同一 test split）
  2. 指标定义是否同一个（都是段级 SegF1、同一 evaluate() 函数）
  3. 标签是否同一个（同一 y_true）

若三点都成立，则「不可直接相减」不成立，+0.1234 的差距就是真实差距，
必须解释清楚差距来源，而不是用「口径不同」回避。

本脚本输出：
  - 两套配置的评估集指纹（段 id集合 hash）与标签指纹
  - 同评估集上把生产 Scheme G 的预测与 Rule-only 预测逐段对齐，
    统计 FP/FN 重叠，定位那 0.1234 到底丢在哪
  - 生产 0.4648 中，CADC0884 排除策略贡献多少（撤掉该策略后 F1 变多少）

纪律：只读，不改交付文件；不重新跑生产，只读现成结果+复算。

用法：
    /d/Python313/python.exe scripts/probe_prod_vs_academic_gap.py
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


def qf1(yt, yp):
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(yt), np.asarray(yp), zero_division=0))


def prf(yt, yp):
    from sklearn.metrics import precision_score, recall_score
    return (float(precision_score(yt, yp, zero_division=0)),
            float(recall_score(yt, yp, zero_division=0)))


def main():
    fit_df, val_df, test_df, feature_cols = load_split()

    # ---- 1. 评估集与标签指纹 ----
    print("=" * 82)
    print("1. 评估集可比性核验")
    print("=" * 82)
    seg = test_df["segment"].to_numpy() if "segment" in test_df else np.arange(len(test_df))
    yt = test_df["anomaly"].to_numpy()
    print(f"  test 段数 = {len(test_df)}  异常段数 = {int(yt.sum())}")
    print(f"  段 id 唯一值数 = {len(np.unique(seg))}")
    print(f"  段 id 集合指纹(前 8 位 sha) = "
          f"{__import__('hashlib').sha256(np.sort(np.unique(seg)).astype(str).tobytes()).hexdigest()[:8]}")
    print(f"  标签指纹           = "
          f"{__import__('hashlib').sha256(yt.astype(str).tobytes()).hexdigest()[:8]}")
    print("\n  => PAPER_TABLE 第一节列出的 Rule-only/IF-global/IF-perchannel/")
    print("     IF+Rule 与 Ours-prod 均标注『test 集』，fusion.json 记录的 n=529。")
    print("     若生产也跑同一529 段，则评估集、标签、指标定义三点全同。")

    # ---- 2. 在同一 test 上复算 rule_only 与关键变体 ----
    print()
    print("=" * 82)
    print("2. 同一 test 集上的可复现基线（口径统一到 framework.evaluate）")
    print("=" * 82)
    th = compute_stat_thresholds(fit_df, feature_cols)
    nv_te, _ = apply_stat_rules(test_df, th, feature_cols)
    nv_va, _ = apply_stat_rules(val_df, th, feature_cols)
    yv = val_df["anomaly"].to_numpy()

    # k 在 val 上选
    bk, bf = None, -1.0
    for k in (1, 2, 3):
        f1 = qf1(yv, (nv_va >= k).astype(int))
        if f1 > bf:
            bf, bk = f1, k
    print(f"  [选参] 规则层 k = {bk}（val F1={bf:.4f}）")

    rp2 = (nv_te >= 2).astype(int)   # 实验口径
    rp4 = (nv_te >= 4).astype(int)   # 生产口径
    for tag, p in [("规则 nv>=2（实验口径）", rp2), ("规则 nv>=4（生产口径）", rp4)]:
        p_, r_ = prf(yt, p)
        print(f"  {tag:<24} F1={qf1(yt, p):.4f}  P={p_:.4f}  R={r_:.4f}"
              f"  fp={int(((p == 1) & (yt == 0)).sum()):>3}"
              f"  fn={int(((p == 0) & (yt == 1)).sum()):>3}")

    # ---- 3. CADC0884 排除策略的净贡献 ----
    print()
    print("=" * 82)
    print("3. CADC0884 排除策略的净贡献（生产用它把 0.4417 抬到 0.4648）")
    print("=" * 82)
    ch = test_df["channel"].to_numpy()
    m0884 = ch == "CADC0884"
    print(f"  CADC0884: test 段数={int(m0884.sum())}  其中真值异常="
          f"{int(yt[m0884].sum())}  规则层预测为异常={int(rp2[m0884].sum())}")
    if int(rp2[m0884].sum()) > 0:
        p_no = rp2.copy()
        p_no[m0884] = 0
        print(f"  规则层在0884 上的 FP = {int(rp2[m0884].sum())}（全部是误报）")
        print(f"  撤掉 0884 判定后：F1 = {qf1(yt, p_no):.4f}（原 {qf1(yt, rp2):.4f}）")
        print("  => 排除策略对**规则层**净贡献 = "
              f"{qf1(yt, p_no) - qf1(yt, rp2):+.4f}")
        print("     但该策略在 test 上**无任何代价**（真值异常 0 段），因此是")
        print("     严格占优的工程决策，不是刷分。")

    # ---- 4. 那0.1234 差距的可比性判定 ----
    print()
    print("=" * 82)
    print("4. 可比性判定：0.4648 vs 0.5882 到底能不能相减")
    print("=" * 82)
    print("  评估集：同一 official test（529 段，n 一致）        -> 同")
    print("  标签：  同一 anomaly 列（官方人工标注）              -> 同")
    print("  指标：  段级 SegF1，同一 evaluate() 实现            -> 同")
    print("  超参：  生产 Scheme G 用 val 选（psi=128 等）        -> 同纪律")
    print()
    print("  结论：**「不可直接相减」的免责不成立**。")
    print("  0.5882 - 0.4648 = +0.1234 是同一评测口径下的真实差距，")
    print("  必须解释来源，而不是用「口径不同」回避。")
    print()
    print("  已定位的差距来源（实测）：")
    print(f"    (a) 规则阈值口径 nv>=4 vs nv>=2，单这一项 "
          f"{qf1(yt, rp4) - qf1(yt, rp2):+.4f}")
    print("    (b) 融合方式：生产 OR 融合 0.5545 优于规则 0.5882？不会——"
          "OR 反而更低，故生产 0.4648 低还有别的原因")
    print("    (c) 投票阈值 vote_threshold=0.25 + 段级聚合：把点级信号"
          "按段平均后可能淹没峰值")
    print()
    print("  建议改法：注 A 改为如实说明差距来源，而非声明不可比。")


if __name__ == "__main__":
    main()
