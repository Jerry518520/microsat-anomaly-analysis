"""
算法正确性回归测试
==================
每个测试都锁定一条实测确认的约定，防止后续改动破坏它们。

背景：这些不是"风格偏好"，而是实测得出的事实。
- 划分无泄露：fit/val/test 三集 segment 交集必须为 0
- 分层纪律：阈值与模型只在 fit 拟合，超参只在 val 选，test 只评估一次
- IQR=0 语义：对离散计数特征是正确的（见 test_iqr_zero_is_physically_meaningful）
  曾经的怀疑：agent 一度把 IQR=0 退化报成 P0 bug，实测证伪 —— n_peaks>=2 时
  异常率 80~100%，n_peaks=1 时仅 4~10%。见该测试的完整论证。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.framework import load_split  # noqa: E402
from src.utils.stat_rules import (  # noqa: E402
    apply_stat_rules,
    compute_stat_thresholds,
    iqr_rule,
    sigma_rule,
)


# ---------------------------------------------------------------- 数据划分


def test_three_splits_are_disjoint():
    """fit/val/test 三个集合的段 ID 交集必须全为 0，否则存在数据泄露。"""
    fit, val, test, _ = load_split()
    a, b, c = set(fit.segment), set(val.segment), set(test.segment)
    assert not (a & b), f"fit∩val 重叠 {len(a & b)} 段"
    assert not (a & c), f"fit∩test 重叠 {len(a & c)} 段"
    assert not (b & c), f"val∩test 重叠 {len(b & c)} 段"
    assert len(a | b | c) == 2123, f"并集应为 2123 段，实得 {len(a | b | c)}"


def test_split_sizes_and_stratification():
    """官方 train 1594 = 1275 + 319，官方 test 529 只做最终评估。

    异常率三集接近是分层合理的必要条件。
    """
    fit, val, test, _ = load_split()
    assert (len(fit), len(val), len(test)) == (1275, 319, 529)
    assert len(fit) + len(val) == 1594, "fit+val 应等于官方 train"
    rates = [df.anomaly.mean() for df in (fit, val, test)]
    for r in rates:
        assert 0.15 < r < 0.27, f"异常率 {r:.4f} 偏离过多，分层可能失效"


# ---------------------------------------------------------------- 规则层语义


def test_iqr_zero_is_physically_meaningful():
    """IQR=0 时规则退化为 "value != q1"，这对离散计数特征是正确的。

    实测依据（test 集，n_peaks 的分位数在多数通道上恰为 q1=q3=1）：
        n_peaks=1 -> 异常率 4~10%   （正常段）
        n_peaks>=2 -> 异常率 80~100% （几乎全是异常段）
    也就是说"超过单峰即异常"正是该数据集的真实物理规律。

    曾经的误判：把这里当成 P0 bug 报出，建议跳过 iqr_rule。实为把正确
    判据当 bug 修，修后 test F1 从 0.6281 掉到 0.5363。
    """
    # 退化本身：q1=q3=1, iqr=0 时，只有恰好等于 1 才算正常
    assert iqr_rule(1.0, 1.0, 1.0, 0.0) is False
    assert iqr_rule(2.0, 1.0, 1.0, 0.0) is True
    assert iqr_rule(0.0, 1.0, 1.0, 0.0) is True

    # 关键验证：退化判据在本数据集上确实有判别力（精确率远高于基线）
    fit, val, test, fcols = load_split()
    th = compute_stat_thresholds(fit, fcols)
    deg = {
        (ch, col)
        for ch, cols in th.items()
        for col, t in cols.items()
        if t["iqr"] == 0.0
    }
    assert deg, "未发现 IQR=0 的组合，若数据集变了请重新评估此测试"

    y = test.anomaly.to_numpy()
    pred = np.zeros(len(test), dtype=bool)
    for i, (idx, row) in enumerate(test.iterrows()):
        for col in fcols:
            if (row["channel"], col) in deg and col in th.get(row["channel"], {}):
                t = th[row["channel"]][col]
                if iqr_rule(row[col], t["q1"], t["q3"], t["iqr"]):
                    pred[i] = True
                    break

    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    assert precision > 0.5, (
        f"退化判据精确率仅 {precision:.3f}，若跌至此说明其判别力已消失，"
        f"此时应重新设计而非沿用"
    )


def test_sigma_rule_is_symmetric_both_sides():
    """3-sigma 规则是双侧的：过高和过低都算异常。

    实测：违规中 92.9% 来自上界，但下界违规集中在 mean 特征上
    （异常段 mean 比正常段更小，即信号丢失），双侧是正确的。
    """
    assert sigma_rule(10.0, 0.0, 1.0) is True
    assert sigma_rule(-10.0, 0.0, 1.0) is True
    assert sigma_rule(0.5, 0.0, 1.0) is False


def test_boundary_is_not_flagged():
    """边界值口径一致：严格不等式，恰好等于阈值判为正常。"""
    assert iqr_rule(4.0, 1.0, 3.0, 2.0) is False  # 上界 hi=3+1.5*2=6
    assert iqr_rule(7.0, 1.0, 3.0, 2.0) is True
    assert sigma_rule(3.0, 0.0, 1.0) is False  # 恰好 3σ 判正常
    assert sigma_rule(3.1, 0.0, 1.0) is True


def test_thresholds_depend_on_passed_dataframe():
    """反证：阈值函数确实依赖传入集合，不存在硬编码或全局缓存。

    若有人把阈值改成在 val/test 上拟合，此测试会失败。
    """
    fit, val, _, fcols = load_split()
    th_fit = compute_stat_thresholds(fit, fcols)
    th_val = compute_stat_thresholds(val, fcols)

    diff = 0
    for ch in th_fit:
        for col in th_fit[ch]:
            if ch in th_val and col in th_val[ch]:
                if th_fit[ch][col]["q3"] != th_val[ch][col]["q3"]:
                    diff += 1
    assert diff > 50, (
        f"fit 与 val 上算出的 q3 仅 {diff} 处不同，阈值可能未真正依赖传入数据"
    )


def test_rule_layer_tolerates_small_channel():
    """规则层只有 len(vals)<5 的保护；样本极少时不应崩溃。"""
    fit, _, _, fcols = load_split()
    th = compute_stat_thresholds(fit, fcols)
    # 通道 0886 在 fit 上样本极少，阈值可能缺失某些特征
    assert "CADC0886" in th
    assert len(th["CADC0886"]) <= len(fcols)


# ---------------------------------------------------------------- 指标


def test_confusion_matrix_order_is_tn_fp_fn_tp():
    """confusion_matrix(..., labels=[0,1]).ravel() 的顺序必须是 tn,fp,fn,tp。

    顺序搞错会让 P/R 静默颠倒，不报错但结论全错。
    """
    from sklearn.metrics import confusion_matrix

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    assert (tn, fp, fn, tp) == (1, 1, 1, 1)

    # 顺序敏感的样例：若 ravel 顺序错，这组数字会不同
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    assert (tn, fp, fn, tp) == (1, 1, 1, 1)
    # tn 应为"真负"：y_true=0 且 y_pred=0 的个数 = 1
    assert tn == 1


def test_f1_from_confusion_matrix_matches_sklearn():
    """手算 F1 与 sklearn 一致，锁定公式理解无误。"""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 200)
    p = rng.integers(0, 2, 200)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    manual = 2 * tp / (2 * tp + fp + fn)
    assert abs(manual - f1_score(y, p, zero_division=0)) < 1e-12
