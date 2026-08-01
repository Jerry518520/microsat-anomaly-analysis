"""
段级 18 维特征提取 — 与官方 dataset_generator.ipynb bug-for-bug 一致

供流式引擎在段闭合(及增长段预览)时调用。与离线训练特征同源:
- duration / gaps_squared 保留官方 .seconds 分量语义(对 86400 取模,
  丢弃天数位)——这是官方实现的固有行为,训练特征亦如此,必须一致;
- 13 个统计变换直接复用 src/features/extract_segments_18d.TRANSFORMATIONS;
- 衍生特征 len_weighted / var_div_duration / var_div_len 公式同官方。

列序与 ipynb 的 feature_cols(segments_18d.csv 去掉 5 个 META_COLS)一致:
duration 在前, len_weighted/var_div_duration/var_div_len 在后。
"""

import warnings

import numpy as np

from src.features.extract_segments_18d import TRANSFORMATIONS

FEATURE_COLS = [
    "duration", "len", "mean", "var", "std", "kurtosis", "skew",
    "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks",
    "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
    "gaps_squared", "len_weighted", "var_div_duration", "var_div_len",
]


def _seconds_component(delta_s: float) -> int:
    """复现 pandas Timedelta.seconds:总秒数的"秒分量"(丢弃天数位)"""
    return int(round(delta_s)) % 86400


def extract_segment_features(values: np.ndarray, timestamps: np.ndarray,
                             sampling: int = 1) -> dict[str, float]:
    """
    对一个(完整或增长中的)段计算 18 维特征。

    Args:
        values: 段内遥测值, shape (N,)
        timestamps: 段内 unix 秒时间戳, shape (N,)
        sampling: 采样间隔(秒), 用于 len_weighted

    Returns:
        dict, 键 = FEATURE_COLS 中的 18 个特征名
    """
    n = len(values)
    feats: dict[str, float] = {}

    feats["duration"] = float(_seconds_component(timestamps[-1] - timestamps[0])) if n > 1 else 0.0

    if n > 2:
        with warnings.catch_warnings():
            # 短前缀上 kurtosis/skew/diff2_* 会产生精度告警,与离线行为一致(nan)
            warnings.simplefilter("ignore")
            for name, fn in TRANSFORMATIONS.items():
                feats[name] = float(fn(values))
    else:
        for name in TRANSFORMATIONS:
            feats[name] = float("nan")

    if n > 1:
        deltas = np.diff(timestamps)
        feats["gaps_squared"] = float(sum(_seconds_component(d) ** 2 for d in deltas))
    else:
        feats["gaps_squared"] = 0.0

    feats["len_weighted"] = float(sampling) * feats["len"]
    feats["var_div_duration"] = feats["var"] / feats["duration"] if feats["duration"] > 0 else 0.0
    feats["var_div_len"] = feats["var"] / feats["len"] if feats["len"] > 0 else 0.0

    return feats


def feature_vector(feats: dict[str, float]) -> np.ndarray:
    """dict → shape (18,) 向量, 列序 = FEATURE_COLS"""
    return np.array([feats[c] for c in FEATURE_COLS], dtype=float)
