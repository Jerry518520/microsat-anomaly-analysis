"""
滑动窗口特征提取模块
核心创新：每个窗口 = 独立样本，标签继承 segment 标签

与段级特征的区别：
- 段级：每个 segment → 1 个特征向量（丢失时序），样本量 = segment 数
- 滑动窗口：每个 segment → N 个窗口特征向量（保留时序演变），样本量 = 窗口总数
  - 数据量从 2123 扩展到数万条
  - 每个窗口继承其所属 segment 的异常标签
  - 10 维特征（mean/std/min/max/median/skew/kurtosis/iqr/range/cv）
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, iqr
from src.utils.data_loader import load_segments, load_config, get_project_root


# 窗口级统计特征（10维）
STATS_FUNCS = {
    "mean": np.mean,
    "std": np.std,
    "min": np.min,
    "max": np.max,
    "median": np.median,
    "skew": lambda x: skew(x) if len(x) > 2 else 0.0,
    "kurtosis": lambda x: kurtosis(x) if len(x) > 3 else 0.0,
    "iqr": iqr,
    "range": lambda x: np.max(x) - np.min(x),
    "cv": lambda x: np.std(x) / (np.mean(x) + 1e-10),  # 变异系数
}

FEATURE_NAMES = list(STATS_FUNCS.keys())


def extract_window_features(values: np.ndarray, window_size: int = 50,
                            step_size: int = 10) -> pd.DataFrame:
    """
    对单个 segment 的遥测序列提取滑动窗口特征

    Args:
        values: 该 segment 的 value 序列 (1D array)
        window_size: 窗口长度（采样点数）
        step_size: 步长

    Returns:
        DataFrame, 每行 = 一个窗口的 10 维特征
        行数 = (len(values) - window_size) // step_size + 1
    """
    records = []
    for start in range(0, len(values) - window_size + 1, step_size):
        window = values[start:start + window_size]
        feat = {}
        for name, func in STATS_FUNCS.items():
            try:
                feat[name] = float(func(window))
            except Exception:
                feat[name] = 0.0
        records.append(feat)

    return pd.DataFrame(records) if records else pd.DataFrame(columns=FEATURE_NAMES)


def extract_all_sliding_features(segments_df: pd.DataFrame,
                                  window_size: int = 50,
                                  step_size: int = 10) -> pd.DataFrame:
    """
    对所有 segment 提取滑动窗口特征（每个窗口 = 独立样本）

    关键设计：
    - 每个窗口继承其所属 segment 的 anomaly 标签
    - 每个窗口继承 channel / train / sampling 元数据
    - 不做二次聚合，保留完整的窗口级时序信息

    Args:
        segments_df: segments.csv 的 DataFrame
        window_size: 窗口长度
        step_size: 步长

    Returns:
        DataFrame, 每行 = 一个窗口样本, 列 = 10维特征 + 元数据
        数据量: 从 2123 segments 扩展到数万窗口
    """
    all_records = []
    segments = segments_df["segment"].unique()
    skipped = 0

    print(f"[SlidingWindow] Processing {len(segments)} segments (window={window_size}, step={step_size})...")

    for idx, seg_id in enumerate(sorted(segments)):
        seg_data = segments_df[segments_df["segment"] == seg_id]
        values = seg_data["value"].values

        # 跳过短于窗口长度的 segment
        if len(values) < window_size:
            skipped += 1
            continue

        # 提取窗口特征
        window_feats = extract_window_features(values, window_size, step_size)

        if len(window_feats) == 0:
            skipped += 1
            continue

        # 元数据（继承自 segment）
        channel = seg_data["channel"].iloc[0]
        anomaly = seg_data["anomaly"].iloc[0]
        train = seg_data["train"].iloc[0]
        sampling = seg_data["sampling"].iloc[0]

        # 为每个窗口添加元数据
        window_feats["channel"] = channel
        window_feats["segment"] = seg_id
        window_feats["anomaly"] = anomaly  # 标签继承
        window_feats["train"] = train
        window_feats["sampling"] = sampling

        all_records.append(window_feats)

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(segments)} segments...")

    result_df = pd.concat(all_records, ignore_index=True)

    # 统计
    n_train = (result_df["train"] == 1).sum()
    n_test = (result_df["train"] == 0).sum()
    n_anomaly = result_df["anomaly"].sum()
    n_normal = len(result_df) - n_anomaly

    print(f"[SlidingWindow] Done.")
    print(f"  Total: {len(result_df)} window samples × {len(FEATURE_NAMES)} features")
    print(f"  Skipped: {skipped} segments (shorter than window_size={window_size})")
    print(f"  Train: {n_train}, Test: {n_test}")
    print(f"  Normal: {n_normal}, Anomaly: {n_anomaly} ({n_anomaly/len(result_df)*100:.1f}%)")

    return result_df


def save_sliding_features(features_df: pd.DataFrame, config: dict = None):
    """保存滑动窗口特征到 data/processed/"""
    if config is None:
        config = load_config()
    root = get_project_root()
    out_dir = os.path.join(root, config["data"]["processed_dir"])
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "sliding_window_features.csv")
    features_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[SlidingWindow] Saved: {out_path}")
    return out_path


def run_sliding_window_pipeline(config: dict = None):
    """完整的滑动窗口特征提取流程"""
    if config is None:
        config = load_config()

    sw_config = config.get("sliding_window", {})
    window_size = sw_config.get("window_size", 50)
    step_size = sw_config.get("step_size", 10)

    segments_df = load_segments(config)
    features_df = extract_all_sliding_features(
        segments_df, window_size=window_size, step_size=step_size
    )
    save_sliding_features(features_df, config)

    return features_df


if __name__ == "__main__":
    features_df = run_sliding_window_pipeline()
    print(f"\n特征列: {FEATURE_NAMES}")
    print(f"元数据列: channel, segment, anomaly, train, sampling")
