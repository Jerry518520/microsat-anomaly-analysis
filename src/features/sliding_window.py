"""
滑动窗口特征提取模块
核心创新：将段级统计特征扩展为滑动窗口时序特征，保留时序信息

与段级特征的区别：
- 段级：每个 segment → 1 个特征向量（丢失时序）
- 滑动窗口：每个 segment → N 个特征向量（保留时序演变）
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, iqr
from src.utils.data_loader import load_segments, load_config, get_project_root


def extract_window_features(series: np.ndarray, window_size: int = 50,
                            step_size: int = 10) -> pd.DataFrame:
    """
    对单个 segment 的遥测序列提取滑动窗口特征

    Args:
        series: 该 segment 的 value 序列
        window_size: 窗口长度（采样点数）
        step_size: 步长

    Returns:
        DataFrame, 每行是一个窗口的特征
    """
    stats_funcs = {
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

    features = []
    for start in range(0, len(series) - window_size + 1, step_size):
        window = series[start:start + window_size]
        feat = {}
        for name, func in stats_funcs.items():
            try:
                feat[name] = func(window)
            except Exception:
                feat[name] = 0.0
        # 额外特征：窗口位置（归一化）
        feat["window_position"] = start / max(len(series) - window_size, 1)
        features.append(feat)

    return pd.DataFrame(features)


def extract_all_sliding_features(segments_df: pd.DataFrame,
                                  window_size: int = 50,
                                  step_size: int = 10) -> pd.DataFrame:
    """
    对所有 segment 提取滑动窗口特征

    Args:
        segments_df: segments.csv 的 DataFrame
        window_size: 窗口长度
        step_size: 步长

    Returns:
        DataFrame, 包含每个 segment 的聚合滑动窗口特征 + 元数据
    """
    all_records = []
    segments = segments_df["segment"].unique()

    print(f"[SlidingWindow] Processing {len(segments)} segments...")

    for idx, seg_id in enumerate(sorted(segments)):
        seg_data = segments_df[segments_df["segment"] == seg_id]
        values = seg_data["value"].values

        # 提取滑动窗口特征
        window_feats = extract_window_features(values, window_size, step_size)

        if len(window_feats) == 0:
            continue

        # 聚合策略：对每个窗口统计量再做段级聚合（mean, std, min, max）
        agg_funcs = ["mean", "std", "min", "max"]
        aggregated = window_feats.agg(agg_funcs)

        record = {}
        for col in window_feats.columns:
            for agg in agg_funcs:
                record[f"sw_{col}_{agg}"] = aggregated.loc[agg, col]

        # 额外特征：窗口数量（反映 segment 长度）
        record["sw_n_windows"] = len(window_feats)

        # 元数据
        record["channel"] = seg_data["channel"].iloc[0]
        record["segment"] = seg_id
        record["anomaly"] = seg_data["anomaly"].iloc[0]
        record["train"] = seg_data["train"].iloc[0]
        record["sampling"] = seg_data["sampling"].iloc[0]

        all_records.append(record)

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(segments)} segments...")

    result_df = pd.DataFrame(all_records)
    print(f"[SlidingWindow] Done. {result_df.shape[0]} segments × {result_df.shape[1]} features")
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
    print(f"\n特征列: {[c for c in features_df.columns if c.startswith('sw_')]}")
