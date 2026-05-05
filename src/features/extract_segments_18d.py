"""
从 segments.csv 原始遥测数据中提取 18 维段级特征
完全复现 OPS-SAT-AD 官方 dataset_generator.ipynb 的计算逻辑

输出: data/processed/segments_18d.csv (2123行 × 23列)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy.stats import kurtosis, skew

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


# ===== 特征计算函数（与 dataset_generator.ipynb 完全一致）=====

def number_of_peaks_finding(array):
    """峰值数，prominence = 0.1 * (max - min)"""
    prominence = 0.1 * (np.max(array) - np.min(array))
    peaks = sig.find_peaks(array, prominence=prominence)[0]
    return len(peaks)


def duration(df):
    """段持续时间（秒），取首尾时间戳之差"""
    t1 = pd.Timestamp(df.head(1).timestamp.values[0])
    t2 = pd.Timestamp(df.tail(1).timestamp.values[0])
    return (t2 - t1).seconds


def smooth10_n_peaks(array):
    """10点平滑后的峰值数"""
    kernel = np.ones(10) / 10
    array_convolved = np.convolve(array, kernel, mode="same")
    return number_of_peaks_finding(array_convolved)


def smooth20_n_peaks(array):
    """20点平滑后的峰值数"""
    kernel = np.ones(20) / 20
    array_convolved = np.convolve(array, kernel, mode="same")
    return number_of_peaks_finding(array_convolved)


def diff_peaks(array):
    """一阶差分的峰值数"""
    return number_of_peaks_finding(np.diff(array))


def diff2_peaks(array):
    """二阶差分的峰值数"""
    return number_of_peaks_finding(np.diff(array, n=2))


def diff_var(array):
    """一阶差分方差"""
    return np.var(np.diff(array))


def diff2_var(array):
    """二阶差分方差"""
    return np.var(np.diff(array, n=2))


def gaps_squared(df):
    """相邻时间戳秒数差的平方和"""
    df = df.copy()
    df['timestamp2'] = df['timestamp'].shift(1)
    df = df.reset_index().iloc[1:, :]
    df['time_delta'] = (df.timestamp - df.timestamp2).dt.seconds
    df['time_delta_squared'] = df['time_delta'] ** 2
    return df.time_delta_squared.sum()


# 与 notebook 中 transformations 字典保持一致的特征列表
TRANSFORMATIONS = {
    "len": len,
    "mean": np.mean,
    "var": np.var,
    "std": np.std,
    "kurtosis": kurtosis,
    "skew": skew,
    "n_peaks": number_of_peaks_finding,
    "smooth10_n_peaks": smooth10_n_peaks,
    "smooth20_n_peaks": smooth20_n_peaks,
    "diff_peaks": diff_peaks,
    "diff2_peaks": diff2_peaks,
    "diff_var": diff_var,
    "diff2_var": diff2_var,
}


def generate_dataset(source_df):
    """
    从 segments 原始数据生成 18 维段级特征
    与 dataset_generator.ipynb 的 generate_dataset 函数逻辑完全一致
    """
    dataset = []
    segments = source_df.segment.unique()

    print(f"[Extract18D] Processing {len(segments)} segments...")

    for i, seg_id in enumerate(sorted(segments)):
        tdf = source_df.loc[source_df.segment == seg_id, :]
        res = []

        # 元数据
        res.append(seg_id)                                      # segment
        res.append(int(tdf.loc[:, "anomaly"].head(1).values[0]))  # anomaly
        res.append(int(tdf.loc[:, "train"].head(1).values[0]))    # train
        res.append(tdf.loc[:, "channel"].head(1).values[0])       # channel
        res.append(int(tdf.loc[:, "sampling"].head(1).values[0])) # sampling
        res.append(duration(tdf))                                 # duration

        # 13个统计特征（按 transformations 字典顺序）
        values = tdf.value.values
        for transformation in TRANSFORMATIONS.values():
            res.append(transformation(values))

        # gaps_squared（需要timestamp列）
        res.append(gaps_squared(tdf))

        dataset.append(res)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(segments)} segments...")

    # 构建 DataFrame
    columns = (["segment", "anomaly", "train", "channel", "sampling", "duration"]
               + list(TRANSFORMATIONS.keys()) + ["gaps_squared"])
    df = pd.DataFrame(data=dataset, columns=columns)

    # 衍生特征（与 notebook 完全一致）
    df["len_weighted"] = df["sampling"] * df["len"]
    df["var_div_duration"] = df["var"] / df["duration"]
    df["var_div_len"] = df["var"] / df["len"]

    # 数据类型对齐
    int_cols = ["segment", "anomaly", "train", "sampling", "duration", "len",
                "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks",
                "diff_peaks", "diff2_peaks", "gaps_squared", "len_weighted"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    print(f"[Extract18D] Done. Shape: {df.shape}")
    return df


def validate_against_dataset(generated_df, dataset_csv_path):
    """
    与官方 dataset.csv 做交叉验证
    抽样10个segment，对比18个特征值
    """
    official_df = pd.read_csv(dataset_csv_path)

    # 共同 segment
    common_segs = set(generated_df.segment) & set(official_df.segment)
    print(f"\n[Validation] 共同 segment 数: {len(common_segs)}")

    # 随机抽样10个
    np.random.seed(42)
    sample_segs = sorted(np.random.choice(list(common_segs), size=min(10, len(common_segs)), replace=False))

    feature_cols = ["duration", "len", "mean", "var", "std", "kurtosis", "skew",
                    "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks",
                    "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
                    "gaps_squared", "len_weighted", "var_div_duration", "var_div_len"]

    total_diff = 0
    max_diff = 0
    mismatches = []

    for seg_id in sample_segs:
        gen_row = generated_df[generated_df.segment == seg_id].iloc[0]
        off_row = official_df[official_df.segment == seg_id].iloc[0]

        for col in feature_cols:
            g_val = float(gen_row[col])
            o_val = float(off_row[col])
            diff = abs(g_val - o_val)
            total_diff += diff
            if diff > max_diff:
                max_diff = diff
            if diff > 1e-6:
                mismatches.append(f"  segment={seg_id}, {col}: generated={g_val:.10f}, official={o_val:.10f}, diff={diff:.2e}")

    print(f"[Validation] 抽样 {len(sample_segs)} 段 × {len(feature_cols)} 特征 = {len(sample_segs)*len(feature_cols)} 次比较")
    print(f"[Validation] 最大绝对差异: {max_diff:.2e}")

    if mismatches:
        print(f"[Validation] ⚠️ 差异 > 1e-6 的项目 ({len(mismatches)}个):")
        for m in mismatches:
            print(m)
    else:
        print("[Validation] ✅ 所有特征值差异 < 1e-6，通过验证")

    return len(mismatches) == 0


def main():
    import os

    # 路径
    raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "segments.csv")
    dataset_path = os.path.join(PROJECT_ROOT, "data", "raw",
                                "dataset-提取的合成特征被计算到每个手动分割和标记的遥测段上.csv")
    out_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "segments_18d.csv")

    # 加载原始数据
    print("[Extract18D] Loading segments.csv...")
    source_df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    print(f"[Extract18D] Loaded {len(source_df)} rows, {source_df.segment.nunique()} segments")

    # 提取特征
    df = generate_dataset(source_df)

    # 保存
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[Extract18D] Saved to: {out_path}")
    print(f"[Extract18D] Shape: {df.shape}")
    print(f"[Extract18D] Columns: {list(df.columns)}")

    # 验证
    validate_against_dataset(df, dataset_path)

    return df


if __name__ == "__main__":
    main()
