"""
数据加载工具模块
从 raw 目录加载 OPS-SAT 遥测数据集
"""

import os
import pandas as pd
import yaml


def load_config(config_path: str = None) -> dict:
    """加载全局配置文件"""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "config.yaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> str:
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def load_segments(config: dict = None) -> pd.DataFrame:
    """
    加载 segments.csv 原始遥测数据
    返回: DataFrame (channel, timestamp, value, label, sampling, anomaly, segment, train)
    """
    if config is None:
        config = load_config()
    root = get_project_root()
    path = os.path.join(root, config["data"]["raw_dir"], config["data"]["segments_file"])
    df = pd.read_csv(path, encoding="utf-8")
    print(f"[DataLoader] segments.csv: {df.shape[0]} rows × {df.shape[1]} cols, "
          f"{df['segment'].nunique()} segments, {df['channel'].nunique()} channels")
    return df


def load_features(config: dict = None) -> pd.DataFrame:
    """
    加载合成特征数据集
    返回: DataFrame (2123 rows × 23 cols)
    """
    if config is None:
        config = load_config()
    root = get_project_root()
    path = os.path.join(root, config["data"]["raw_dir"], config["data"]["features_file"])
    df = pd.read_csv(path, encoding="utf-8")
    print(f"[DataLoader] features: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def get_train_test_split(features_df: pd.DataFrame):
    """
    按 train 列划分训练集和测试集
    返回: (X_train, X_test, y_train, y_test)
    """
    meta_cols = ["channel", "segment", "anomaly", "train", "sampling"]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    train_mask = features_df["train"] == 1
    X_train = features_df.loc[train_mask, feature_cols]
    y_train = features_df.loc[train_mask, "anomaly"]
    X_test = features_df.loc[~train_mask, feature_cols]
    y_test = features_df.loc[~train_mask, "anomaly"]

    print(f"[DataLoader] Train: {len(X_train)} (anomaly: {y_train.sum()}), "
          f"Test: {len(X_test)} (anomaly: {y_test.sum()})")
    return X_train, X_test, y_train, y_test


def get_channel_data(features_df: pd.DataFrame, channel: str):
    """获取指定通道的数据"""
    return features_df[features_df["channel"] == channel]


if __name__ == "__main__":
    config = load_config()
    segments = load_segments(config)
    features = load_features(config)
    X_train, X_test, y_train, y_test = get_train_test_split(features)
    print(f"\n通道列表: {features['channel'].unique().tolist()}")
