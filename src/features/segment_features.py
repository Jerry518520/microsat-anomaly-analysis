"""
段级特征模块 (Baseline)
复用 OPS-SAT-AD 官方合成特征数据集
"""

import os
import pandas as pd
from src.utils.data_loader import load_features, get_train_test_split, load_config, get_project_root


def run_segment_baseline(config: dict = None):
    """
    段级特征 baseline：
    直接使用官方 dataset-提取的合成特征.csv
    返回: (X_train, X_test, y_train, y_test, feature_names)
    """
    if config is None:
        config = load_config()

    features_df = load_features(config)
    X_train, X_test, y_train, y_test = get_train_test_split(features_df)

    meta_cols = ["channel", "segment", "anomaly", "train", "sampling"]
    feature_names = [c for c in features_df.columns if c not in meta_cols]

    print(f"[SegmentBaseline] {len(feature_names)} features, "
          f"train={len(X_train)}, test={len(X_test)}")

    return X_train, X_test, y_train, y_test, feature_names


def save_segment_features(config: dict = None):
    """将段级特征保存到 data/processed/"""
    if config is None:
        config = load_config()

    features_df = load_features(config)
    root = get_project_root()
    out_dir = os.path.join(root, config["data"]["processed_dir"])
    os.makedirs(out_dir, exist_ok=True)

    train_df = features_df[features_df["train"] == 1]
    test_df = features_df[features_df["train"] == 0]

    train_path = os.path.join(out_dir, "segment_features_train.csv")
    test_path = os.path.join(out_dir, "segment_features_test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"[SegmentBaseline] Train saved: {train_path}")
    print(f"[SegmentBaseline] Test saved: {test_path}")


if __name__ == "__main__":
    run_segment_baseline()
    save_segment_features()
