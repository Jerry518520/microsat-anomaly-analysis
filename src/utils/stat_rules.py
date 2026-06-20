"""
统计阈值规则工具
从 rule_fallback.py 提取，供 production 和 experiment 代码共用
"""

import numpy as np
import pandas as pd


def compute_stat_thresholds(train_features_df, feature_cols):
    """从训练集计算每个通道每个特征的阈值参数"""
    thresholds = {}
    for ch in train_features_df["channel"].unique():
        ch_data = train_features_df[train_features_df["channel"] == ch]
        ch_thresh = {}
        for col in feature_cols:
            vals = ch_data[col].dropna()
            if len(vals) < 5:
                continue
            ch_thresh[col] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()) if vals.std() > 1e-6 else 1.0,
                "q1": float(vals.quantile(0.25)),
                "q3": float(vals.quantile(0.75)),
                "iqr": float(vals.quantile(0.75) - vals.quantile(0.25)),
            }
        thresholds[ch] = ch_thresh
    return thresholds


def sigma_rule(value, mean, std, n_sigma=3.0):
    """3-sigma规则：超出mean±3σ视为异常"""
    return abs(value - mean) > n_sigma * std


def iqr_rule(value, q1, q3, iqr_val, k=1.5):
    """IQR规则：超出Q1-1.5*IQR~Q3+1.5*IQR视为异常"""
    lo = q1 - k * iqr_val
    hi = q3 + k * iqr_val
    return value < lo or value > hi


def apply_stat_rules(features_df, thresholds, feature_cols):
    """对所有样本应用统计阈值规则，返回每个样本的异常特征数"""
    n_anomaly_features = pd.Series(0, index=features_df.index, dtype=float)
    n_total_features = pd.Series(0, index=features_df.index, dtype=int)

    for idx in features_df.index:
        ch = features_df.loc[idx, "channel"]
        if ch not in thresholds:
            continue
        ch_th = thresholds[ch]
        violated = 0
        total = 0
        for col in feature_cols:
            if col not in ch_th or pd.isna(features_df.loc[idx, col]):
                continue
            v = features_df.loc[idx, col]
            th = ch_th[col]
            total += 1
            if sigma_rule(v, th["mean"], th["std"]) or iqr_rule(v, th["q1"], th["q3"], th["iqr"]):
                violated += 1
        n_anomaly_features.loc[idx] = violated
        n_total_features.loc[idx] = total

    return n_anomaly_features, n_total_features
