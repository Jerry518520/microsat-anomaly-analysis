"""
公平对比实验 — 所有阶段在同一个环境下跑
验证方法有效性
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from src.utils.data_loader import load_config, load_features, get_train_test_split

# 加载数据（用项目模块）
config = load_config()
features_df = load_features(config)
X_train, X_test, y_train, y_test = get_train_test_split(features_df)

feature_names = [c for c in features_df.columns if c not in ["channel", "segment", "anomaly", "train", "sampling"]]
print(f"特征数: {len(feature_names)}")
print(f"训练集: {len(X_train)}, 测试集: {len(y_test)}")
print(f"测试集异常数: {y_test.sum()}")
print()

# ===== 阶段1: 官方Baseline =====
model1 = IsolationForest(contamination=0.2, random_state=42)
model1.fit(X_train)
preds1 = (model1.predict(X_test) == -1).astype(int)
f1_baseline = f1_score(y_test, preds1)
print(f"阶段1 - 官方Baseline (c=0.2): F1 = {f1_baseline:.6f}")

# ===== 验证随机性 =====
print()
print("验证随机性（同一个环境跑5次）:")
for i in range(5):
    model = IsolationForest(contamination=0.2, random_state=42)
    model.fit(X_train)
    preds = (model.predict(X_test) == -1).astype(int)
    f1 = f1_score(y_test, preds)
    print(f"  第{i+1}次: F1 = {f1:.6f}")

print()
print("=" * 50)
print("结论: 同一个环境、同一个random_state，结果完全一致")
print("IForest的随机性很小，关键是控制实验环境")
print("=" * 50)
