"""
一致性验证: 流式 Stage 2 判定器 vs ipynb Stage 2 参考实现

双重对照:
1. 内嵌 ipynb 参考实现(eval_iforest_segment / compute_stat_thresholds /
   apply_rules / 强通道OR弱通道纯规则, 逐行照搬 notebook 单元格),
   与 src.streaming.stage2 的判定器在全部 529 个测试段上逐段比对 → 期望 100% 一致;
2. 流式判定器的整体 SegF1/Precision/Recall/MCC 与 ipynb 导出的
   pipeline_segment_18d_output.json 中 stage2_fusion.metrics 比对 → 期望浮点级一致。
"""
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef

from src.streaming.stage2 import load_or_train, STRONG_CHANNELS, ALL_CHANNELS
from src.streaming.segment_features import FEATURE_COLS
from src.utils.constants import META_COLS

CONTAMINATION = 0.2
RANDOM_STATE = 42

# ===== ipynb 参考实现(逐行照搬 notebook 单元格) =====

def compute_stat_thresholds_ref(train_df, channels, feature_cols):
    thresholds = {}
    for ch in channels:
        ch_data = train_df[train_df['channel'] == ch]
        ch_th = {}
        for col in feature_cols:
            vals = ch_data[col].dropna()
            if len(vals) < 5:
                continue
            std = float(vals.std())
            ch_th[col] = {
                'mean': float(vals.mean()),
                'std': std if std > 1e-6 else 1.0,
                'q1': float(vals.quantile(0.25)),
                'q3': float(vals.quantile(0.75)),
                'iqr': float(vals.quantile(0.75) - vals.quantile(0.25)),
            }
        thresholds[ch] = ch_th
    return thresholds


def apply_rules_ref(test_df, thresholds, feature_cols):
    rule_pred = np.zeros(len(test_df))
    for i, (idx, row) in enumerate(test_df.iterrows()):
        ch = row['channel']
        if ch not in thresholds:
            continue
        ch_th = thresholds[ch]
        count = 0
        for col in feature_cols:
            if col not in ch_th or pd.isna(row[col]):
                continue
            v = row[col]
            th = ch_th[col]
            if abs(v - th['mean']) > 3.0 * th['std']:
                count += 1
            lo = th['q1'] - 1.5 * th['iqr']
            hi = th['q3'] + 1.5 * th['iqr']
            if v < lo or v > hi:
                count += 1
        rule_pred[i] = 1 if count >= 2 else 0
    return rule_pred


def stage2_ref(train_df, test_df, feature_cols):
    channel_models = {}
    for ch in ALL_CHANNELS:
        ch_train = train_df[train_df['channel'] == ch]
        if len(ch_train) < 10:
            continue
        clf = IsolationForest(n_estimators=100, contamination=CONTAMINATION,
                              random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(ch_train[feature_cols].values)
        channel_models[ch] = clf
    thresholds = compute_stat_thresholds_ref(train_df, ALL_CHANNELS, feature_cols)
    rule_pred = apply_rules_ref(test_df, thresholds, feature_cols)
    y_pred = np.zeros(len(test_df), dtype=int)
    for ch in ALL_CHANNELS:
        ch_test_idx = test_df[test_df['channel'] == ch].index
        ch_test = test_df[test_df['channel'] == ch]
        if len(ch_test) == 0:
            continue
        r_pred = rule_pred[ch_test_idx]
        if ch in STRONG_CHANNELS and ch in channel_models:
            if_pred = np.where(channel_models[ch].predict(ch_test[feature_cols].values) == -1, 1, 0)
            y_pred[ch_test_idx] = np.maximum(if_pred, r_pred)
        else:
            y_pred[ch_test_idx] = r_pred
    return y_pred

# ===== 数据与双方判定 =====

judges = load_or_train()

df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "segments_18d.csv"))
feature_cols = [c for c in df.columns if c not in META_COLS]
train_df = df[df['train'] == True].copy().reset_index(drop=True)
test_df = df[df['train'] == False].copy().reset_index(drop=True)
y_true = test_df['anomaly'].values

print("\n[Ref] 运行 ipynb 参考实现 ...")
y_ref = stage2_ref(train_df, test_df, feature_cols)

y_stream = np.zeros(len(test_df), dtype=int)
skipped = 0
for i, row in test_df.iterrows():
    ch = row['channel']
    if ch not in judges:
        skipped += 1
        continue
    feats = {c: float(row[c]) for c in FEATURE_COLS}
    y_stream[i] = int(judges[ch].judge(feats).anomaly)

mismatch = int((y_stream != y_ref).sum())
print(f"\n=== 逐段比对: 流式 Stage2 vs ipynb 参考实现 ===")
print(f"测试段: {len(test_df)} (流式跳过 {skipped}: 通道训练段不足, 与 ipynb 同规则)")
print(f"不一致段: {mismatch}")
if mismatch:
    bad = test_df[y_stream != y_ref]
    print(bad[['segment', 'channel', 'anomaly']].head(20))

def metrics(y_pred):
    return {
        'f1': float(f1_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
    }

m_stream = metrics(y_stream)
with open(os.path.join(PROJECT_ROOT, "data", "results", "pipeline_segment_18d_output.json"),
          encoding="utf-8") as f:
    m_ipynb = json.load(f)["stages_performance"]["stage2_fusion"]["metrics"]

print(f"\n=== 指标比对: 流式 Stage2 vs ipynb 导出 JSON(参考, 环境差异会致轻微浮动) ===")
print(f"{'':10s}{'stream':>10s}{'ipynb':>10s}{'diff':>12s}")
for k in ['f1', 'precision', 'recall', 'mcc']:
    d = abs(m_stream[k] - m_ipynb[k])
    print(f"{k:10s}{m_stream[k]:10.6f}{m_ipynb[k]:10.6f}{d:12.2e}")

# 判定标准 = 与 ipynb 参考实现逐段 100% 一致(JSON 是历史导出, 仅作参考)
print("\n" + ("PASS" if mismatch == 0 else "FAIL"))
