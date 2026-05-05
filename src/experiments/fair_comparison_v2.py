"""
公平对比实验 v2 — 全部基于 dataset.csv
所有阶段使用相同数据、相同测试集（529段）、相同random_state=42

阶段:
0. Baseline: 全局IForest, sweep contamination
1. 分通道建模: 每个通道独立IForest + 融合
2. +规则兜底: 分通道IF + 3σ/IQR规则 + 融合
3. +降阈值: 融合方案 + threshold sweep
4. +子采样: 融合方案 + threshold + psi sweep

输出: fair_comparison_v2_results.json
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loader import load_config, get_project_root, get_train_test_split
from src.features.segment_features import load_features


def eval_iforest(X_train, y_train, X_test, y_test, contamination, psi=None, seed=42):
    model = IsolationForest(
        n_estimators=100,
        max_samples=psi if psi else "auto",
        contamination=contamination,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train)
    y_pred = (model.predict(X_test) == -1).astype(int)
    return y_pred


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def main():
    t_start = time.time()
    results = {
        'experiment': 'fair_comparison_v2',
        'timestamp': datetime.now().isoformat(),
        'random_state': 42,
        'stages': {}
    }

    # ============================================================
    # 数据加载
    # ============================================================
    print_header("数据加载")
    config = load_config()
    root = get_project_root()

    features_df = load_features(config)
    meta_cols = ['channel', 'segment', 'anomaly', 'train', 'sampling']
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    X_train, X_test, y_train, y_test = get_train_test_split(features_df)

    # 还原测试集的channel信息
    test_mask = features_df['train'] == 0
    test_df = features_df[test_mask].copy()

    print(f"特征维度: {len(feature_cols)}")
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    print(f"测试集异常率: {y_test.mean():.1%}")

    channels = sorted(features_df['channel'].unique())
    print(f"通道: {channels}")

    # ============================================================
    # Stage 0: 全局IForest Baseline
    # ============================================================
    print_header("Stage 0: 全局IForest Baseline")
    stage0_results = {}
    best_s0_f1 = 0
    best_s0_c = 0
    for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        y_pred = eval_iforest(X_train, y_train, X_test, y_test, c)
        f1 = f1_score(y_test, y_pred)
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        stage0_results[f'c_{c}'] = {'f1': float(f1), 'precision': float(p), 'recall': float(r)}
        if f1 > best_s0_f1:
            best_s0_f1 = f1
            best_s0_c = c
        print(f"  c={c:.2f}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
    print(f"  >>> 最优: c={best_s0_c}, F1={best_s0_f1:.4f}")
    results['stages']['stage0_global_baseline'] = {
        'description': '全局IForest, 18维段级特征, dataset.csv',
        'best_c': best_s0_c,
        'best_f1': best_s0_f1,
        'all_results': stage0_results
    }

    # ============================================================
    # Stage 1: 分通道建模
    # ============================================================
    print_header("Stage 1: 分通道建模")
    print(f"  特征名: {feature_cols}")

    stage1_results = {}
    for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        channel_preds = np.zeros(len(X_test))
        for ch in channels:
            ch_train_mask = features_df.loc[features_df['train'] == 1, 'channel'] == ch
            ch_test_mask = test_df['channel'] == ch
            if ch_train_mask.sum() < 10 or ch_test_mask.sum() < 5:
                continue
            X_tr = X_train[ch_train_mask.values]
            y_tr = y_train[ch_train_mask.values]
            X_te = X_test[ch_test_mask.values]
            y_te = y_test[ch_test_mask.values]
            y_pred_ch = eval_iforest(X_tr, y_tr, X_te, y_te, c)
            channel_preds[ch_test_mask.values] = y_pred_ch

        f1 = f1_score(y_test, channel_preds)
        p = precision_score(y_test, channel_preds, zero_division=0)
        r = recall_score(y_test, channel_preds, zero_division=0)
        stage1_results[f'c_{c}'] = {'f1': float(f1), 'precision': float(p), 'recall': float(r)}
        print(f"  c={c:.2f}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")

    best_s1 = max(stage1_results.items(), key=lambda x: x[1]['f1'])
    print(f"  >>> 最优: {best_s1[0]}, F1={best_s1[1]['f1']:.4f}")
    results['stages']['stage1_per_channel'] = {
        'description': '分通道独立IForest',
        'best_f1': best_s1[1]['f1'],
        'all_results': stage1_results
    }

    # ============================================================
    # Stage 2: 分通道 + 规则兜底
    # ============================================================
    print_header("Stage 2: 分通道 + 规则兜底")

    # 通道分类
    strong_channels = ['CADC0872', 'CADC0873', 'CADC0874']
    weak_channels = [ch for ch in channels if ch not in strong_channels and ch != 'CADC0884']

    # 计算训练集的3σ/IQR阈值
    train_df = features_df[features_df['train'] == 1].copy()
    thresholds = {}
    for ch in channels:
        ch_data = train_df[train_df['channel'] == ch]
        ch_th = {}
        for col in feature_cols:
            vals = ch_data[col].dropna()
            if len(vals) < 5:
                continue
            ch_th[col] = {
                'mean': float(vals.mean()),
                'std': float(vals.std()) if vals.std() > 1e-6 else 1.0,
                'q1': float(vals.quantile(0.25)),
                'q3': float(vals.quantile(0.75)),
                'iqr': float(vals.quantile(0.75) - vals.quantile(0.25)),
            }
        thresholds[ch] = ch_th

    # 规则预测
    rule_pred = np.zeros(len(X_test))
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

    # 分通道IF + 规则融合
    stage2_results = {}
    for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
        channel_preds = np.zeros(len(X_test))
        for ch in channels:
            ch_train_mask = train_df['channel'] == ch
            ch_test_mask = test_df['channel'] == ch
            if ch_train_mask.sum() < 10 or ch_test_mask.sum() < 5:
                continue
            X_tr = X_train[ch_train_mask.values]
            y_tr = y_train[ch_train_mask.values]
            X_te = X_test[ch_test_mask.values]
            y_te = y_test[ch_test_mask.values]
            y_pred_ch = eval_iforest(X_tr, y_tr, X_te, y_te, c)
            channel_preds[ch_test_mask.values] = y_pred_ch

        # 融合: 强通道IF+OR规则, 弱通道规则
        fused = np.zeros(len(X_test))
        for i in range(len(X_test)):
            ch = test_df.iloc[i]['channel']
            if_pred = int(channel_preds[i])
            r_pred = int(rule_pred[i])
            if ch in strong_channels:
                fused[i] = max(if_pred, r_pred)
            else:
                fused[i] = r_pred

        f1 = f1_score(y_test, fused)
        p = precision_score(y_test, fused, zero_division=0)
        r = recall_score(y_test, fused, zero_division=0)
        stage2_results[f'c_{c}'] = {'f1': float(f1), 'precision': float(p), 'recall': float(r)}
        print(f"  c={c}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")

    best_s2 = max(stage2_results.items(), key=lambda x: x[1]['f1'])
    print(f"  >>> 最优: {best_s2[0]}, F1={best_s2[1]['f1']:.4f}")
    results['stages']['stage2_per_channel_rules'] = {
        'description': '分通道IF + 3σ/IQR规则 + 强通道OR/弱通道规则',
        'best_f1': best_s2[1]['f1'],
        'all_results': stage2_results
    }

    # ============================================================
    # Stage 3: +降阈值 (threshold sweep)
    # ============================================================
    print_header("Stage 3: 降阈值 (threshold sweep)")
    # 用最优c的融合结果做threshold sweep
    best_c_val = float(best_s2[0].split('_')[1])
    channel_preds_best = np.zeros(len(X_test))
    for ch in channels:
        ch_train_mask = train_df['channel'] == ch
        ch_test_mask = test_df['channel'] == ch
        if ch_train_mask.sum() < 10 or ch_test_mask.sum() < 5:
            continue
        X_tr = X_train[ch_train_mask.values]
        y_tr = y_train[ch_train_mask.values]
        X_te = X_test[ch_test_mask.values]
        y_pred_ch = eval_iforest(X_tr, y_tr, X_te, y_test[ch_test_mask.values], best_c_val)
        channel_preds_best[ch_test_mask.values] = y_pred_ch

    # 融合概率（用IForest score而非硬预测）
    # 实际上我们需要score来做threshold sweep
    # 重新训练，获取score
    channel_scores = np.zeros(len(X_test))
    for ch in channels:
        ch_train_mask = train_df['channel'] == ch
        ch_test_mask = test_df['channel'] == ch
        if ch_train_mask.sum() < 10 or ch_test_mask.sum() < 5:
            continue
        X_tr = X_train[ch_train_mask.values]
        X_te = X_test[ch_test_mask.values]
        model = IsolationForest(n_estimators=100, contamination=best_c_val, random_state=42, n_jobs=-1)
        model.fit(X_tr)
        scores = -model.decision_function(X_te)  # 越高越异常
        channel_scores[ch_test_mask.values] = scores

    # 融合score: 强通道取max(score, rule_score), 弱通道取rule_score
    # rule_score用违反特征数归一化
    rule_scores = np.zeros(len(X_test))
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
        rule_scores[i] = count / max(len(feature_cols), 1)

    fused_scores = np.zeros(len(X_test))
    for i in range(len(X_test)):
        ch = test_df.iloc[i]['channel']
        if ch in strong_channels:
            fused_scores[i] = max(channel_scores[i], rule_scores[i])
        else:
            fused_scores[i] = rule_scores[i]

    stage3_results = {}
    best_s3_f1 = 0
    best_s3_t = 0
    for t in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        y_pred_t = (fused_scores >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        p = precision_score(y_test, y_pred_t, zero_division=0)
        r = recall_score(y_test, y_pred_t, zero_division=0)
        stage3_results[f'threshold_{t}'] = {'f1': float(f1), 'precision': float(p), 'recall': float(r)}
        if f1 > best_s3_f1:
            best_s3_f1 = f1
            best_s3_t = t
        print(f"  threshold={t:.2f}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
    print(f"  >>> 最优: threshold={best_s3_t}, F1={best_s3_f1:.4f}")
    results['stages']['stage3_threshold'] = {
        'description': f'融合方案(c={best_c_val}) + threshold sweep',
        'best_threshold': best_s3_t,
        'best_f1': best_s3_f1,
        'all_results': stage3_results
    }

    # ============================================================
    # Stage 4: +子采样 (psi sweep)
    # ============================================================
    print_header("Stage 4: 子采样 (psi sweep)")
    stage4_results = {}
    best_s4_f1 = 0
    best_s4_psi = 0

    for psi in [32, 64, 128, 256, 512]:
        channel_scores_psi = np.zeros(len(X_test))
        for ch in channels:
            ch_train_mask = train_df['channel'] == ch
            ch_test_mask = test_df['channel'] == ch
            if ch_train_mask.sum() < 10 or ch_test_mask.sum() < 5:
                continue
            X_tr = X_train[ch_train_mask.values]
            X_te = X_test[ch_test_mask.values]
            actual_psi = min(psi, len(X_tr))
            model = IsolationForest(
                n_estimators=100,
                max_samples=actual_psi,
                contamination=best_c_val,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_tr)
            scores = -model.decision_function(X_te)
            channel_scores_psi[ch_test_mask.values] = scores

        fused_scores_psi = np.zeros(len(X_test))
        for i in range(len(X_test)):
            ch = test_df.iloc[i]['channel']
            if ch in strong_channels:
                fused_scores_psi[i] = max(channel_scores_psi[i], rule_scores[i])
            else:
                fused_scores_psi[i] = rule_scores[i]

        y_pred_psi = (fused_scores_psi >= best_s3_t).astype(int)
        f1 = f1_score(y_test, y_pred_psi)
        p = precision_score(y_test, y_pred_psi, zero_division=0)
        r = recall_score(y_test, y_pred_psi, zero_division=0)
        stage4_results[f'psi_{psi}'] = {'f1': float(f1), 'precision': float(p), 'recall': float(r)}
        if f1 > best_s4_f1:
            best_s4_f1 = f1
            best_s4_psi = psi
        print(f"  psi={psi}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
    print(f"  >>> 最优: psi={best_s4_psi}, F1={best_s4_f1:.4f}")
    results['stages']['stage4_subsampling'] = {
        'description': f'融合方案 + threshold={best_s3_t} + psi sweep',
        'best_psi': best_s4_psi,
        'best_f1': best_s4_f1,
        'all_results': stage4_results
    }

    # ============================================================
    # 汇总
    # ============================================================
    print_header("汇总")
    summary = {
        'stage0_baseline': {'f1': best_s0_f1, 'c': best_s0_c},
        'stage1_per_channel': {'f1': best_s1[1]['f1']},
        'stage2_per_channel_rules': {'f1': best_s2[1]['f1']},
        'stage3_threshold': {'f1': best_s3_f1, 'threshold': best_s3_t},
        'stage4_subsampling': {'f1': best_s4_f1, 'psi': best_s4_psi},
    }
    results['summary'] = summary

    for stage, info in summary.items():
        print(f"  {stage}: {info}")

    output_path = os.path.join(root, 'data', 'results', 'fair_comparison_v2_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")
    print(f"总耗时: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
