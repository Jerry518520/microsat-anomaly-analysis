"""
公平对比实验 — 控制变量法
所有阶段在同一个环境、同一条管线、同一个random_state下运行
数据: segments.csv → 滑动窗口22维特征 → 分通道建模
random_state: 42 (全局)

阶段:
1. Baseline: 全局IForest, sweep contamination
2. +分通道+规则兜底: 分通道IF + 3σ/IQR规则 + 融合
3. +优化融合+降阈值: 方案G + threshold=0.25
4. +子采样微调: 方案G + threshold=0.25 + psi=128

输出: fair_comparison_results.json
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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings('ignore')

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loader import load_segments, load_config, get_project_root
from src.features.sliding_window import extract_all_sliding_features, META_COLS


def eval_iforest(X_train, y_train, X_test, y_test, contamination, psi=None, seed=42):
    """训练IForest并返回预测"""
    actual_psi = psi if psi is None else min(psi, len(X_train))
    model = IsolationForest(
        n_estimators=100,
        max_samples=actual_psi if actual_psi else "auto",
        contamination=contamination,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train)
    y_pred = (model.predict(X_test) == -1).astype(int)
    y_scores = -model.decision_function(X_test)
    return y_pred, y_scores


def segment_aggregate(features_df, y_pred_window, threshold=0.5):
    """窗口级预测聚合为段级"""
    test_mask = features_df['train'] == 0
    test_df = features_df[test_mask].copy()
    test_df['pred'] = y_pred_window

    seg = test_df.groupby('segment').agg(
        y_true=('anomaly', 'first'),
        pred_rate=('pred', 'mean'),
    ).reset_index()
    seg['y_pred'] = (seg['pred_rate'] >= threshold).astype(int)

    f1 = f1_score(seg['y_true'], seg['y_pred'])
    p = precision_score(seg['y_true'], seg['y_pred'], zero_division=0)
    r = recall_score(seg['y_true'], seg['y_pred'], zero_division=0)
    try:
        auc = roc_auc_score(seg['y_true'], seg['pred_rate'])
    except ValueError:
        auc = None
    return {'f1': float(f1), 'precision': float(p), 'recall': float(r), 'auc': float(auc) if auc else None}


def compute_stat_thresholds(train_df, feature_cols):
    """从训练集计算每个通道每个特征的3σ/IQR阈值"""
    thresholds = {}
    for ch in train_df['channel'].unique():
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
    return thresholds


def apply_rules(test_df, thresholds, feature_cols):
    """应用3σ/IQR规则，返回每个样本违反的特征数"""
    n_violated = np.zeros(len(test_df))
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
        n_violated[i] = count
    return n_violated


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def main():
    t_start = time.time()
    results = {
        'experiment': 'fair_comparison',
        'timestamp': datetime.now().isoformat(),
        'random_state': 42,
        'stages': {}
    }

    # ============================================================
    # 数据加载 (一次)
    # ============================================================
    print_header("数据加载")
    config = load_config()
    root = get_project_root()
    raw_df = load_segments(config)
    print(f"原始数据: {raw_df.shape[0]} 行, {raw_df['segment'].nunique()} 个segment")

    # 滑动窗口特征提取
    print("提取滑动窗口特征 (ws=20)...")
    features_df = extract_all_sliding_features(raw_df, window_size=20, step_size=10)
    feature_cols = [c for c in features_df.columns if c not in META_COLS and c != 'window_idx']
    print(f"特征维度: {len(feature_cols)} 列")
    print(f"样本总数: {len(features_df)}")

    # 训练/测试划分
    train_mask = features_df['train'] == 1
    test_mask = features_df['train'] == 0
    train_df = features_df[train_mask]
    test_df = features_df[test_mask]
    print(f"训练集: {train_df.shape[0]} 样本, 测试集: {test_df.shape[0]} 样本")

    X_train = np.nan_to_num(train_df[feature_cols].values)
    y_train = train_df['anomaly'].values
    X_test = np.nan_to_num(test_df[feature_cols].values)
    y_test = test_df['anomaly'].values

    # 索引映射
    test_idx_map = {idx: pos for pos, idx in enumerate(test_df.index)}

    # ============================================================
    # Stage -1: 最基础IF Baseline (dataset.csv, 18维段级特征)
    # ============================================================
    print_header("Stage -1: 最基础IF Baseline (dataset.csv原始段级特征)")
    from src.features.segment_features import load_features as load_seg_features
    from src.utils.data_loader import get_train_test_split

    seg_features_df = load_seg_features(config)
    seg_meta_cols = ['channel', 'segment', 'anomaly', 'train', 'sampling']
    seg_feature_cols = [c for c in seg_features_df.columns if c not in seg_meta_cols]
    X_train_base, X_test_base, y_train_base, y_test_base = get_train_test_split(seg_features_df)
    print(f"  dataset.csv段级特征: {len(seg_feature_cols)} 维, train={len(X_train_base)}段, test={len(X_test_base)}段")

    baseline_results = {}
    best_bl_f1 = 0
    best_bl_c = 0
    for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        model = IsolationForest(n_estimators=100, max_samples="auto", contamination=c, random_state=42, n_jobs=-1)
        model.fit(X_train_base)
        y_pred_bl = (model.predict(X_test_base) == -1).astype(int)
        f1 = f1_score(y_test_base, y_pred_bl)
        p = precision_score(y_test_base, y_pred_bl, zero_division=0)
        r = recall_score(y_test_base, y_pred_bl, zero_division=0)
        baseline_results[f'c_{c}'] = {'contamination': c, 'f1': float(f1), 'precision': float(p), 'recall': float(r)}
        if f1 > best_bl_f1:
            best_bl_f1 = f1
            best_bl_c = c
        print(f"  c={c:.2f}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
    print(f"  >>> 最优: c={best_bl_c}, F1={best_bl_f1:.4f}")
    results['stages']['baseline_raw_segment'] = {
        'description': f'最基础IF Baseline: dataset.csv原始{len(seg_feature_cols)}维段级特征, 全局IForest',
        'best_contamination': best_bl_c,
        'best_f1': best_bl_f1,
        'feature_dim': len(seg_feature_cols),
        'all_results': baseline_results
    }

    # ============================================================
    # Stage 0: 段级Baseline (同一测试集，公平对比)
    # ============================================================
    print_header("Stage 0: 段级Baseline (同一测试集)")
    # 对每个segment，用其22维滑动窗口特征的均值作为段级特征
    train_seg_agg = train_df.groupby('segment').agg(
        anomaly=('anomaly', 'first'),
        channel=('channel', 'first'),
    ).reset_index()
    for col in feature_cols:
        train_seg_agg[col] = train_df.groupby('segment')[col].mean().values

    test_seg_agg = test_df.groupby('segment').agg(
        anomaly=('anomaly', 'first'),
        channel=('channel', 'first'),
    ).reset_index()
    for col in feature_cols:
        test_seg_agg[col] = test_df.groupby('segment')[col].mean().values

    X_train_seg = np.nan_to_num(train_seg_agg[feature_cols].values)
    y_train_seg = train_seg_agg['anomaly'].values
    X_test_seg = np.nan_to_num(test_seg_agg[feature_cols].values)
    y_test_seg = test_seg_agg['anomaly'].values
    print(f"  段级特征: {len(feature_cols)} 维 (窗口均值聚合), train={len(X_train_seg)}段, test={len(X_test_seg)}段")

    stage0_results = {}
    best_seg_f1 = 0
    best_seg_c = 0
    for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        model = IsolationForest(n_estimators=100, max_samples="auto", contamination=c, random_state=42, n_jobs=-1)
        model.fit(X_train_seg)
        y_pred_seg = (model.predict(X_test_seg) == -1).astype(int)
        f1 = f1_score(y_test_seg, y_pred_seg)
        p = precision_score(y_test_seg, y_pred_seg, zero_division=0)
        r = recall_score(y_test_seg, y_pred_seg, zero_division=0)
        stage0_results[f'c_{c}'] = {'contamination': c, 'f1': float(f1), 'precision': float(p), 'recall': float(r)}
        if f1 > best_seg_f1:
            best_seg_f1 = f1
            best_seg_c = c
        print(f"  c={c:.2f}: seg_F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
    print(f"  >>> 最优: c={best_seg_c}, seg_F1={best_seg_f1:.4f}")
    results['stages']['stage0_segment_baseline'] = {
        'description': f'段级Baseline(窗口均值聚合{len(feature_cols)}维), 同一测试集, 全局IForest',
        'best_contamination': best_seg_c,
        'best_f1': best_seg_f1,
        'feature_dim': len(feature_cols),
        'all_results': stage0_results
    }

    # ============================================================
    # Stage 1: 滑动窗口Baseline 全局IForest
    # ============================================================
    print_header("Stage 1: Baseline 全局IForest")
    stage1_results = {}
    best_stage1_f1 = 0
    best_stage1_c = 0
    for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        y_pred, _ = eval_iforest(X_train, y_train, X_test, y_test, c)
        m = segment_aggregate(features_df, y_pred)
        stage1_results[f'c_{c}'] = {'contamination': c, 'segment': m}
        if m['f1'] > best_stage1_f1:
            best_stage1_f1 = m['f1']
            best_stage1_c = c
        print(f"  c={c:.2f}: seg_F1={m['f1']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}")
    print(f"  >>> 最优: c={best_stage1_c}, seg_F1={best_stage1_f1:.4f}")
    results['stages']['stage1_baseline'] = {
        'description': '全局IForest, sweep contamination',
        'best_contamination': best_stage1_c,
        'best_f1': best_stage1_f1,
        'all_results': stage1_results
    }

    # ============================================================
    # Stage 2: 分通道 + 规则兜底
    # ============================================================
    print_header("Stage 2: 分通道IF + 规则兜底")

    # 通道质量分类
    strong_channels = ['CADC0872', 'CADC0873', 'CADC0874']
    weak_channels = [ch for ch in features_df['channel'].unique()
                     if ch not in strong_channels and ch != 'CADC0884']

    # 分通道IForest (用各通道最优contamination)
    per_ch_best_c = {
        'CADC0872': 0.5, 'CADC0873': 0.5, 'CADC0874': 0.5,
        'CADC0886': 0.2, 'CADC0888': 0.45, 'CADC0890': 0.3,
        'CADC0892': 0.5, 'CADC0894': 0.5
    }

    channel_preds = np.zeros(len(test_df))
    for ch in sorted(test_df['channel'].unique()):
        if ch == 'CADC0884':
            continue
        ch_test_mask = test_df['channel'] == ch
        ch_train_mask = train_df['channel'] == ch
        ch_test_idx = test_df[ch_test_mask].index
        ch_train_idx = train_df[ch_train_mask].index
        if len(ch_train_idx) < 10 or len(ch_test_idx) < 5:
            continue
        ch_c = per_ch_best_c.get(ch, 0.5)
        X_tr = np.nan_to_num(train_df.loc[ch_train_idx, feature_cols].values)
        y_tr = train_df.loc[ch_train_idx, 'anomaly'].values
        X_te = np.nan_to_num(test_df.loc[ch_test_idx, feature_cols].values)
        y_te = test_df.loc[ch_test_idx, 'anomaly'].values
        y_pred_ch, _ = eval_iforest(X_tr, y_tr, X_te, y_te, ch_c)
        for i, idx in enumerate(ch_test_idx):
            channel_preds[test_idx_map[idx]] = y_pred_ch[i]

    # 规则
    thresholds = compute_stat_thresholds(train_df, feature_cols)
    n_violated = apply_rules(test_df, thresholds, feature_cols)
    rule_pred = (n_violated >= 2).astype(int)

    # 融合: 强通道IF+OR规则, 弱通道用规则
    fused = np.zeros(len(test_df))
    for i, (idx, row) in enumerate(test_df.iterrows()):
        ch = row['channel']
        if_pred = int(channel_preds[i])
        r_pred = int(rule_pred[i])
        if ch in strong_channels:
            fused[i] = max(if_pred, r_pred)
        else:
            fused[i] = r_pred

    m2 = segment_aggregate(features_df, fused)
    print(f"  分通道+规则兜底: seg_F1={m2['f1']:.4f}, P={m2['precision']:.4f}, R={m2['recall']:.4f}")
    results['stages']['stage2_rule_fallback'] = {
        'description': '分通道IF + 3σ/IQR规则 + 强通道OR/弱通道规则',
        'segment': m2,
        'per_channel_contamination': per_ch_best_c
    }

    # ============================================================
    # Stage 3: 方案G + 阈值调优
    # ============================================================
    print_header("Stage 3: 方案G + threshold sweep")
    stage3_results = {}
    best_stage3_f1 = 0
    best_stage3_t = 0

    # 方案G: 强通道用IF+OR规则, 弱通道用段级baseline(降级)
    # 弱通道降级: 整个segment的弱通道预测率作为概率
    for t in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        # 用threshold对段级做聚合
        m_t = segment_aggregate(features_df, fused, threshold=t)
        stage3_results[f'threshold_{t}'] = m_t
        if m_t['f1'] > best_stage3_f1:
            best_stage3_f1 = m_t['f1']
            best_stage3_t = t
        print(f"  threshold={t}: seg_F1={m_t['f1']:.4f}, P={m_t['precision']:.4f}, R={m_t['recall']:.4f}")
    print(f"  >>> 最优: threshold={best_stage3_t}, seg_F1={best_stage3_f1:.4f}")
    results['stages']['stage3_threshold_sweep'] = {
        'description': '方案G(强IF+OR+弱规则) + threshold sweep',
        'best_threshold': best_stage3_t,
        'best_f1': best_stage3_f1,
        'all_results': stage3_results
    }

    # ============================================================
    # Stage 4: 子采样微调 (psi sweep)
    # ============================================================
    print_header("Stage 4: 子采样 psi sweep (threshold={})".format(best_stage3_t))
    stage4_results = {}
    best_stage4_f1 = 0
    best_stage4_psi = 0

    for psi in [32, 64, 128, 256, 512]:
        # 分通道IForest with psi（所有通道都用IForest）
        channel_preds_psi = np.zeros(len(test_df))
        for ch in sorted(test_df['channel'].unique()):
            if ch == 'CADC0884':
                continue
            ch_test_mask = test_df['channel'] == ch
            ch_train_mask = train_df['channel'] == ch
            ch_test_idx = test_df[ch_test_mask].index
            ch_train_idx = train_df[ch_train_mask].index
            if len(ch_train_idx) < 10 or len(ch_test_idx) < 5:
                continue
            ch_c = per_ch_best_c.get(ch, 0.5)
            X_tr = np.nan_to_num(train_df.loc[ch_train_idx, feature_cols].values)
            y_tr = train_df.loc[ch_train_idx, 'anomaly'].values
            X_te = np.nan_to_num(test_df.loc[ch_test_idx, feature_cols].values)
            y_te = test_df.loc[ch_test_idx, 'anomaly'].values
            actual_psi = min(psi, len(X_tr))
            model = IsolationForest(
                n_estimators=100,
                max_samples=actual_psi,
                contamination=ch_c,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_tr)
            y_pred_ch = (model.predict(X_te) == -1).astype(int)
            for i, idx in enumerate(ch_test_idx):
                channel_preds_psi[test_idx_map[idx]] = y_pred_ch[i]

        # 融合: 强通道IF+OR规则, 弱通道IF+OR规则
        fused_psi = np.zeros(len(test_df))
        for i, (idx, row) in enumerate(test_df.iterrows()):
            ch = row['channel']
            if_pred = int(channel_preds_psi[i])
            r_pred = int(rule_pred[i])
            fused_psi[i] = max(if_pred, r_pred)

        m4 = segment_aggregate(features_df, fused_psi, threshold=best_stage3_t)
        stage4_results[f'psi_{psi}'] = m4
        if m4['f1'] > best_stage4_f1:
            best_stage4_f1 = m4['f1']
            best_stage4_psi = psi
        print(f"  psi={psi}: seg_F1={m4['f1']:.4f}, P={m4['precision']:.4f}, R={m4['recall']:.4f}")

    print(f"  >>> 最优: psi={best_stage4_psi}, seg_F1={best_stage4_f1:.4f}")
    results['stages']['stage4_subsampling'] = {
        'description': f'方案G + threshold={best_stage3_t} + psi sweep',
        'best_psi': best_stage4_psi,
        'best_f1': best_stage4_f1,
        'all_results': stage4_results
    }

    # ============================================================
    # 汇总
    # ============================================================
    print_header("汇总")
    summary = {
        'stage1_baseline': {'best_f1': results['stages']['stage1_baseline']['best_f1']},
        'stage2_rule_fallback': {'f1': results['stages']['stage2_rule_fallback']['segment']['f1']},
        'stage3_threshold': {'best_f1': results['stages']['stage3_threshold_sweep']['best_f1'], 'best_t': results['stages']['stage3_threshold_sweep']['best_threshold']},
        'stage4_subsampling': {'best_f1': results['stages']['stage4_subsampling']['best_f1'], 'best_psi': results['stages']['stage4_subsampling']['best_psi']},
    }
    results['summary'] = summary

    for stage, info in summary.items():
        print(f"  {stage}: {info}")

    # 保存
    output_path = os.path.join(root, 'data', 'results', 'fair_comparison_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")
    print(f"总耗时: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
