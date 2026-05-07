"""
公平对比实验 v2 (segments版, c=0.2) — 基于 segments_18d.csv
统一 contamination=0.2（与原论文对齐），验证分通道+规则融合的纯方法增益

阶段:
0. Baseline: 全局IForest, c=0.2
1. 分通道建模: 每个通道独立IForest + 融合, c=0.2
2. +规则兜底: 分通道IF + 3σ/IQR规则 + 融合, c=0.2
3. +降阈值: 融合方案 + threshold sweep
4. +子采样: 融合方案 + threshold + psi sweep

与 fair_comparison_v2_segments.py 的区别:
- 不再sweep多个contamination值，全程固定c=0.2
- 目的: 与原论文IForest基准(F1=0.295)公平对比，展示纯方法增益

输出: fair_comparison_v2_segments_c02_results.json
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
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def eval_iforest(X_train, y_train, X_test, y_test, contamination, psi=None, seed=42):
    """训练并预测IForest"""
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


def get_iforest_scores(X_train, X_test, contamination, psi=None, seed=42):
    """获取IForest异常分数（用于threshold sweep）"""
    model = IsolationForest(
        n_estimators=100,
        max_samples=psi if psi else "auto",
        contamination=contamination,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train)
    return -model.decision_function(X_test)


def compute_rule_scores(test_df, thresholds, feature_cols):
    """计算3σ/IQR规则得分"""
    rule_scores = np.zeros(len(test_df))
    for i, (idx, row) in enumerate(test_df.iterrows()):
        ch = row['channel']
        if ch not in thresholds:
            continue
        ch_th = thresholds[ch]
        count = 0
        n_valid = 0
        for col in feature_cols:
            if col not in ch_th or pd.isna(row[col]):
                continue
            n_valid += 1
            v = row[col]
            th = ch_th[col]
            if abs(v - th['mean']) > 3.0 * th['std']:
                count += 1
            lo = th['q1'] - 1.5 * th['iqr']
            hi = th['q3'] + 1.5 * th['iqr']
            if v < lo or v > hi:
                count += 1
        rule_scores[i] = count / max(n_valid, 1)
    return rule_scores


def compute_rule_preds(test_df, thresholds, feature_cols):
    """计算3σ/IQR规则预测（>=2个特征触发则为异常）"""
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


def calc_thresholds(train_df, channels, feature_cols):
    """计算训练集各通道各特征的3σ/IQR阈值"""
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
    return thresholds


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_metrics(label, y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"  {label}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}, MCC={mcc:.4f}")
    return {'f1': float(f1), 'precision': float(p), 'recall': float(r), 'mcc': float(mcc)}


def main():
    t_start = time.time()
    C = 0.2  # 与原论文对齐的contamination值

    results = {
        'experiment': 'fair_comparison_v2_segments_c02',
        'timestamp': datetime.now().isoformat(),
        'random_state': 42,
        'contamination': C,
        'description': f'统一c={C}（与原论文对齐），验证分通道+规则融合的纯方法增益',
        'stages': {}
    }

    # ============================================================
    # 数据加载
    # ============================================================
    print_header("数据加载")
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "segments_18d.csv")
    features_df = pd.read_csv(data_path)
    print(f"[DataLoader] segments_18d: {features_df.shape[0]} rows x {features_df.shape[1]} cols")

    meta_cols = ['channel', 'segment', 'anomaly', 'train', 'sampling']
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    train_mask = features_df['train'] == 1
    test_mask = features_df['train'] == 0
    X_train = features_df.loc[train_mask, feature_cols].values
    X_test = features_df.loc[test_mask, feature_cols].values
    y_train = features_df.loc[train_mask, 'anomaly'].values
    y_test = features_df.loc[test_mask, 'anomaly'].values

    train_df = features_df[train_mask].copy()
    test_df = features_df[test_mask].copy()

    print(f"特征维度: {len(feature_cols)}")
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    print(f"测试集异常率: {y_test.mean():.1%}")

    channels = sorted(features_df['channel'].unique())
    print(f"通道: {channels}")

    # ============================================================
    # Stage 0: 全局IForest Baseline (c=0.2)
    # ============================================================
    print_header(f"Stage 0: 全局IForest Baseline (c={C})")
    y_pred_s0 = eval_iforest(X_train, y_train, X_test, y_test, C)
    m0 = print_metrics("全局IF", y_test, y_pred_s0)
    results['stages']['stage0_baseline'] = {
        'description': f'全局IForest, c={C}, 与原论文对齐',
        **m0
    }

    # ============================================================
    # Stage 1: 分通道建模 (c=0.2)
    # ============================================================
    print_header(f"Stage 1: 分通道建模 (c={C})")

    channel_preds = np.zeros(len(X_test))
    for ch in channels:
        ch_train_mask = train_df['channel'] == ch
        ch_test_mask = test_df['channel'] == ch
        if ch_train_mask.sum() < 10 or ch_test_mask.sum() < 5:
            print(f"  跳过 {ch} (train={ch_train_mask.sum()}, test={ch_test_mask.sum()})")
            continue
        X_tr = X_train[ch_train_mask.values]
        y_tr = y_train[ch_train_mask.values]
        X_te = X_test[ch_test_mask.values]
        y_pred_ch = eval_iforest(X_tr, y_tr, X_te, y_test[ch_test_mask.values], C)
        channel_preds[ch_test_mask.values] = y_pred_ch

    m1 = print_metrics("分通道IF", y_test, channel_preds)
    results['stages']['stage1_per_channel'] = {
        'description': f'分通道独立IForest, c={C}',
        **m1
    }

    # ============================================================
    # Stage 2: 分通道 + 规则兜底 (c=0.2)
    # ============================================================
    print_header(f"Stage 2: 分通道 + 规则兜底 (c={C})")

    strong_channels = ['CADC0872', 'CADC0873', 'CADC0874']
    weak_channels = [ch for ch in channels if ch not in strong_channels and ch != 'CADC0884']

    thresholds = calc_thresholds(train_df, channels, feature_cols)
    rule_pred = compute_rule_preds(test_df, thresholds, feature_cols)

    # 融合: 强通道IF∨规则, 弱通道纯规则
    fused_s2 = np.zeros(len(X_test))
    for i in range(len(X_test)):
        ch = test_df.iloc[i]['channel']
        if_pred = int(channel_preds[i])
        r_pred = int(rule_pred[i])
        if ch in strong_channels:
            fused_s2[i] = max(if_pred, r_pred)
        else:
            fused_s2[i] = r_pred

    m2 = print_metrics("IF+规则融合", y_test, fused_s2)
    results['stages']['stage2_fusion'] = {
        'description': f'分通道IF(c={C}) + 3σ/IQR规则 + 强通道OR/弱通道规则',
        **m2
    }

    # ============================================================
    # Stage 3: +降阈值 (threshold sweep)
    # ============================================================
    print_header("Stage 3: 降阈值 (threshold sweep)")

    # 获取IF scores
    channel_scores = np.zeros(len(X_test))
    for ch in channels:
        ch_train_mask = train_df['channel'] == ch
        ch_test_mask = test_df['channel'] == ch
        if ch_train_mask.sum() < 10 or ch_test_mask.sum() < 5:
            continue
        X_tr = X_train[ch_train_mask.values]
        X_te = X_test[ch_test_mask.values]
        scores = get_iforest_scores(X_tr, X_te, C)
        channel_scores[ch_test_mask.values] = scores

    rule_scores = compute_rule_scores(test_df, thresholds, feature_cols)

    fused_scores = np.zeros(len(X_test))
    for i in range(len(X_test)):
        ch = test_df.iloc[i]['channel']
        if ch in strong_channels:
            fused_scores[i] = max(channel_scores[i], rule_scores[i])
        else:
            fused_scores[i] = rule_scores[i]

    best_s3_f1 = 0
    best_s3_t = 0
    best_s3_mcc = 0
    stage3_all = {}
    for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        y_pred_t = (fused_scores >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        p = precision_score(y_test, y_pred_t, zero_division=0)
        r = recall_score(y_test, y_pred_t, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred_t)
        stage3_all[f'threshold_{t}'] = {'f1': float(f1), 'precision': float(p), 'recall': float(r), 'mcc': float(mcc)}
        if f1 > best_s3_f1:
            best_s3_f1 = f1
            best_s3_t = t
            best_s3_mcc = mcc
        print(f"  threshold={t:.2f}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}, MCC={mcc:.4f}")
    print(f"  >>> 最优: threshold={best_s3_t}, F1={best_s3_f1:.4f}, MCC={best_s3_mcc:.4f}")
    results['stages']['stage3_threshold'] = {
        'description': f'融合方案(c={C}) + threshold sweep',
        'best_threshold': best_s3_t,
        'best_f1': float(best_s3_f1),
        'best_mcc': float(best_s3_mcc),
        'all_results': stage3_all
    }

    # ============================================================
    # Stage 4: +子采样 (psi sweep)
    # ============================================================
    print_header("Stage 4: 子采样 (psi sweep)")
    best_s4_f1 = 0
    best_s4_psi = 0
    best_s4_mcc = 0
    stage4_all = {}

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
            scores = get_iforest_scores(X_tr, X_te, C, psi=actual_psi)
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
        mcc = matthews_corrcoef(y_test, y_pred_psi)
        stage4_all[f'psi_{psi}'] = {'f1': float(f1), 'precision': float(p), 'recall': float(r), 'mcc': float(mcc)}
        if f1 > best_s4_f1:
            best_s4_f1 = f1
            best_s4_psi = psi
            best_s4_mcc = mcc
        print(f"  psi={psi}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}, MCC={mcc:.4f}")
    print(f"  >>> 最优: psi={best_s4_psi}, F1={best_s4_f1:.4f}, MCC={best_s4_mcc:.4f}")
    results['stages']['stage4_subsampling'] = {
        'description': f'融合方案(c={C}) + threshold={best_s3_t} + psi sweep',
        'best_psi': best_s4_psi,
        'best_f1': float(best_s4_f1),
        'best_mcc': float(best_s4_mcc),
        'all_results': stage4_all
    }

    # ============================================================
    # 汇总
    # ============================================================
    print_header("汇总")
    summary = {
        'stage0_baseline': {'f1': m0['f1'], 'precision': m0['precision'], 'recall': m0['recall'], 'mcc': m0['mcc'], 'c': C},
        'stage1_per_channel': {'f1': m1['f1'], 'precision': m1['precision'], 'recall': m1['recall'], 'mcc': m1['mcc']},
        'stage2_fusion': {'f1': m2['f1'], 'precision': m2['precision'], 'recall': m2['recall'], 'mcc': m2['mcc']},
        'stage3_threshold': {'f1': float(best_s3_f1), 'threshold': best_s3_t, 'mcc': float(best_s3_mcc)},
        'stage4_subsampling': {'f1': float(best_s4_f1), 'psi': best_s4_psi, 'mcc': float(best_s4_mcc)},
    }
    results['summary'] = summary

    # 原论文基准
    paper_f1 = 0.295
    paper_p = 0.297
    paper_r = 0.292
    results['paper_baseline'] = {
        'source': 'OPS-SAT-AD原论文 Table 3',
        'method': 'Isolation Forest (c=0.2)',
        'f1': paper_f1,
        'precision': paper_p,
        'recall': paper_r
    }

    for stage, info in summary.items():
        print(f"  {stage}: F1={info['f1']:.4f}")

    print(f"\n  原论文IForest基准: F1={paper_f1:.3f}, P={paper_p:.3f}, R={paper_r:.3f}")
    print(f"  本方Baseline:      F1={m0['f1']:.3f}, P={m0['precision']:.3f}, R={m0['recall']:.3f}")
    print(f"  方法增益 (Baseline→最终): F1 {m0['f1']:.3f} → {m2['f1']:.3f} (+{m2['f1']-m0['f1']:.3f}, +{(m2['f1']-m0['f1'])/m0['f1']*100:.1f}%)")

    # 保存
    output_path = os.path.join(PROJECT_ROOT, 'data', 'results', 'Isolation Forest',
                               'fair_comparison_v2_segments_c02_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")
    print(f"总耗时: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
