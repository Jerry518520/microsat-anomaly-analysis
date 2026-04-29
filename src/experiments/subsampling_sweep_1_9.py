"""
IForest子采样实验 (1.9)
目标: 论文指出子采样大小psi从默认256降至更小值可提升AUC(0.67->0.91)
思路: 固定方案G+投票阈值0.25, 只变max_samples(psi), 扫描psi=32/64/128/256/512
      观察段级F1变化, 找最优psi
参考: IForest原论文 Liu et al. 2008, psi=256是默认值, 但对小数据集不一定最优
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from src.utils.data_loader import load_segments, load_config, get_project_root, load_features
from src.features.sliding_window import extract_all_sliding_features, META_COLS
from src.experiments.rule_fallback import compute_stat_thresholds, apply_stat_rules


def segment_aggregate(features_df, y_pred_window, threshold=0.25):
    """窗口级预测聚合为段级"""
    test_mask = features_df['train'] == 0
    test_df = features_df[test_mask].copy()
    test_df['pred'] = y_pred_window.values if hasattr(y_pred_window, 'values') else y_pred_window
    seg = test_df.groupby('segment').agg(
        y_true=('anomaly', 'first'),
        pred_rate=('pred', 'mean'),
    ).reset_index()
    seg['y_pred'] = (seg['pred_rate'] >= threshold).astype(int)
    seg_f1 = f1_score(seg['y_true'], seg['y_pred'])
    seg_p = precision_score(seg['y_true'], seg['y_pred'], zero_division=0)
    seg_r = recall_score(seg['y_true'], seg['y_pred'], zero_division=0)
    try:
        seg_auc = roc_auc_score(seg['y_true'], seg['pred_rate'])
    except ValueError:
        seg_auc = None
    return {'f1': seg_f1, 'precision': seg_p, 'recall': seg_r, 'auc': seg_auc}, seg


def per_channel_iforest_with_psi(train_df, test_df, feature_cols, best_c_dict, psi):
    """分通道IForest, 指定psi"""
    channel_preds = {}
    channel_f1 = {}
    for ch in sorted(test_df['channel'].unique()):
        ch_test = test_df[test_df['channel'] == ch]
        ch_train = train_df[train_df['channel'] == ch]
        if len(ch_train) < 10 or len(ch_test) < 5:
            continue
        ch_c = best_c_dict.get(ch, 0.5)
        X_tr = np.nan_to_num(ch_train[feature_cols].values)
        y_tr = ch_train['anomaly'].values
        X_te = np.nan_to_num(ch_test[feature_cols].values)
        y_te = ch_test['anomaly'].values
        actual_psi = min(psi, len(X_tr))
        model = IsolationForest(
            n_estimators=100,
            max_samples=actual_psi,
            contamination=ch_c,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr)
        ch_pred = (model.predict(X_te) == -1).astype(int)
        for i, idx in enumerate(ch_test.index):
            channel_preds[idx] = int(ch_pred[i])
        ch_f1 = f1_score(y_te, ch_pred)
        channel_f1[ch] = float(ch_f1)
    return channel_preds, channel_f1


def segment_baseline_iforest(psi=None):
    """段级baseline IForest"""
    config = load_config()
    root = get_project_root()
    dataset_df = load_features(config)
    meta_cols = ['channel', 'segment', 'anomaly', 'train', 'sampling']
    feat_cols = [c for c in dataset_df.columns if c not in meta_cols]
    train_mask = dataset_df['train'] == 1
    X_train = dataset_df.loc[train_mask, feat_cols].values
    X_test = dataset_df.loc[~train_mask, feat_cols].values
    test_segments = dataset_df.loc[~train_mask, 'segment'].values
    actual_psi = min(psi, len(X_train)) if psi else 'auto'
    model = IsolationForest(
        n_estimators=100,
        max_samples=actual_psi,
        contamination=0.25,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    y_pred = (model.predict(X_test) == -1).astype(int)
    seg_pred_map = {}
    for i in range(len(test_segments)):
        seg_pred_map[test_segments[i]] = int(y_pred[i])
    return seg_pred_map


def main():
    config = load_config()
    root = get_project_root()
    results_dir = os.path.join(root, config['data']['results_dir'])
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Features
    print('Step 1: Feature extraction (ws=20)')
    segments_df = load_segments(config)
    features_df = extract_all_sliding_features(segments_df, window_size=20, step_size=10)
    feature_cols = [c for c in features_df.columns if c not in META_COLS]
    train_mask = features_df['train'] == 1
    train_df = features_df[train_mask]
    test_df = features_df[~train_mask].copy()

    # Step 2: Statistical rules
    print('Step 2: Statistical rules')
    thresholds = compute_stat_thresholds(train_df, feature_cols)
    n_violated, n_total = apply_stat_rules(test_df, thresholds, feature_cols)
    test_df['rule_violations'] = n_violated
    test_df['rule_total'] = n_total
    rule_pred = pd.Series((n_violated / n_total.replace(0, 1) >= 0.2).astype(int), index=test_df.index)

    # Step 3: Best contamination per channel
    best_c = {
        'CADC0872': 0.5, 'CADC0873': 0.5, 'CADC0874': 0.5,
        'CADC0886': 0.2, 'CADC0888': 0.45, 'CADC0890': 0.3,
        'CADC0892': 0.5, 'CADC0894': 0.5,
    }
    strong_channels = {'CADC0872', 'CADC0873', 'CADC0874'}

    # Step 4: Psi sweep
    psi_values = [32, 64, 128, 256, 512]
    voting_threshold = 0.25
    all_results = {}

    header = "{:>6} | {:>6} | {:>6} | {:>6} | {:>6}".format('Psi', 'SegF1', 'P', 'R', 'AUC')
    print(header)
    print('-' * 45)

    best_f1 = 0
    best_psi = 256
    for psi in psi_values:
        print('Running psi={}...'.format(psi))

        channel_preds, channel_f1 = per_channel_iforest_with_psi(
            train_df, test_df, feature_cols, best_c, psi
        )

        seg_baseline_map = segment_baseline_iforest(psi=psi)

        pred_g = pd.Series(0, index=test_df.index, dtype=int)
        for idx in test_df.index:
            ch = test_df.loc[idx, 'channel']
            seg_id = test_df.loc[idx, 'segment']
            if ch in strong_channels:
                if_pred = channel_preds.get(idx, 0)
                r_pred = int(rule_pred.loc[idx])
                pred_g.loc[idx] = max(if_pred, r_pred)
            else:
                pred_g.loc[idx] = seg_baseline_map.get(seg_id, 0)

        metrics, seg_df = segment_aggregate(features_df, pred_g, threshold=voting_threshold)
        all_results[str(psi)] = {
            'psi': psi,
            'voting_threshold': voting_threshold,
            'metrics': metrics,
            'per_channel_f1': channel_f1,
        }
        auc_str = "{:.3f}".format(metrics['auc']) if metrics['auc'] else 'N/A'
        row = "{:>6} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6}".format(
            psi, metrics['f1'], metrics['precision'], metrics['recall'], auc_str)
        print(row)

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_psi = psi

    print('\nBest: psi={}, SegF1={:.3f}'.format(best_psi, best_f1))
    print('Baseline (psi=auto): SegF1=0.418')

    output = {
        'experiment': 'subsampling_sweep_1.9',
        'method': 'G_strong_IF_OR_Rule+weak_seg',
        'window_size': 20,
        'step_size': 10,
        'voting_threshold': voting_threshold,
        'best_contamination': best_c,
        'psi_values_tested': psi_values,
        'results': all_results,
        'best_psi': best_psi,
        'best_seg_f1': best_f1,
        'baseline_f1': 0.418,
    }
    out_path = os.path.join(results_dir, 'subsampling_sweep_1.9_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print('\n[Saved] {}'.format(out_path))
    return output


if __name__ == '__main__':
    main()
