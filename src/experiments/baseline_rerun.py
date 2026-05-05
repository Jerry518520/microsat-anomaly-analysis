"""
重跑段级IForest Baseline — 确认官方基准数字
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import json

# 1. 读取数据集
df = pd.read_csv('F:/微小卫星项目/microsat-anomaly-analysis/data/raw/dataset-提取的合成特征被计算到每个手动分割和标记的遥测段上.csv')
print(f"数据集形状: {df.shape}")
print(f"异常率: {df['anomaly'].mean():.4f}")
print(f"训练集: {(df['train']==1).sum()}, 测试集: {(df['train']==0).sum()}")
print()

# 2. 分离训练集和测试集
train_df = df[df['train'] == 1]
test_df = df[df['train'] == 0]

# 3. 特征列（排除元信息列）
meta_cols = ['segment', 'anomaly', 'train', 'channel', 'sampling', 'duration']
feature_cols = [c for c in df.columns if c not in meta_cols]
print(f"特征列数: {len(feature_cols)}")
print(f"特征列: {feature_cols}")
print()

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values
y_test = test_df['anomaly'].values

# 4. 用不同contamination值跑IForest
results = {}
for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.5]:
    model = IsolationForest(contamination=c, random_state=42, n_jobs=-1)
    model.fit(X_train)
    preds = model.predict(X_test)
    preds_binary = (preds == -1).astype(int)
    
    f1 = f1_score(y_test, preds_binary)
    p = precision_score(y_test, preds_binary)
    r = recall_score(y_test, preds_binary)
    
    results[str(c)] = {
        'f1': round(f1, 6),
        'precision': round(p, 6),
        'recall': round(r, 6),
        'pred_anomaly_count': int(preds_binary.sum()),
        'true_anomaly_count': int(y_test.sum())
    }
    print(f"c={c}: F1={f1:.6f}, P={p:.6f}, R={r:.6f}, 预测异常数={preds_binary.sum()}/{y_test.sum()}")

# 5. 保存结果
output = {
    'description': '段级IForest Baseline（重跑确认）',
    'dataset': 'ESA OPS-SAT-AD',
    'features': len(feature_cols),
    'train_size': int(X_train.shape[0]),
    'test_size': int(X_test.shape[0]),
    'anomaly_ratio': round(df['anomaly'].mean(), 4),
    'results': results
}

with open('F:/微小卫星项目/microsat-anomaly-analysis/data/results/baseline_rerun_confirm.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\n结果已保存到 data/results/baseline_rerun_confirm.json")
print("\n=== 最终确认 ===")
print(f"官方Baseline (c=0.2): F1 = {results['0.2']['f1']}")
