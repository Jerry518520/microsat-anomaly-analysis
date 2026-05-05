import pandas as pd

# 1. 段级特征数据集 (baseline用)
df1 = pd.read_csv('data/raw/dataset-提取的合成特征被计算到每个手动分割和标记的遥测段上.csv')
print('=== dataset.csv (baseline用) ===')
print('行数:', df1.shape[0], '列数:', df1.shape[1])
print('segment数:', df1['segment'].nunique())
test1 = df1[df1['train'] == 0]
print('test集异常数:', test1['anomaly'].sum())
print()

# 2. 原始遥测数据 (滑动窗口实验用)
df2 = pd.read_csv('data/raw/segments.csv')
print('=== segments.csv (滑动窗口实验用) ===')
print('行数:', df2.shape[0], '列数:', df2.shape[1])
print('segment数:', df2['segment'].nunique())
# test集异常数
seg_info = df2.groupby('segment').first().reset_index()
test2 = seg_info[seg_info['train'] == 0]
print('test集异常数:', test2['anomaly'].sum())
print()

# 3. 比较两个数据集的segment是否一致
segs1 = set(df1['segment'].unique())
segs2 = set(df2['segment'].unique())
print('segment交集:', len(segs1 & segs2))
print('dataset独有:', len(segs1 - segs2))
print('segments独有:', len(segs2 - segs1))
