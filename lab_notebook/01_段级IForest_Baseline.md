# 实验01: 段级IForest Baseline

日期: 2026-04-26

## 目标
复现官方benchmark，建立基线。如果基线结果和官方一致，说明我们的环境和流程是正确的。

## 思路
用最朴素的方式：每个segment提取统计特征→IForest检测。不做任何优化，先看"裸跑"效果。

## 参数
- 特征: 原始数据集的18维段级特征（segments_18d.csv / dataset.csv）
- 算法: IsolationForest, contamination=0.2（官方默认）
- sklearn版本: 1.7.2（运行时环境，poetry.lock锁定1.8.0，见实验10）
- 评估: 段级F1（与官方对齐）

## 结果
- **F1 = 0.262**（sklearn 1.7.2环境，4月26日运行）
- 原论文IForest基准: **F1 = 0.295**（OPS-SAT-AD原论文 Table 3, c=0.2）
- 对比: 官方AdaBoost(监督) F1=0.836
- ⚠️ 注: sklearn版本不同会导致IForest结果漂移（见实验10）。sklearn 1.8.0下同一配置Baseline F1=0.283

## 思路反思
0.262太低了，但这是无监督算法的"起跑线"。关键发现：
1. **无监督vs监督差距巨大**（0.262 vs 0.836），纯靠IForest走不远
2. **contamination=0.2是官方默认**，但实际异常比20.4%，这个参数没调过
3. **段级特征丢失了时序信息**——一个segment可能几百个数据点，压成一个特征向量太粗糙

## 下一步方向
保留时序信息 → 滑动窗口
