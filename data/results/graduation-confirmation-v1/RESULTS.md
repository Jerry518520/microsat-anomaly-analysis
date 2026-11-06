# 毕业设计检测对照数据

规程为 graduation-confirmation-v1。统一对照与当前工程方案分别登记。

| 方法 | Precision | Recall | F1 | Accuracy | MCC | FP | FN |
|---|---|---|---|---|---|---|---|
| AND | 0.5678 | 0.5929 | 0.5801 | 0.8166 | 0.4630 | 51 | 46 |
| OR | 0.4027 | 0.8053 | 0.5369 | 0.7032 | 0.3984 | 135 | 22 |
| channel_if | 0.4129 | 0.7345 | 0.5287 | 0.7202 | 0.3807 | 118 | 30 |
| engineering_gate | 0.5758 | 0.6726 | 0.6204 | 0.8242 | 0.5095 | 56 | 37 |
| gate | 0.5141 | 0.6460 | 0.5725 | 0.7940 | 0.4441 | 69 | 40 |
| global_if | 0.2969 | 0.5044 | 0.3738 | 0.6389 | 0.1533 | 135 | 56 |
| rule | 0.5245 | 0.6637 | 0.5859 | 0.7996 | 0.4617 | 68 | 38 |
| soft | 0.5191 | 0.6018 | 0.5574 | 0.7958 | 0.4276 | 63 | 45 |

529 段属于已查看基准复核。engineering_gate 使用历史固定通道参数，单独作为工程方案参照。

nested_metrics.json 使用官方训练部分的 1594 段完成 3 折嵌套评价。每个外层段只评分一次。所有参数与通道算子均在对应内层验证部分选择。

metrics.json 含逐通道指标、混淆矩阵、适用 ROC-AUC 与 AP、通道簇 bootstrap 区间及配对差值。小样本通道和单类通道均保留。通道簇只有 9 个，区间需结合各通道结果阅读。

独立复算命令为 .venv/Scripts/python.exe -m src.evaluation.saved_predictions data/results/graduation-confirmation-v1/predictions.csv --output metrics-recomputed.json。
