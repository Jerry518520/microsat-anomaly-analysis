"""
项目公共常量
消除 CHANNEL_MAP、META_COLS 等在多文件中的重复定义
"""

# 通道映射：遥测通道代码 → 物理含义
CHANNEL_MAP = {
    "CADC0872": "磁力计 X轴",
    "CADC0873": "磁力计 Y轴",
    "CADC0874": "磁力计 Z轴",
    "CADC0884": "光电二极管 1",
    "CADC0886": "光电二极管 2",
    "CADC0888": "光电二极管 3",
    "CADC0890": "光电二极管 4",
    "CADC0892": "光电二极管 5",
    "CADC0894": "光电二极管 6",
}

# 英文通道描述（用于 RAG prompt）
# 通道物理含义 —— 依据官方基准论文（Scientific Data 2024, DOI 10.1038/s41597-025-05035-3）
# 与 arXiv:2407.04730 原文：
#   "They include 3 magnetometer telemetry channels: I_B_FB_MM_0 (CADC0872),
#    I_B_FB_MM_1 (CADC0873), I_B_FB_MM_2 (CADC0874), and 6 photo diode (PD)
#    channels: I_PD1_THETA (CADC0884), I_PD2_THETA (CADC0886),
#    I_PD3_THETA (CADC0888), I_PD4_THETA (CADC0890), I_PD5_THETA (CADC0892),
#    and I_PD6_THETA (CADC0894)."
#
# ⚠ 更正记录（此前本表有两处臆测，均已按官方原文修正）：
#   1. 磁力计原写 "X-axis/Y-axis/Z-axis" —— **官方未定义轴向**，只给出
#      I_B_FB_MM_0/1/2 编号。不得臆断为三轴。原文亦未说明 I_ 前缀是否表示
#      输出电流及具体单位。
#   2. 光电二极管原写 "Photodiode N angle" 但漏了通道名中的 _THETA，
#      且中文只写「光电二极管N」。官方通道名为 I_PD*_THETA。
#
# 采样率与实测值域（本项目 segments.csv 实测，供 LLM 解释时参考）：
#   MM 系列（0872/0873/0874）：sampling=1s，value 量级 ~1e-5（±5e-5）
#   PD_THETA 系列（0884~0894）：sampling=5s，value ∈ [0, π/2]，
#     CADC0884 与 CADC0892 上界实测恰为 1.5708 = π/2 → **确为角度量纲（弧度）**，
#     非光强。这解释了为何其数值与磁力计差 6 个数量级。
#
# 官方论文明确给出的异常表现（原文列举，用于 RAG 解释的物理依据）：
#   "Several types of signal distortions are depicted, including peaks,
#    deformations, noise (CADC0873), irregular periodicity (CADC0886),
#    short (CADC0892, CADC0894) and long data gaps (CADC0874)."
CHANNEL_PHYSICS = {
    "CADC0872": "Magnetometer 0 (I_B_FB_MM_0) 磁力计0号，1s采样，量级~1e-5",
    "CADC0873": "Magnetometer 1 (I_B_FB_MM_1) 磁力计1号，官方示出异常表现=噪声(noise)",
    "CADC0874": "Magnetometer 2 (I_B_FB_MM_2) 磁力计2号，官方示出异常表现=长数据缺口(long data gap)",
    "CADC0884": "Photodiode 1 theta (I_PD1_THETA) 光电二极管1测角，5s采样，量纲=弧度",
    "CADC0886": "Photodiode 2 theta (I_PD2_THETA) 光电二极管2测角，官方示出异常表现=不规则周期性(irregular periodicity)",
    "CADC0888": "Photodiode 3 theta (I_PD3_THETA) 光电二极管3测角，量纲=弧度",
    "CADC0890": "Photodiode 4 theta (I_PD4_THETA) 光电二极管4测角，量纲=弧度",
    "CADC0892": "Photodiode 5 theta (I_PD5_THETA) 光电二极管5测角，官方示出异常表现=短数据缺口(short data gap)",
    "CADC0894": "Photodiode 6 theta (I_PD6_THETA) 光电二极管6测角，官方示出异常表现=短数据缺口(short data gap)",
}

# 无异常通道（数据集中始终为正常）
# 实测（load_split 后按集统计）：CADC0884 在 fit=97段/0异常、val=25段/0异常、
# test=36段/0异常 —— **三个集合全为 0**，因此排除它的依据是「该通道在全部
# 可用数据上无正类样本，无法训练也无法评估」，与 test 的具体结果无关。
# 注意：不要用「test 上全错所以排除」作为理由，那属于用测试集信息做决策。
NO_ANOMALY_CHANNELS = {"CADC0884"}

# ===== 异常形态分类（anomaly type classification）=====
#
# 【为什么需要这一段】原 `_classify_anomaly_type` 读的 5 个特征名
# （n_peaks_1.0 / crest_factor / impulse_factor / min / max）在数据集里
# **一个都不存在**，导致所有异常都落到 fallback `unusual_shapes`。
# 下面是把规则与阈值集中到constants 的地方，pipeline 只负责读表 + 应用。
#
# 【阈值来源】全部由 src/experiments/framework.py::load_split() 实测标定，
# 只用 fit 集设计、val 集验证，**未使用 test 集**。
# 形态规则只用「跨通道可比」的量：峰计数与差分峰计数。
#
# 【为什么不看 std/var/mean 做阈值】这些是物理量纲，跨通道差4 个数量级
# （实测段中位数：磁力计通道 std≈2e-5，光电二极管通道 std≈0.3），
# 同一绝对阈值在不同通道含义完全不同，不能直接分类。
#
# 实测富集度（enrichment = 该形态段异常率 / 全体段异常率 0.201）：
#   intermittent_spikes : fit 4.96 / val 4.98← 唯一强判别形态
#   high_freq_jitter    : fit 0.61 / val 0.59   ← 负向富集，见下方说明
#   low_activity        : fit 0.44 / val 0.57   ← 负向富集，见下方说明
#   unusual_shapes      : fit 1.59 / val 1.57   ← fallback
#
# ⚠ 富集度 <1 的两类**不是**「正常形态」，它们是「在已判定为异常的段里
#   仍然能被稳定识别出的形态标签」。低频跳变/持续高频抖动这两类本身
#   在正常段里更常见（故负向富集），但仍是有解释力的形态描述；
#   阈值按 fit 段计数分布的 p20/p80 边界标定，val 上分布一致故保留。

# 形态分类实际使用的特征（跨通道可比的计数/无量纲量）
ANOMALY_TYPE_FEATURES = [
    "n_peaks",        # 段内峰值数（主判别特征，全体 AUC 0.836）
    "diff_peaks",     # 一阶差分峰数（全体 AUC 0.386，反向：异常段反而更低）
    "diff2_peaks",    # 二阶差分峰数（全体 AUC 0.321，反向同上）
]

# 送进 feature_summary 的特征（分类用到的 + 有解释价值的）
FEATURE_SUMMARY_KEYS = [
    "n_peaks", "smooth10_n_peaks", "kurtosis", "skew",
    "diff_peaks", "diff2_peaks", "std", "var", "len",
]

# 形态判定阈值（fit 集实测标定，val 集验证一致）
#   n_peaks：正常段 p50=1 / p75=1 / p95=2；异常段 p50=2 / p90=3
#   diff_peaks：正常段 p50=6；异常段 p20=1 / p50=2 / p80=13
TYPE_THRESHOLDS = {
    # 间歇尖峰：峰数明显高于正常段中位，且帧间差分极低（尖峰孤立在平坦段上）
    "intermittent_spikes_n_peaks_min": 3,
    "intermittent_spikes_diff_peaks_max": 2,
    # 持续高频抖动：一阶差分峰数超过异常段 p80
    "high_freq_jitter_diff_peaks_min": 10,
    # 低活动：峰数与两阶差分同时处于异常段低分位（信号近乎静止）
    "low_activity_n_peaks_max": 1,
    "low_activity_diff2_peaks_max": 5,
}

# 形态 → 中文说明（供 RAG prompt 与前端展示）
ANOMALY_TYPE_LABELS = {
    "intermittent_spikes": "间歇性尖峰（孤立脉冲）",
    "high_freq_jitter": "持续高频抖动",
    "low_activity": "低活动/近静止",
    "unusual_shapes": "其他异常形态",
}

# 告警阈值
FAULT_THRESHOLD = 0.05

# 元数据列（段级特征通用）
META_COLS = ["channel", "segment", "anomaly", "train", "sampling"]
