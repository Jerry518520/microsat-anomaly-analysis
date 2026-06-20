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
CHANNEL_PHYSICS = {
    "CADC0872": "Magnetometer X-axis 磁力计X轴",
    "CADC0873": "Magnetometer Y-axis 磁力计Y轴",
    "CADC0874": "Magnetometer Z-axis 磁力计Z轴",
    "CADC0884": "Photodiode 1 angle 光电二极管1",
    "CADC0886": "Photodiode 2 angle 光电二极管2",
    "CADC0888": "Photodiode 3 angle 光电二极管3",
    "CADC0890": "Photodiode 4 angle 光电二极管4",
    "CADC0892": "Photodiode 5 angle 光电二极管5",
    "CADC0894": "Photodiode 6 angle 光电二极管6",
}

# 无异常通道（数据集中始终为正常）
NO_ANOMALY_CHANNELS = {"CADC0884"}

# 告警阈值
FAULT_THRESHOLD = 0.05

# 元数据列（滑动窗口/段级特征通用）
META_COLS = ["channel", "segment", "anomaly", "train", "sampling"]
