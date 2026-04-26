# 微小卫星遥测智能异常检测与RAG解释辅助系统

> Shanghai Dianji University — 大学生创新创业训练计划项目
> 周期：2026.4 - 2027.3 | 指导教师：芦立华

## 项目概述

本项目基于ESA OPS-SAT卫星遥测数据集，实现：
1. **Isolation Forest异常检测**：段级特征 + 滑动窗口特征对比
2. **特征贡献度分析**：解释哪些特征驱动异常判定
3. **RAG辅助解释**：结合卫星手册知识库，用LLM生成异常原因的自然语言解释
4. **Streamlit Web平台**：集成检测+解释+可视化

## 项目结构

```
microsat-anomaly-analysis/
├── pyproject.toml          # Poetry 项目配置
├── .venv/                  # Python 虚拟环境 (poetry managed)
├── configs/                # 配置文件
│   └── config.yaml         # 全局参数配置
├── data/
│   ├── raw/                # 原始数据 (segments.csv, 合成特征.csv)
│   ├── processed/          # 处理后的特征数据
│   └── results/            # 实验结果 (指标、图表)
├── src/
│   ├── features/           # 特征工程
│   │   ├── segment_features.py   # 段级特征 (baseline)
│   │   └── sliding_window.py     # 滑动窗口特征提取
│   ├── models/             # 模型
│   │   ├── iforest_baseline.py   # IForest 基准模型
│   │   ├── iforest_sliding.py    # IForest + 滑动窗口
│   │   └── comparison.py         # 段级 vs 滑动窗口对比实验
│   ├── evaluation/         # 评估
│   │   ├── metrics.py            # F1, AUC_ROC, per-channel F1
│   │   └── feature_importance.py # 特征贡献度分析
│   ├── rag/                # RAG 解释模块 (Week2+)
│   └── utils/
│       ├── data_loader.py        # 数据加载工具
│       └── visualization.py      # 可视化工具
├── notebooks/              # Jupyter 实验笔记本
├── docs/                   # 文档
└── tests/                  # 单元测试
```

## 数据集

基于 ESA OPS-SAT 遥测数据集：
- `segments.csv`：303,493 行 × 8 列，2,123 个 segment，9 个遥测通道
- 合成特征集：2,123 行 × 23 列，18 个特征 + 5 个元数据列
- 异常比例：20.4%（正常 1689 / 异常 434）

## 快速开始

```bash
# 进入项目目录
cd microsat-anomaly-analysis

# 激活虚拟环境
.venv\Scripts\activate

# 安装依赖
poetry install

# 复制原始数据到 data/raw/
# (从 ESA OPS-SAT-AD 数据集目录复制)

# 运行基准实验
poetry run python src/models/iforest_baseline.py
```

## 团队

| 成员 | 年级 | 职责 |
|------|------|------|
| 李冰杰 | 23级 | 核心算法 + RAG + 系统架构 |
| 王翌钧 | 25级 | 数据处理 + 可视化 + 前端 |
| 陶诗怡 | 25级 | 知识库建设 + 成果展示 + 文档 |

## 开题冲刺

- **Deadline**：2026-05-10
- **当前阶段**：Week 1 — IF baseline + 滑动窗口特征
