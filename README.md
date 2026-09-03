# 微小卫星遥测智能异常检测与RAG解释辅助系统

> Shanghai Dianji University — 大学生创新创业训练计划项目
> 周期：2026.4 - 2027.3 | 指导教师：芦立华

---

## 📋 目录

- [项目概述](#项目概述)
- [团队成员](#团队成员)
- [项目结构](#项目结构)
- [数据集说明](#数据集说明)
- [技术架构](#技术架构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [核心模块详解](#核心模块详解)
- [配置文件说明](#配置文件说明)
- [前端开发](#前端开发)
- [实验与评估](#实验与评估)
- [常见问题](#常见问题)
- [参考资料](#参考资料)

---

## 🚀 队友部署指南（分发版 · 必读）

> 本仓库**工作分支是 `main`**。**克隆务必加 `-b main`**，否则可能拿到旧分支的空项目。

### 前置要求
- **NVIDIA 显卡 + CUDA 11.8+**（本项目锁死 `faiss-gpu`，无 GPU 无法安装/运行；向量检索走 GPU）
- **Python 3.11 / 3.12**（推荐；faiss-gpu 官方 wheel 齐全，3.13 需自行解决安装）
- **Node.js 18+**（仅前端开发需要）
- Git

### 第 1 步：克隆（必须 `-b main`）
```bash
git clone -b main https://github.com/Jerry518520/microsat-anomaly-analysis.git
cd microsat-anomaly-analysis
```

### 第 2 步：Python 依赖
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # Linux / macOS
pip install -r requirements.txt
```

### 第 3 步：解压数据包（已随仓库分发，无需额外索取）
仓库根目录的 `data_share.zip`（约 22MB）包含运行所需的全部数据：
`data/raw/segments.csv` + `data/results/*.json` + `data/vectorstore/*`。
在**项目根目录**解压即可（解压后应与 `src/`、`frontend/` 同级出现 `data/`）：
```bash
# Windows：右键解压，或 PowerShell
Expand-Archive data_share.zip -DestinationPath .
# Linux / macOS
unzip data_share.zip
```

### 第 4 步：配置环境变量
```bash
cp .env.example .env        # 然后编辑 .env
```
- `VOLCENGINE_API_KEY`：**RAG 问答接口必需**；若只做看板/检测/波形联调可暂时留空。
- `EMBEDDING_MODEL_PATH`：选填。留空则首次调用 RAG 时自动从 HuggingFace 镜像（hf-mirror.com）下载 `BAAI/bge-m3`；若已拿到本地模型，设为 `models/Xorbits/bge-m3`。

### 第 5 步：启动
**Windows（推荐，一键启动后端+前端）：**
```bash
python scripts/start_ui.py
```
**Linux / macOS（`start_ui.py` 为 Windows 专用，需手动启动）：**
```bash
# 终端 1：后端
.venv/bin/activate
uvicorn src.api.main:app --port 8000 --host 127.0.0.1
# 终端 2：前端
cd frontend && npm install && npm run dev
```
启动后访问：
- 前端界面：http://localhost:5180
- 后端 API 文档：http://localhost:8000/docs

### 接口对齐（前端开发）
- 接口契约见 **`API.md`**（/api/health、/api/dashboard、/api/detection、/api/explanation）。
- 对接清单与分工见 **`HANDOFF.md`**。
- **除 `/api/explanation/query`（RAG 问答）需要 4.4GB 嵌入模型外，其余接口仅需 `data_share.zip` 数据即可联调**，无需模型权重。

---

## 项目概述

本项目基于 **ESA OPS-SAT 卫星遥测数据集**，实现了一套完整的微小卫星遥测异常检测与智能解释系统：

1. **Isolation Forest 异常检测**：分通道自适应段级特征检测
2. **特征贡献度分析**：解释哪些特征驱动异常判定
3. **RAG 辅助解释**：结合卫星手册知识库，用 LLM 生成异常原因的自然语言解释
4. **Web 可视化平台**：FastAPI + React 前后端分离架构，集成检测+解释+可视化

### 核心功能

| 功能模块 | 说明 | 状态 |
|---------|------|------|
| 异常检测引擎 | Isolation Forest 分通道检测 | ✅ 完成 |
| 特征工程 | 18维段级特征提取 | ✅ 完成 |
| RAG 解释系统 | FAISS + BGE-M3 + DeepSeek | ✅ 完成 |
| FastAPI 后端 | REST API 接口 | ✅ 完成 |
| React 前端 | 可视化诊断界面 | ✅ 完成 |
| Streamlit 旧版 | 早期原型 (已弃用) | ⚠️ 维护模式 |

---

## 团队成员

| 成员 | 年级 | 职责 | 联系方式 |
|------|------|------|----------|
| 李冰杰 | 23级 | 核心算法 + RAG + 系统架构 | - |
| 王翌钧 | 25级 | 数据处理 + 可视化 + 前端 | - |
| 陶诗怡 | 25级 | 知识库建设 + 成果展示 + 文档 | - |

---

## 项目结构

```
microsat-anomaly-analysis/
├── .env                          # 环境变量 (API Key 等，不提交 git)
├── .gitignore                    # Git 忽略规则
├── .streamlit/                   # Streamlit 配置
│   └── config.toml
├── configs/                      # 配置文件
│   ├── config.yaml               # 全局参数配置
│   ├── rag_config.yaml           # RAG 系统配置
│   └── run_config.json           # 运行时配置
├── data/                         # 数据目录
│   ├── raw/                      # 原始数据 (segments.csv, 合成特征.csv)
│   ├── processed/                # 处理后的特征数据
│   ├── results/                  # 实验结果 (JSON, 图表)
│   ├── vectorstore/              # FAISS 向量索引
│   └── embeddings_cache/         # 嵌入向量缓存
├── docs/                         # 文档
│   ├── knowledge_base/           # RAG 知识库源文件 (PDF/MD)
│   ├── paper/                    # 论文相关
│   ├── papers/                   # 参考文献
│   └── *.md                      # 项目文档
├── frontend/                     # React 前端
│   ├── src/                      # 前端源码
│   │   ├── App.tsx               # 主应用 (单文件，含所有页面)
│   │   ├── main.tsx              # 入口
│   │   └── index.css             # 全局样式
│   ├── package.json              # Node.js 依赖
│   ├── vite.config.ts            # Vite 配置
│   └── tsconfig.json             # TypeScript 配置
├── models/                       # 预训练模型 (不提交 git)
│   └── Xorbits/bge-m3/           # BGE-M3 嵌入模型
├── notebooks/                    # Jupyter 实验笔记本
├── scripts/                      # 脚本
│   ├── build_index.py            # 构建 FAISS 索引
│   ├── run_rag_experiment.py     # RAG 实验脚本
│   ├── start_server.bat          # Windows 启动脚本
│   ├── start_ui.py               # 一键启动前后端
│   └── gen_ppt_diagrams.py       # 生成 PPT 图表
├── src/                          # Python 源码
│   ├── api/                      # FastAPI 后端
│   │   ├── main.py               # FastAPI 入口
│   │   ├── models.py             # 数据模型
│   │   └── routes/               # API 路由
│   │       ├── dashboard.py      # 仪表盘 API
│   │       ├── detection.py      # 检测 API
│   │       └── explanation.py    # 解释 API
│   ├── evaluation/               # 评估模块
│   │   ├── metrics.py            # F1, AUC_ROC 等指标
│   │   └── feature_importance.py # 特征贡献度分析
│   ├── experiments/              # 实验脚本
│   │   ├── baseline_rerun.py     # 基准实验
│   │   └── fair_comparison.py    # 公平对比实验
│   ├── features/                 # 特征工程
│   │   ├── segment_features.py   # 段级特征 (baseline)
│   │   └── extract_segments_18d.py # 18维特征提取
│   ├── integration/              # 集成模块
│   │   └── anomaly_rag_pipeline.py # 异常检测+RAG完整流程
│   ├── models/                   # 模型
│   │   └── iforest_baseline.py   # IForest 基准模型
│   ├── rag/                      # RAG 解释模块
│   │   ├── embedding.py          # BGE-M3 嵌入
│   │   ├── llm_client.py         # LLM API 客户端
│   │   ├── pipeline.py           # RAG 流程
│   │   ├── prompts.py            # Prompt 模板
│   │   └── vectorstore.py        # FAISS 向量库
│   ├── ui/                       # Streamlit UI (旧版)
│   │   └── app.py                # Streamlit 入口
│   └── utils/                    # 工具模块
│       ├── constants.py          # 常量定义
│       ├── data_loader.py        # 数据加载
│       ├── results_loader.py     # 结果加载
│       ├── stat_rules.py         # 统计规则
│       └── visualization.py      # 可视化工具
├── tests/                        # 测试
│   ├── test_rag/                 # RAG 模块测试
│   └── output/                   # 测试输出 (gitignore)
├── pyproject.toml                # Poetry 项目配置
├── requirements.txt              # pip 依赖
├── poetry.lock                   # Poetry 锁定文件
├── README.md                     # 本文件
└── DEPLOY.md                     # 部署指南
```

---

## 数据集说明

### 数据来源

基于 **ESA OPS-SAT 遥测异常检测数据集** (OPS-SAT-AD)：
- 论文: Ruszczak et al., "Scientific Data", 2025
- 下载: [GitHub - OPS-SAT-AD](https://github.com/ops-sat-ad)

### 数据文件

| 文件 | 说明 | 位置 |
|------|------|------|
| `segments.csv` | 303,493 行 × 8 列，2,123 个 segment | `data/raw/` |
| `dataset-*.csv` | 2,123 行 × 23 列，18 个特征 + 5 个元数据列 | `data/raw/` |

### 数据特征

- **遥测通道**: 9 个 (CADC0872-CADC0894)
- **异常比例**: 20.4% (正常 1689 / 异常 434)
- **特征维度**: 18 维段级特征

### 通道说明

| 通道 ID | 物理含义 | 类型 |
|---------|----------|------|
| CADC0872 | Magnetometer X-axis (磁力计X轴) | 强通道 |
| CADC0873 | Magnetometer Y-axis (磁力计Y轴) | 强通道 |
| CADC0874 | Magnetometer Z-axis (磁力计Z轴) | 强通道 |
| CADC0884 | Photodiode 1 (光电二极管1) | 弱通道 |
| CADC0886 | Photodiode 2 (光电二极管2) | 弱通道 |
| CADC0888 | Photodiode 3 (光电二极管3) | 弱通道 |
| CADC0890 | Photodiode 4 (光电二极管4) | 弱通道 |
| CADC0892 | Photodiode 5 (光电二极管5) | 弱通道 |
| CADC0894 | Photodiode 6 (光电二极管6) | 弱通道 |

---

## 技术架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    前端 (React + TypeScript)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Dashboard │  │ Detection│  │Explainer │  │ Settings │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    后端 (FastAPI)                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  /api/dashboard  /api/detection  /api/explanation    │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                核心业务逻辑                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ IForest     │  │ Feature     │  │ RAG         │  │   │
│  │  │ Detection   │  │ Engineering │  │ Explanation │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    数据层                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ FAISS       │  │ BGE-M3      │  │ LLM API     │        │
│  │ VectorStore │  │ Embedding   │  │ (DeepSeek)  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| **前端** | React 19 + TypeScript + Vite + Tailwind CSS | 单页应用 |
| **后端** | FastAPI + Uvicorn | REST API |
| **异常检测** | scikit-learn (Isolation Forest) | 机器学习 |
| **特征工程** | pandas + numpy + 自定义提取器 | 18维特征 |
| **嵌入模型** | BGE-M3 (sentence-transformers) | 向量化 |
| **向量库** | FAISS (faiss-gpu, 需 CUDA) | 相似度检索 |
| **LLM** | DeepSeek V3 (火山引擎 Ark) | 文本生成 |
| **RAG 框架** | LangChain | 流程编排 |
| **旧版 UI** | Streamlit | 已弃用 |

---

## 环境配置

### 系统要求

- **Python**: 3.11 / 3.12（推荐；faiss-gpu 官方 wheel 齐全，3.13 需自行解决 faiss-gpu 安装）
- **Node.js**: 18+ (前端开发)
- **CUDA**: 11.8+（**必需**，本项目使用 faiss-gpu，无 NVIDIA/CUDA 无法安装或运行）
- **操作系统**: Windows 10/11, Linux, macOS

### Python 环境

#### 方式一: Poetry (推荐)

```bash
# 安装 Poetry
pip install poetry

# 安装依赖
poetry install

# 激活虚拟环境
poetry shell
```

#### 方式二: pip + venv

```bash
# 创建虚拟环境
python -m venv .venv

# 激活 (Windows)
.venv\Scripts\activate

# 激活 (Linux/Mac)
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### Node.js 环境 (前端)

```bash
cd frontend
npm install
```

### 环境变量

参考仓库根目录的 `.env.example` 创建 `.env` 文件：

```env
# 火山引擎 API Key（RAG 问答必需；仅做看板/检测可留空）
VOLCENGINE_API_KEY=your_api_key_here

# 可选：HuggingFace 镜像（加速 BGE-M3 下载）
HF_ENDPOINT=https://hf-mirror.com

# 可选：本地嵌入模型路径。留空则使用 BAAI/bge-m3 自动下载；
#       若已拿到本地模型则设为 models/Xorbits/bge-m3
EMBEDDING_MODEL_PATH=
```

### 模型下载

BGE-M3 嵌入模型需要提前下载到 `models/` 目录：

```bash
# 方式一: 使用 ModelScope (国内推荐)
pip install modelscope
modelscope download --model Xorbits/bge-m3 --local_dir models/Xorbits/bge-m3

# 方式二: 使用 HuggingFace
# 需要配置 HF_ENDPOINT 或使用代理
```

---

## 快速开始

### 1. 启动完整系统 (推荐)

```bash
# 一键启动 FastAPI + React
python scripts/start_ui.py

# 或分别启动
python scripts/start_ui.py --api    # 仅启动后端
python scripts/start_ui.py --web    # 仅启动前端
```

启动后访问:
- **前端**: http://localhost:5180
- **后端 API**: http://localhost:8000/docs
- **旧版 UI**: http://localhost:8501 (Streamlit)

### 2. 构建 FAISS 索引

首次使用 RAG 功能前需要构建索引：

```bash
python scripts/build_index.py
```

### 3. 运行异常检测实验

```bash
python src/models/iforest_baseline.py
```

### 4. 运行 RAG 实验

```bash
python scripts/run_rag_experiment.py
```

---

## 核心模块详解

### 1. 异常检测模块 (`src/models/`)

**Isolation Forest 基准模型**:
- 文件: `iforest_baseline.py`
- 功能: 分通道自适应异常检测
- 参数: `contamination=0.2`, `n_estimators=100`

**使用方式**:
```python
from src.models.iforest_baseline import IForestBaseline

model = IForestBaseline(contamination=0.2)
model.fit(features)
predictions = model.predict(test_features)
```

### 2. 特征工程模块 (`src/features/`)

**18维段级特征**:
- 文件: `extract_segments_18d.py`
- 特征包括: 均值、标准差、偏度、峰度、最大值、最小值、中位数、四分位距等

**使用方式**:
```python
from src.features.extract_segments_18d import extract_features

features_df = extract_features(segments_df)
```

### 3. RAG 解释模块 (`src/rag/`)

**组件**:
- `embedding.py`: BGE-M3 嵌入模型封装
- `vectorstore.py`: FAISS 向量库封装
- `llm_client.py`: DeepSeek API 客户端
- `prompts.py`: Prompt 模板
- `pipeline.py`: RAG 完整流程

**使用方式**:
```python
from src.rag.embedding import BGE_M3_Embedder
from src.rag.vectorstore import FAISSVectorStore
from src.rag.llm_client import LLMClient
from src.rag.prompts import PromptTemplates

# 初始化组件
embedder = BGE_M3_Embedder()
vectorstore = FAISSVectorStore(embedder=embedder)
llm = LLMClient()
prompts = PromptTemplates()

# 检索
results = vectorstore.search("磁力计异常", top_k=5)

# 生成解释
prompt_data = prompts.get_anomaly_analysis_prompt(
    channel_id="CADC0872",
    anomaly_type="unusual_shapes",
    anomaly_description="磁力计X轴异常",
    context=context
)
answer = llm.generate(prompt=prompt_data["user"], system_prompt=prompt_data["system"])
```

### 4. FastAPI 后端 (`src/api/`)

**API 端点**:
- `GET /api/health`: 健康检查
- `GET /api/dashboard/*`: 仪表盘数据
- `POST /api/detection/*`: 异常检测
- `POST /api/explanation/*`: RAG 解释

**启动方式**:
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 5. React 前端 (`frontend/`)

**页面**:
- Dashboard: 总览仪表盘
- Detection: 异常检测详情
- Explanation: RAG 解释界面

**开发模式**:
```bash
cd frontend
npm run dev
```

---

## 配置文件说明

### configs/config.yaml

全局参数配置：

```yaml
# 数据路径
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  results_dir: "data/results"
  segments_file: "segments.csv"
  features_file: "dataset-提取的合成特征被计算到每个手动分割和标记的遥测段上.csv"

# 模型参数
model:
  iforest:
    n_estimators: 100
    contamination: 0.2
    random_state: 42

# 评估参数
evaluation:
  metrics: [f1, auc_roc, accuracy, precision, recall]
  per_channel: true
```

### configs/rag_config.yaml

RAG 系统配置：

```yaml
# 嵌入模型
embedding:
  model_name: "BAAI/bge-m3"  # 默认从 HuggingFace 镜像自动下载；可用环境变量 EMBEDDING_MODEL_PATH 覆盖为本地路径
  device: "cuda"
  batch_size: 32
  max_length: 512

# 向量库
vectorstore:
  type: "faiss"
  persist_directory: "data/vectorstore"

# 检索参数
retrieval:
  top_k: 5
  score_threshold: 0.3
  max_tokens: 2000

# LLM 配置
llm:
  provider: "volcengine"
  model: "deepseek-v3-2-251201"
  temperature: 0.1
  max_tokens: 2048
```

---

## 前端开发

### 技术栈

- **React 19**: UI 框架
- **TypeScript**: 类型安全
- **Vite**: 构建工具
- **Tailwind CSS**: 样式框架
- **Plotly.js**: 图表库
- **Framer Motion**: 动画库

### 开发命令

```bash
cd frontend

# 开发模式
npm run dev

# 构建生产版本
npm run build

# 预览生产版本
npm run preview

# 代码检查
npm run lint
```

### 目录结构

```
frontend/
├── src/
│   ├── App.tsx          # 主应用 (所有页面和组件)
│   ├── main.tsx         # 入口文件
│   ├── index.css        # 全局样式
│   └── react-plotly.d.ts # Plotly 类型声明
├── public/              # 静态资源
├── package.json         # 依赖配置
├── vite.config.ts       # Vite 配置
└── tsconfig.json        # TypeScript 配置
```

---

## 实验与评估

### 评估指标

| 指标 | 说明 | 位置 |
|------|------|------|
| F1 Score | 精确率和召回率的调和平均 | `src/evaluation/metrics.py` |
| AUC-ROC | ROC 曲线下面积 | `src/evaluation/metrics.py` |
| Per-channel F1 | 分通道 F1 分数 | `src/evaluation/metrics.py` |
| Feature Importance | 特征贡献度 | `src/evaluation/feature_importance.py` |

### 运行实验

```bash
# 基准实验
python src/experiments/baseline_rerun.py

# 公平对比实验
python src/experiments/fair_comparison.py

# RAG 实验
python scripts/run_rag_experiment.py
```

### 实验结果

实验结果保存在 `data/results/` 目录：
- `anomaly_rag_results.json`: RAG 完整结果
- `rag_retrieval_stats.json`: 检索统计
- `fair_comparison_results.json`: 对比实验结果

---

## 常见问题

### Q1: 没有 NVIDIA / CUDA 能用吗？

本项目的向量检索依赖 `faiss-gpu`，**必须有 NVIDIA 显卡和 CUDA 环境**才能安装运行。若机器无 GPU：
- 看板/检测/波形接口仍可读 `data_share.zip` 中的数据并联调前端；
- 但 RAG 问答（`/api/explanation/query`）与嵌入计算需要 GPU，无法在无卡环境运行。

如确需 CPU 版兜底，可把 `requirements.txt` 中的 `faiss-gpu` 换成 `faiss-cpu`，并将 `rag_config.yaml` 的 `embedding.device` 改为 `cpu`（属非官方方案，未经充分测试）。

### Q2: FAISS 索引构建失败？

```bash
# 确保已下载 BGE-M3 模型
ls models/Xorbits/bge-m3/

# 重新构建索引
python scripts/build_index.py
```

### Q3: LLM API 调用失败？

```bash
# 检查 .env 文件
cat .env

# 确认 API Key 已设置
echo $VOLCENGINE_API_KEY

# 测试 API 连接
python -c "from src.rag.llm_client import LLMClient; llm = LLMClient(); print('OK')"
```

### Q4: 前端启动失败？

```bash
cd frontend

# 清理缓存
rm -rf node_modules .vite

# 重新安装
npm install

# 启动
npm run dev
```

### Q5: Streamlit PyArrow 崩溃？

这是已知问题，已通过以下方式修复：
- 所有 DataFrame 列转换为 `str()` 类型
- 使用 `st.dataframe()` 替代 `st.table()`

---

## 参考资料

### 论文与数据集

- [OPS-SAT-AD 数据集](https://github.com/ops-sat-ad)
- [Ruszczak et al., "Scientific Data", 2025](docs/papers/！Ruszczak_OPS-SAT_AD_ScientificData_2025.pdf)
- [IForest 原始论文](docs/papers/IForest_Liu_Ting_Zhou_ICDM2008.pdf)

### 技术文档

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [React 文档](https://react.dev/)
- [FAISS 文档](https://faiss.ai/)
- [LangChain 文档](https://python.langchain.com/)
- [BGE-M3 模型](https://huggingface.co/BAAI/bge-m3)

### 项目文档

- [设计文档](docs/design_proposal.md)
- [部署指南](DEPLOY.md)
- [UI 设计](docs/streamlit_ui_design.md)
- [参考文献](docs/REFERENCES.md)

---

## Git 工作流

### 分支策略

- `main`: 主分支，稳定版本
- `dev`: 开发分支，日常开发
- `feature/*`: 功能分支

### 提交规范

```
feat: 新功能
fix: 修复 bug
docs: 文档更新
style: 代码格式
refactor: 重构
test: 测试
chore: 构建/工具
```

### 常用命令

```bash
# 查看状态
git status

# 提交更改
git add .
git commit -m "feat: 添加新功能"

# 推送
git push origin dev

# 合并到 main
git checkout main
git merge dev
```

---

## 更新日志

### 2026-06-23
- 整理项目文件结构
- 更新 README 文档

### 2026-06-20
- 全量代码审查修复
- 前端重写为专业监控风格

### 2026-06-03
- DashboardView 自动轮询
- 通道波形矩阵移植

### 2026-05-27
- RAG 全链路优化
- Streamlit UI 重写

---

## 许可证

本项目为上海电机学院大学生创新创业训练计划项目，仅供学术研究使用。

---

## 联系方式

如有问题，请联系项目负责人或指导教师。
