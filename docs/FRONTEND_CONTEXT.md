# 前端设计上下文包（Frontend Design Context Pack）

> **用途**：本文档供「调用高级大模型**从零重写**前端」的队友直接粘贴进对话使用。
> 它包含三层：① API 契约（端点+响应字段+样例）② 数据字典/领域语义 ③ 设计简报。
> 配合仓库内的 `.codegraph/`（代码知识图谱）一起用效果最佳，见文末第 7 节。

---

## 0. 给高级大模型的一句话指令（可直接粘贴）

你是来**设计 / 从零重写**「OPS-SAT 卫星遥测异常诊断系统」前端大屏的。后端 FastAPI 已完整就绪，提供稳定的 REST 接口；你负责生成一套全新的 `frontend/` 代码，可完全丢弃现有实现。

**硬性约束**（这些是「锚点」，重写时不可违背；其余结构你自由发挥）：
- **可全量重写** `frontend/`：可丢弃现有 `App.tsx`、组件与 fetch 封装，从零生成全新代码（第 6 节仅作现状参考，不必复用）。
- 技术栈必须沿用：React 19 + Vite + TypeScript + Tailwind CSS 4 + Plotly.js（见第 2 节）。
- 所有数据来自第 3 节的接口，**严格对齐 API.md 契约**（端点 URL、字段名、JSON 结构都不可改）；不要在前端硬编码任何业务阈值（阈值以后端返回为准）。

---

## 1. 系统背景与领域语义（让设计"有意义"）

- **系统**：基于 ESA OPS-SAT 卫星 18 天遥测数据的异常诊断系统。
- **检测流程（三级）**：
  1. Stage 0 全局 Isolation Forest（contamination=0.2）
  2. Stage 1 分通道独立 Isolation Forest
  3. Stage 2 IF + 规则融合
- **解释**：RAG（知识库用 BGE-M3 向量化 + FAISS 索引 + 大模型生成自然语言诊断）。
- **基线指标（用于大屏展示，请勿当作实时值）**：
  - F1：Stage0=0.2996 → Stage1=0.5381 → Stage2=0.5683
  - 最佳 SegF1 = 0.5683
- **大屏目的**：实时监控告警中心 + 单异常深度诊断 + RAG 解释可视化。

---

## 2. 技术栈（必须沿用，勿引入冲突依赖）

| 层 | 技术 | 版本 |
|---|---|---|
| 框架 | React | 19.2.x |
| 构建 | Vite | 8.x |
| 语言 | TypeScript | 6.x |
| 样式 | Tailwind CSS | 4.x（`@tailwindcss/vite`） |
| 图表 | Plotly.js（`plotly.js-dist`） | 2.35.x |
| 图标 | lucide-react | 1.17.x |
| Markdown | react-markdown | 10.x |

- 后端：FastAPI 2.0.0，运行于 `http://localhost:8000`。
- 开发服务器：Vite 默认 `http://localhost:5173` 或 `5180`；CORS 已放开这两个源。
- 图表库为 Plotly.js；现有 `OscilloscopePlotly`/`ChannelSparkline`/`Plot` 仅是参考实现，重写时可重新设计封装，但图表渲染必须基于 Plotly。

---

## 3. API 契约（Layer A — 最重要）

> 本节契约的**正本**为仓库根目录 `API.md`（人类可读接口文档）。本文档是其面向 LLM 的派生视图；若字段描述与 `API.md` 冲突，以 `API.md` 为准并同步回此处。

**Base URL**：
- 开发态：使用**相对路径** `/api/...`（Vite dev server 已代理到 `:8000`）。
- 生产态：绝对 `http://localhost:8000/api/...`（或部署域名）。

**路由分组**：
- `/api/dashboard/*` —— 实时告警中心
- `/api/detection/*` —— 算法/实验数据
- `/api/explanation/*` —— 单异常深度诊断 + RAG 问答
- `/api/health` —— 健康检查

### 3.1 GET `/api/dashboard/metrics`
核心指标卡数据。响应（字段来自 `CoreMetrics`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `anomaly_rate` | float | 异常行占比（0~1）——**点级口径**，分母为采样点数 |
| `anomaly_rate_str` | str | 形如 `"12.3%"`——**点级口径** |
| `anomaly_rows` | int | 异常行数（点级） |
| `total_rows` | int | 总行数（采样点数，非段数） |
| `fault_count` | int | 异常分 > 0.05 的故障数 |
| `total_anomalies` | int | RAG 结果中的异常条目数 |
| `best_f1` | float | 最佳 SegF1（实测 = best_seg_f1 = 0.5683，取自 subsampling_sweep_1.9_results.json） |
| `throughput` | int | 处理行数 |
| `channel_count` | int | 固定 9 |

示例：
```json
{
  "anomaly_rate": 0.3304,
  "anomaly_rate_str": "33.0%（点级）",
  "anomaly_rows": 100264,
  "total_rows": 303493,
  "fault_count": 200,
  "total_anomalies": 200,
  "best_f1": 0.5683,
  "throughput": 303493,
  "channel_count": 9
}
> **口径警告（重要）**：`anomaly_rate` / `anomaly_rate_str` / `anomaly_rows` / `total_rows` 均为**点级（行级）口径**，分母是遥测采样点数（303,493），不是段数。项目另有一套**段级口径**异常率 **20.44%（434 / 2123 段）**（论文主表口径），两者分母不同、不可混用。注意 `best_f1` 是**段级** F1，与上述点级字段口径不同。完整定义见 `docs/指标口径说明.md`。
>
> 注：以上为 2026-07-30 实际计算值（来源 `data/raw/segments.csv` + `data/results/anomaly_rag_results.json`）：total_rows=303493、anomaly_rows=100264、anomaly_rate=33.0%（点级）、fault_count=total_anomalies=200（anomaly_score>0.05 的 RAG 条目数）、best_f1=0.5683（=best_seg_f1，段级）。
```

### 3.2 GET `/api/dashboard/channels`
9 个遥测通道的 sparkline + 状态。响应：`{ "channels": [ChannelInfo] }`。

`ChannelInfo`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `channel` | str | 通道代码（见第 4 节） |
| `label` | str | 物理含义中文标签 |
| `anomaly_rate` | float | 该通道异常占比 |
| `status` | str | `nominal`\|`caution`\|`warning`\|`critical` |
| `sparkline_values` | float[] | 最近 500 点遥测值 |
| `anomaly_indices` | int[] | 异常点在窗口内的相对下标 |

示例（单通道）：
```json
{
  "channel": "CADC0872",
  "label": "磁力计 X轴",
  "anomaly_rate": 0.08,
  "status": "caution",
  "sparkline_values": [0.12, 0.15, 0.11, 0.9, 0.13],
  "anomaly_indices": [3]
}
```

### 3.3 GET `/api/dashboard/alerts`
故障告警队列（异常分 > 0.05，按分数降序）。响应：`{ "alerts": [AlertItem], "total": int }`。

`AlertItem`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `segment` | str | 异常段 ID |
| `channel` | str | 通道代码 |
| `channel_label` | str | 通道中文标签 |
| `anomaly_score` | float | 异常分（0~1） |
| `severity` | str | `nominal`\|`caution`\|`warning`\|`critical` |
| `summary` | str | RAG 解释前 120 字摘要 |
| `anomaly_type` | str\|null | 异常类型（可能为 `"未知"`） |

### 3.4 GET `/api/dashboard/status`
系统健康。响应：`{ "segments_available": bool, "rag_available": bool, "rag_error": str|null }`。
（`utc_time` 在模型里声明但未在代码中返回，前端不要依赖它。）

### 3.5 GET `/api/detection/rag-config`
RAG 知识库状态。响应（`RagConfig`）：`status`(`online`|`offline`)、`doc_count`({pdf,md,html})、`chunk_count`(int|null)、`embedding_model`(str, 默认 `"BGE-M3"`)。

### 3.6 GET `/api/detection/experiments`
算法演进 F1 Benchmark。响应：`{ "experiments": [ExperimentStage] }`。
`ExperimentStage`：`version`(str)、`strategy`(str)、`f1`(float)、`improvement`(str, 如 `"+89.7%"`)。
默认数据：Stage0=0.2996(+—)、Stage1=0.5381(+79.6%)、Stage2=0.5683(+89.7%)。

### 3.7 GET `/api/detection/channel-f1`
分通道灵敏度。响应：`{ "channels": [ChannelF1], "available": bool }`。
`ChannelF1`：`channel`(str)、`f1`(float)、`precision`(float|null)、`recall`(float|null)。

### 3.8 GET `/api/detection/system-params`
算法系统参数。响应：`{ "algorithm": "Isolation Forest", "dimensions": 18, "contamination": 0.2 }`。

### 3.9 GET `/api/explanation/detail?segment={seg}&channel={ch}`
单异常完整诊断。响应字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `segment` / `channel` | str | 标识 |
| `channel_label` | str | 通道标签 |
| `anomaly_type` | str | 异常类型 |
| `anomaly_score` | float | 异常分 |
| `urgency` | str | 中文紧急度：低/中/高/紧急 |
| `urgency_level` | str | 英文：low/medium/high/critical |
| `sections` | dict | 解析后的 RAG 段落，键：`可能原因`/`影响评估`/`结论`/`建议措施`/`来源` |
| `garbled` | bool | 是否为乱码/失败文本（前端应提示"解释不可用"） |
| `feature_summary` | dict\|null | 特征摘要 |
| `sources` | list | `[{filename, page, score, local_path}]` 知识来源 |
| `raw_explanation` | str | 原始解释全文（非乱码时） |

### 3.10 GET `/api/explanation/waveform?channel={ch}&segment={seg?}&context_size={200}`
遥测波形（异常段上下文窗口）。响应：

| 字段 | 类型 | 说明 |
|---|---|---|
| `timestamps` | list | 时间轴 |
| `values` | list | 遥测值 |
| `anomaly_mask` | list[int] | 0/1 异常标记 |
| `mean` / `std` | float | 均值/标准差 |
| `upper_3sigma` / `lower_3sigma` | float | 3σ 上下界（画阈值带用） |
| `channel` / `channel_label` | str | 通道信息 |

### 3.11 POST `/api/explanation/query`
RAG 自由问答。请求体：`{ "query": str, "channel": str, "segment": str, "context": {} }`。
响应：`{ "answer": str, "sources": [{ "filename": str, "score": float }] }`；出错时 `{ "error": str, "answer": "RAG 引擎暂不可用。" }`。

### 3.12 GET `/api/health`
`{ "status": "ok", "version": "2.0.0" }`。

---

## 4. 数据字典 / 领域语义（Layer B — 让数字"看得懂"）

### 4.1 9 个遥测通道（`CHANNEL_MAP`）

| 代码 | 物理含义 | 备注 |
|---|---|---|
| `CADC0872` | 磁力计 X轴 | |
| `CADC0873` | 磁力计 Y轴 | |
| `CADC0874` | 磁力计 Z轴 | |
| `CADC0884` | 光电二极管 1 | **无异常通道**（数据集中恒为正常，severity 永远 `nominal`） |
| `CADC0886` | 光电二极管 2 | |
| `CADC0888` | 光电二极管 3 | |
| `CADC0890` | 光电二极管 4 | |
| `CADC0892` | 光电二极管 5 | |
| `CADC0894` | 光电二极管 6 | |

### 4.2 状态/严重度枚举与阈值

- `status`（通道级，按异常占比 `rate`）：
  - `rate > 0.4` → `critical`；`> 0.15` → `warning`；`> 0` → `caution`；否则 `nominal`。
  - `CADC0884` 永远 `nominal`。
- `severity`（告警级，按 `anomaly_score`）：
  - `> 0.4` → `critical`；`> 0.15` → `warning`；`> FAULT_THRESHOLD(0.05)` → `caution`；否则 `nominal`。
- `FAULT_THRESHOLD = 0.05`：告警入队门槛（异常分 ≤ 0.05 不进告警队列）。
- 推荐配色：nominal=绿，caution=黄，warning=橙，critical=红。

### 4.3 RAG 解释段落结构（`sections` 的键）
`可能原因` / `影响评估` / `结论` / `建议措施` / `来源`。
前端渲染时按此顺序用 `react-markdown` 渲染各段；`garbled=true` 时显示兜底提示。

### 4.4 紧急度映射
`低→low`、`中→medium`、`高→high`、`紧急→critical`（字段 `urgency_level` 已是英文，可直接用）。

---

## 5. 设计简报（Layer C — 大屏该长什么样）

### 5.1 多视图结构（沿用现有 `currentPage` 切换）
- **DashboardView**（实时告警中心）：核心指标卡 + 9 通道状态网格(带 sparkline) + 告警队列。
- **DetectionView**（算法/实验）：系统参数 + 实验演进 F1 曲线 + 分通道 F1 条形 + RAG 配置状态。
- **ExplanationView**（单异常深度诊断）：波形示波器(3σ 带) + RAG 解释卡片(分 section, markdown) + 知识来源列表 + RAG 自由问答框。

### 5.2 必含面板清单
1. 核心指标卡：`anomaly_rate_str` / `best_f1` / `fault_count` / `throughput` / `channel_count`
2. 9 通道状态网格：每格显示 `label` + `status` 色块 + `ChannelSparkline`
3. 告警队列：按 `severity` 排序，点击 → 跳转 ExplanationView 并带 `{channelId, anomalyData}`
4. 异常波形示波器：`OscilloscopePlotly` 用 `values`+`anomaly_mask`+`upper/lower_3sigma` 画阈值带
5. RAG 解释卡片：`sections` 用 `react-markdown` 分块渲染，处理 `garbled`
6. 实验演进 F1 曲线 / 分通道 F1 条形
7. 系统状态条：展示 `rag_available` / `segments_available`

### 5.3 视觉与交互
- 暗色大屏主题（航天监控风），状态色见 4.2。
- 响应式栅格；大屏 16:9 优先。
- 自动刷新：现有已有 `refreshing` 状态，新设计应保留定时刷新（建议 10~30s 轮询 metrics/channels/alerts）。
- 导航：左侧/顶部视图切换（`dashboard` / `detection` / `explanation`）。

---

## 6. 现有前端结构（仅供参考，重写时可丢弃，来自 `.codegraph`）

`frontend/src/App.tsx` 组件树（共 9 个节点）：
```
App
├─ DashboardView      （并发拉 /api/dashboard/{metrics,channels,alerts}）
├─ DetectionView
├─ ExplanationView    （拉 /api/explanation/{waveform,detail}）
├─ OscilloscopePlotly （Plotly 波形封装）
├─ ChannelSparkline   （Plotly 迷你折线）
├─ TerminalComponent
├─ Plot
└─ getSeverity        （severity → 颜色映射工具）
```

**现有 fetch 模式（仅供参考）**：
- DashboardView 用 `Promise.all([fetch('/api/dashboard/metrics'), fetch('/api/dashboard/channels'), fetch('/api/dashboard/alerts')])`（见 App.tsx ~L348-350）。
- ExplanationView 用 `Promise.all([fetch(\`/api/explanation/waveform?channel=${id}&segment=${seg}\`), fetch(\`/api/explanation/detail?channel=${id}&segment=${seg}\`)])`（见 ~L616-617）。
- 连接失败提示：`无法连接后端服务 (localhost:8000)，请确认已运行 python start_ui.py`。
- 状态管理：`useState` / `useEffect`；`currentPage` 控制视图；`targetContext` 在视图间传递 `{channelId, anomalyData}`。

**重写时**：你可自由重新组织组件与 HTTP 请求方式（可自建轻量 fetch 封装），但调用的端点 URL 与字段名必须严格对应第 3 节的 API 契约，不得臆造。

---

## 7. `.codegraph` 使用指引（队友的高级大模型若支持直接检索）

- 仓库已含代码知识图谱：`.codegraph/codegraph.db`（SQLite）+ `docs/_codegraph_data.json`（JSON 导出）。
- 生成于 2026-06-27 ~ 07-01；**API 与前端结构此后未变更，当前有效**（如后续改了 `src/api/*` 或 `frontend/src/*`，请用原工具重新生成）。
- **图谱能力**：`nodes`(785 个代码实体，含签名/docstring/返回类型) / `edges`(1279 条调用关系) / `files`(58 个文件) / 全文检索 `nodes_fts`。
- **推荐检索目标**：
  - API 端点：`SELECT * FROM nodes WHERE file_path LIKE 'src/api/routes/%'` → 拿到 12 个路由 handler 及其 `signature`。
  - 响应模型字段：`SELECT name, signature FROM nodes WHERE file_path='src/api/models.py'`。
  - 前端组件：`SELECT name, kind FROM nodes WHERE file_path='frontend/src/App.tsx'`。
- **限制**：`unresolved_refs`(635) 是第三方库（Plotly/React 等）调用，图谱不解析；运行时真实 JSON 样例以本文档第 3 节为准，不要仅靠图谱推断响应体。

---

## 8. 给队友的速查清单
- [ ] 后端起服务：`python start_ui.py`（监听 :8000），否则前端拉不到数据。
- [ ] 前端起服务：`cd frontend && npm install && npm run dev`（:5173 / :5180）。
- [ ] 设计前先读本文件第 3、4、5 节；让高级大模型据此从零重写 frontend/（第 6 节仅作现状参考，无需复用）。
- [ ] 阈值/通道含义改了 → 同步改 `src/utils/constants.py`，前端只读接口。
- [ ] 验证：DashboardView 三接口、ExplanationView 两接口、RAG 问答框都要能跑通。
