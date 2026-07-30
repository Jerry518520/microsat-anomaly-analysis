# 后端 API 接口文档（前端对接用）

> 本文档给**前端开发同学**看。后台所有接口由 FastAPI 提供（`src/api/main.py`），前端只需按下面的契约对接，无需关心 Python 实现。

---

## 1. 概述

- **后端启动**：`uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
- **前端代理**：React/Vite 开发服务器默认 `:5180`，已在 `frontend/vite.config.ts` 配置代理 `/api` → `http://localhost:8000`，所以前端直接请求 `/api/...` 即可，不用写完整域名。
- **CORS**：已放行 `http://localhost:5180`、`http://localhost:5173`。
- **鉴权**：接口本身**不需要**任何 token。RAG 诊断/问答所需的 `VOLCENGINE_API_KEY` 由**服务端**从环境变量读取（见 `.env.example`），前端不要传 key。

---

## 2. 通道与严重度约定（前端必须对齐）

9 个遥测通道定义在 `src/utils/constants.py` 的 `CHANNEL_MAP`：

| channel | label |
|---|---|
| CADC0872 | 磁力计 X轴 |
| CADC0873 | 磁力计 Y轴 |
| CADC0874 | 磁力计 Z轴 |
| CADC0884 | 光电二极管 1 |
| CADC0886 | 光电二极管 2 |
| CADC0888 | 光电二极管 3 |
| CADC0890 | 光电二极管 4 |
| CADC0892 | 光电二极管 5 |
| CADC0894 | 光电二极管 6 |

**严重度阈值**（来自 `dashboard._severity` / `constants.FAULT_THRESHOLD`）：

| 条件 | severity |
|---|---|
| 通道在 `NO_ANOMALY_CHANNELS`（如 CADC0884） | `nominal` / `offline` |
| anomaly_score > 0.4 | `critical` |
| anomaly_score > 0.15 | `warning` |
| anomaly_score > FAULT_THRESHOLD | `caution` |
| 其他 | `nominal` |

**建议配色**（与 `frontend/src/App.tsx` 一致）：
`nominal #22C55E` / `caution #F97316` / `warning #F59E0B` / `critical #EF4444` / `offline #64748B`

---

## 3. 接口列表

基础前缀：`/api`。所有响应为 JSON。

### 3.1 健康检查
`GET /api/health`
```json
{ "status": "ok", "version": "2.0.0" }
```

### 3.2 Dashboard —— `/api/dashboard`

**GET `/metrics`** — 核心指标
```json
{
  "anomaly_rate": 0.3304,
  "anomaly_rate_str": "33.0%",
  "anomaly_rows": 100264,
  "total_rows": 303493,
  "fault_count": 200,
  "total_anomalies": 200,
  "best_f1": 0.5683,
  "throughput": 303493,
  "channel_count": 9
}
```
> 注：`best_f1` 即 `best_seg_f1`，取自 `data/results/subsampling_sweep_1.9_results.json` = **0.5683**（与 `/detection/experiments` 的 Stage-2 融合 F1 相同）；`0.425` 仅是该文件缺失时的兜底默认值，不是真实值。其余字段为 2026-07-30 实际计算值（来源 `segments.csv` + `anomaly_rag_results.json`）：total_rows=303493、anomaly_rows=100264、anomaly_rate=33.0%、fault_count=total_anomalies=200。

**GET `/channels`** — 各通道 sparkline + 状态（每通道取最后 500 点）
```json
{
  "channels": [
    {
      "channel": "CADC0872",
      "label": "磁力计 X轴",
      "anomaly_rate": 0.0123,
      "status": "nominal",
      "sparkline_values": [0.1, 0.2, 0.15],
      "anomaly_indices": [42, 130]
    }
  ]
}
```

**GET `/alerts`** — 故障告警队列（anomaly_score > FAULT_THRESHOLD 的条目，按分数降序）
```json
{
  "alerts": [
    {
      "segment": "18",
      "channel": "CADC0874",
      "channel_label": "磁力计 Z轴",
      "anomaly_score": 0.87,
      "severity": "critical",
      "summary": "AI 诊断摘要前 120 字...",
      "anomaly_type": "未知"
    }
  ],
  "total": 12
}
```

**GET `/status`** — 系统健康
```json
{ "segments_available": true, "rag_available": true, "rag_error": null }
```

### 3.3 Detection —— `/api/detection`

**GET `/rag-config`**
```json
{ "status": "online", "doc_count": {"pdf":10,"md":3,"html":4}, "chunk_count": 5064, "embedding_model": "BGE-M3" }
```
> `doc_count` 为 `docs/knowledge_base`(pdf4/md1/html1) 与 `docs/papers`(pdf6/md2/html3) 两目录合计；`chunk_count` 部署后由 faiss 索引实际分块数决定（当前知识库约 5064）。
```

**GET `/experiments`** — 实验演进 F1 Benchmark
```json
{
  "experiments": [
    {"version":"Stage 0","strategy":"全局 IForest (c=0.2)","f1":0.2996,"improvement":"—"},
    {"version":"Stage 1","strategy":"分通道独立 IForest","f1":0.5381,"improvement":"+79.6%"},
    {"version":"Stage 2","strategy":"IF + 规则融合","f1":0.5683,"improvement":"+89.7%"}
  ]
}
```

**GET `/channel-f1`** — 分通道灵敏度 F1
```json
{
  "channels": [
    {"channel":"CADC0874","f1":0.62,"precision":0.58,"recall":0.66}
  ],
  "available": true
}
```

**GET `/system-params`**
```json
{ "algorithm":"Isolation Forest", "dimensions":18, "contamination":0.2 }
```

### 3.4 Explanation —— `/api/explanation`

**GET `/detail?segment={seg}&channel={ch}`** — 单异常完整诊断
```json
{
  "segment": "18",
  "channel": "CADC0874",
  "channel_label": "磁力计 Z轴",
  "anomaly_type": "未知",
  "anomaly_score": 0.87,
  "urgency": "高",
  "urgency_level": "high",
  "sections": {
    "可能原因": "...",
    "影响评估": "...",
    "结论": "...",
    "建议措施": "...",
    "来源": "..."
  },
  "garbled": false,
  "feature_summary": "...",
  "sources": [
    {"filename":"！Ruszczak_OPS-SAT_AD_ScientificData_2025.pdf","page":"3","score":0.81,"local_path":"..."}
  ],
  "raw_explanation": "..."
}
```
> 若 `garbled=true`，前端应展示「诊断文本异常，请重试」之类提示，不要直接渲染 `raw_explanation`。

**GET `/waveform?channel={ch}&segment={seg}&context_size=200`** — 异常段上下文波形
```json
{
  "timestamps": [0,1,2],
  "values": [0.1,0.2,0.15],
  "anomaly_mask": [0,1,0],
  "mean": 0.15,
  "std": 0.03,
  "upper_3sigma": 0.24,
  "lower_3sigma": 0.06,
  "channel": "CADC0874",
  "channel_label": "磁力计 Z轴"
}
```

**POST `/query`** — RAG 自由问答
请求体：
```json
{ "query":"这是什么异常？", "channel":"CADC0874", "segment":"18", "context": {} }
```
响应：
```json
{ "answer":"...", "sources":[{"filename":"...","score":0.8}] }
```

---

## 4. 队友本地跑后端（联调用）

1. 建虚拟环境并装依赖：
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   > `requirements.txt` 含 `faiss-gpu`，**需要 NVIDIA 显卡 + CUDA**。无显卡机器请把 `faiss-gpu` 改成 `faiss-cpu` 再装。

2. 把队长给的 **`data_share.zip`** 解压到项目根目录（得到 `data/raw/segments.csv`、`data/results/*`、`data/vectorstore/*`）。

3. 复制 `.env.example` 为 `.env`，填入 `VOLCENGINE_API_KEY`（向队长索取）。

4. （可选）若本地已有 BGE-M3 模型，在 `.env` 加 `EMBEDDING_MODEL_PATH=models/Xorbits/bge-m3`；不填则首次运行自动从 HuggingFace 镜像下载 `BAAI/bge-m3`。

5. 启动：
   ```bash
   uvicorn src.api.main:app --port 8000
   ```
   另开终端跑前端（如用本项目 React 前端）：
   ```bash
   cd frontend && npm install && npm run dev
   # 访问 http://localhost:5180
   ```

---

## 5. 与 Streamlit 的关系

项目有**两套前端**共用同一套后端/算法：
- **Streamlit**（`src/ui/app.py`）：主交付物，三页式监控界面，直接 `streamlit run src/ui/app.py`。
- **React**（`frontend/`）：可选美化版 Web 大屏，通过本 API 取数。

前端开发请**以本文档接口为准**对齐字段；后端字段定义在 `src/api/routes/*.py` 与 `src/utils/constants.py`，如有变动以代码为准并同步更新本文档。
