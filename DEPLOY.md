# 部署指南

本文件分两部分：
- **第 0 节「从零克隆」** — 新机器首次部署必读
- 第 1 节起 — 公网分享（ngrok）

---

## 0. 从零克隆（新机器首次部署）

### 0.1 环境要求

| 项 | 要求 | 备注 |
|---|---|---|
| Python | 3.13（本项目在 3.13.13 验证） | `pyproject.toml` 声明 `>=3.11` |
| 磁盘 | **≥ 10 GB 可用** | 模型 2.2 GiB + 索引 22 MiB + 依赖约 2 GiB |
| 网络 | 需能访问 `hf-mirror.com` | 见 0.4 |
| 显卡 | **可选，但强烈建议** | 见下方说明 |

**关于显卡（两个环节需求不同）**

| 环节 | 需要显卡吗 | 实测 |
|---|---|---|
| BGE-M3 向量化（建索引 / 编码） | **强烈建议有** | 编码 128 条：CPU **2.17 s** vs GPU **0.26 s**（**8.3 倍**） |
| FAISS 检索 | 不需要 | `faiss-cpu` 纯 CPU，4868 chunk 实测正常 |

`configs/rag_config.yaml` 的 `embedding.device` 默认是 **`cuda`**。
无 NVIDIA 显卡时**必须改成 `cpu`**，否则建索引会因找不到 CUDA 设备而失败：

```yaml
embedding:
  device: cpu        # 无显卡时改这一行
```

只做检测/看板（不生成 RAG 解释）时无需改 —— 该路径不触发向量化。

**不需要 `faiss-gpu`**：实测该包在 Python 3.13 + Windows 下
`No matching distribution`，实际可用的是 `faiss-cpu`（纯 CPU 检索）。

### 0.2 克隆与安装依赖

```bash
git clone https://github.com/Jerry518520/microsat-anomaly-analysis.git
cd microsat-anomaly-analysis

# 建虚拟环境
python -m venv .venv
# Windows:
.venv/Scripts/python.exe -m pip install -U pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
# Linux/macOS:
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` 已锁定 `faiss-cpu==1.14.2`（索引由该版本写出，混用版本可能读不出）。

### 0.3 准备环境变量

```bash
cp .env.example .env
```

至少填两项：

```env
# LLM（必需，RAG 解释用）
DEEPSEEK_API_KEY=sk-xxxxx

# 嵌入模型本地路径（离线环境必需，见 0.4）
EMBEDDING_MODEL_PATH=models/Xorbits/bge-m3

# API 认证（**上线必填**，见 0.3b）
API_AUTH_KEYS=your-key-here,another-key
```

### 0.3b API 认证（**上线必配**）

后端自 v2.1.0 起对**除 `/api/health` 外**的所有接口启用 API Key 认证。

**未配置 `API_AUTH_KEYS` 时所有受保护接口返回 503** —— 这是刻意的安全默认：
忘记配置应明确报错，而非静默裸奔。

三种传 key 方式（按优先级）：

| 方式 | 用途 |
|---|---|
| `X-API-Key: <key>` 头 | HTTP 请求（推荐，密钥不进访问日志） |
| `Authorization: Bearer <key>` | 便于网关统一鉴权 |
| `?api_key=<key>` | **仅 WebSocket 用** —— 浏览器 WS API 无法设置自定义头 |

**限流规则**（固定窗口）：

| 路径 | 限制 |
|---|---|
| `/api/explanation/query` | 10 秒内 2 次（**每次真实调用 DeepSeek，耗钱**） |
| `/api/explanation` | 10 秒内 4 次 |
| `/api/stream` | 10 秒内 20 次 |
| 其余 `/api/*` | 10 秒内 200 次 |

超限返回 **429** + `Retry-After` 头。

可选开关：

```env
API_AUTH_DISABLED=true      # 本地开发临时关闭认证（严禁公网部署）
API_RATE_LIMIT_DISABLED=true # 关闭限流
```

前端会自动带 key：优先读 `VITE_API_KEY`，其次 `localStorage.api_key`。
⚠ 生产部署**不要**把 key 打进前端包（`VITE_*` 会被内联进产物）；
正式上线应在反向代理（Nginx/网关）层注入并鉴权。

验证：

```bash
curl http://127.0.0.1:8000/api/health                              # 200，免认证
curl http://127.0.0.1:8000/api/dashboard/metrics                 # 401，无 key
curl -H "X-API-Key: $KEY" http://127.0.0.1:8000/api/dashboard/metrics  # 200
```

> ⚠ 本机若配置了代理，`curl` 本机地址可能返回 502（代理拦截）。
> 用 Python 绕过：`urllib.request.build_opener(urllib.request.ProxyHandler({}))`。

### 0.3c 告警持久化

告警与段判定写 SQLite（默认 `data/alerts.db`，已在 `.gitignore` 中），
**进程重启不丢**。表结构幂等创建，无需手动初始化。

| 接口 | 用途 |
|---|---|
| `GET /api/stream/alerts` | 查历史告警，支持 `since`/`until`（unix 秒）、`channel`、`severity` 过滤，`limit`/`offset` 分页（limit ≤ 1000） |
| `GET /api/stream/alerts/stats` | 总数、按严重度/通道分布 |

> 与 `/api/stream/state` 里的 `alerts` 不同：那个只有内存最近 200 条且**重启即丢**，
> `/api/stream/alerts` 读的是持久化库。

### 0.4 下载嵌入模型（**必做，约 2.2 GiB**）

模型**不入 git**（`.gitignore` 忽略 `models/`），须本地下载。

```bash
# Windows
.venv/Scripts/python.exe scripts/download_model.py
# Linux/macOS
.venv/bin/python scripts/download_model.py
```

常用参数：

| 参数 | 用途 |
|---|---|
| `--check` | 只校验完整性，不下载 |
| `--skip-weights` | 只下配置文件，跳过 2.2 GiB 权重 |
| `--source official` | 换用 `huggingface.co`（默认走 hf-mirror） |
| `--target <dir>` | 指定存放目录（须与 `.env` 一致） |

> **为什么默认走 hf-mirror**：实测 `huggingface.co` 直连 **502**，hf-mirror **0.5 s 可达**。
> 脚本会带 User-Agent（实测空 UA 会 403），并对失败重试 3 次（镜像偶发超时）。
> 权重支持断点续传，中断后重跑即可接着下。

验证：

```bash
.venv/Scripts/python.exe scripts/download_model.py --check
# 预期输出：✓ 模型完整
```

### 0.5 FAISS 索引

索引（22 MiB）**已入库**，clone 后无需操作。

若修改了知识库或嵌入模型，才需重建（约 2.5 分钟）：

```bash
.venv/Scripts/python.exe scripts/build_index.py
```

### 0.6 启动

**后端**（FastAPI）：

```bash
# ⚠ 必须用 `python -m uvicorn`，不能直接 `uvicorn`
#   （裸命令会使 sys.path[0] 变成 venv/Scripts，导致 import src 失败）
python -m uvicorn src.api.main:app --port 8000 --host 127.0.0.1
```

**前端**（React）：

```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:5180

### 0.7 冒烟验证

```bash
# 跑全部测试（应 128 passed, 1 skipped）
.venv/Scripts/python.exe -m pytest -q

# 端到端跑一次检测 + RAG 解释
.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
from src.integration.anomaly_rag_pipeline import AnomalyRAGPipeline
p = AnomalyRAGPipeline()
r = p.detect_and_explain(max_explanations=3)
print('完成，解释条数 =', len(r))
"
```

预期：`生产 F1=0.644628`，3 条解释正常生成。

### 0.8 健康检查

```bash
curl http://127.0.0.1:8000/api/health
```

> ⚠ 本机若配置了代理，`curl` 本机地址可能返回 502（代理拦截）。
> 改用 Python 绕过代理：
> ```python
> import urllib.request
> op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
> print(op.open('http://127.0.0.1:8000/api/health', timeout=10).read())
> ```

### 0.9 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `No module named 'fastapi'` | 依赖没装全 | 重跑 0.2 的 pip install |
| `No module named 'src'` | 用了裸 `uvicorn` | 改 `python -m uvicorn` |
| `Found no NVIDIA driver` / CUDA 设备报错 | 配置里 `device: cuda` 但无显卡 | `configs/rag_config.yaml` 的 `embedding.device` 改 `cpu` |
| 建索引很慢 | 无显卡，CPU 编码比 GPU 慢约 8 倍 | 属正常；加显卡可显著提速 |
| 嵌入模型相关异常 | 模型没下或路径不对 | 跑 `download_model.py --check` |
| 调接口返回 **401** | 未提供或 key 错 | 加头 `X-API-Key: <key>`；检查 `.env` 的 `API_AUTH_KEYS` |
| 调接口返回 **503** | 未配置 `API_AUTH_KEYS` | 在 `.env` 配至少一个 key 后重启服务 |
| 调接口返回 **429** | 触发限流 | 按 `Retry-After` 头等待；query 接口 10 秒仅 2 次 |
| 前端报「未授权（401）」 | 前端没拿到 key | `localStorage.api_key = '<key>'` 或设 `VITE_API_KEY` 后重新构建 |
| WebSocket 连不上 | WS 需用 query 传 key | `ws://host/api/stream/ws?api_key=<key>`（浏览器 WS 不能加自定义头） |
| 索引读不出 | faiss 版本不符 | 确认 `faiss-cpu==1.14.2` |
| 修改代码后 `git_dirty=True` | 自指循环，见下 | 正常，非缺陷 |

> **关于 `meta.git_dirty` 恒为 True**：结果 JSON 每次生成都会改写自身，
> 因此记录 meta 的那一刻工作区必然非干净。**这不是缺陷**。
> 溯源是否可信以 `meta.git_commit` 是否与 HEAD 一致、以及数字能否独立复算为准。

> **关于 faiss-gpu**：`requirements.txt` 用的是 `faiss-cpu==1.14.2`，**不要**改成
> `faiss-gpu` —— 实测该包在 Python 3.13 + Windows 下 `No matching distribution`，
> 装不上。检索走 CPU 即可，向量化才需要 GPU。

---

## 1. 实时数据接入（`ingest` 通道）

系统**可以接收外部实时遥测数据**，不只回放历史 CSV。实测验证过完整链路。

### 1.1 两种数据源

| 模式 | 用途 | 段 id 来源 | 界面标注 |
|---|---|---|---|
| `replay` | 回放 `data/raw/segments.csv`（18 天历史） | 数据自带，**与官方段完全一致** | 须标 `SIMULATED LIVE` |
| **`ingest`** | **外部数据源实时推送** | 无 id 时按采样间隔规则近似切段 | 正常 live |

> OPS-SAT 已于 2024-05-22 离轨，**不存在真实时流**，
> 所以默认用 `replay` 演示；`ingest` 是为将来接入真实地面站数据准备的通道。

### 1.2 推送数据

```bash
POST /api/stream/ingest
Content-Type: application/json

{
  "frames": [
    {"ts": 1767225600.0, "channel": "CADC0892", "value": 0.0,   "sampling": 1},
    {"ts": 1767225601.0, "channel": "CADC0892", "value": 0.545, "sampling": 1}
  ]
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `ts` | **float（unix 秒）** | **不是 ISO 字符串** |
| `channel` | str | 通道 ID，如 `CADC0892` |
| `value` | float | 遥测读数 |
| `segment_id` | int? | 可选。有则按官方段 id 分组；**缺省则按采样间隔近似切段** |
| `sampling` | int | 采样间隔（秒），默认 1 |

返回：

```json
{"accepted": 27, "rejected": 0}
```

`rejected > 0` 通常表示该通道正在回放（同一时刻只允许一个数据源）。

### 1.3 段闭合与判定

**判定只在段闭合时发生。** 切段规则（`src/streaming/segmenter.py`）：

> 相邻点时间差超过 `max(3 × sampling, 5s)` 即切段。

因此推完一批点后，**需再推一个时间跳跃的点触发切段**，才会产生判定：

```python
# 第 1 批：一个真实异常段的 27 个点
POST /ingest {"frames": [... ts=t0..t0+26 ...]}
# 第 2 批：时间跳跃，触发段闭合 → 立刻判定
POST /ingest {"frames": [{"ts": t0+9999, "channel": "CADC0892", "value": 0.0}]}
```

### 1.4 双层语义（诚实性设计）

| 层 | 时机 | 是否出告警 |
|---|---|---|
| **正式判定层** | 段闭合时 | ✅ 是 |
| 增长段预览层 | 段未闭合时（每点刷新曲线） | ❌ 否，一律带 `provisional: true` |

预览分仅供大屏曲线着色，**不是判定边界**。阈值由训练分 τ95 + σ 分级决定。

### 1.5 订阅告警（WebSocket）

```
WS /api/stream/ws
```

| 帧类型 | 内容 |
|---|---|
| `snapshot` | 连接时的全量状态快照 |
| `batch` | 批量分数更新 |
| `alert` | **新告警**（含段 id、通道、严重度、规则违例数、段统计） |

### 1.6 实测验证结果

用 `data/raw/segments.csv` 中两个**真实标注为 anomaly 的段**经 `/ingest` 推入：

| 段 | 通道 | 判定 | 规则违例 |
|---|---|---|---|
| `CADC0892` seg235 | 光电二极管 5 | **critical** | 17 条（≥2 判异常） |
| `CADC0888` seg1043 | 光电二极管 3 | **critical** | 11 条 |

`/api/stream/state` 返回 `judged_segments: 2`、`alerts: 2`，告警记录含
段统计（`len` / `mean` / `std`）与规则命中数。WebSocket 收到 `snapshot` + `batch` 帧。

> 判定参数（`Stage 2: 强通道 IF(c=0.2) OR 3σ/IQR 规则; 弱通道纯规则; 段闭合判定`）
> 与离线 `fusion_v3.py` 的口径差异是刻意的：`fusion_v3.py` 用 val 选出的
> 逐通道门控配方（`gate_perchannel`），而 Stage 2 用统一的强/弱通道规则。
> **两套数字不可直接相减。**

### 1.7 未实现

- **无 Kafka / MQTT 接入** —— 目前是 HTTP POST 推送。若地面站走消息队列，
  需自行加一层转发。
- **无结果持久化** —— 告警只存内存 `deque(maxlen=200)`，进程重启即丢失。
- **无鉴权** —— `/api/stream/ingest` 与 `/api/stream/ws` 均无认证，
  公网暴露前必须加（见 1.8）。

### 1.8 公网暴露前必做

`ingest` 与 `ws` 是**写接口**且**无认证**，任何人可推送数据、订阅告警。
公网部署前至少补：

1. API Key 或 JWT 认证
2. 限流（防止刷爆 LLM 配额与 CPU）
3. 结构化日志（便于追查谁推了什么）

---

## 2. 公网分享（ngrok）


本项目使用 Streamlit 构建前端，支持通过 ngrok 内网穿透实现公网远程访问。

## 前提条件

1. 已安装 Python 3.11+
2. 已注册 ngrok 账号（https://ngrok.com，免费）

## 快速启动（局域网）

双击 `start_server.bat`，等待出现启动信息后访问 http://localhost:8501

## 公网分享步骤

### 1. 安装 ngrok

- 访问 https://ngrok.com/download 下载 Windows 版
- 解压到任意目录（如 `C:\ngrok\`）
- 注册账号后，在 https://dashboard.ngrok.com/get-started/setup 获取 authtoken
- 在命令行运行：

```bash
ngrok config add-authtoken 你的token
```

### 2. 启动 Streamlit

双击 `start_server.bat`，等待出现：

```
You can now view your Streamlit app in your browser
```

### 3. 启动 ngrok 隧道

打开**另一个**命令行窗口，运行：

```bash
ngrok http 8501
```

在输出中找到 `Forwarding` 行：

```
Forwarding  https://xxxx-xxx.ngrok-free.app -> http://localhost:8501
```

将 `https://xxxx-xxx.ngrok-free.app` 这个链接发给别人即可访问。

### 4. 停止服务

关闭两个命令行窗口即可。

## 注意事项

- ngrok 免费版链接**每次启动会变化**，这是正常的
- 电脑需要保持开机状态，关闭终端则服务停止
- RAG 诊断引擎暂未部署，仅展示模式（看板 + 深度诊断详情页可用，实时重算和问答功能暂不可用）
- 未来部署 RAG 到云服务器后，可通过 FastAPI 接口接入，无需改动前端

## 手动启动（不用 bat）

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
streamlit run src/ui/app.py --server.address 0.0.0.0 --server.port 8501
```
