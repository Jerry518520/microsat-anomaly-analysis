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
```

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
| 嵌入模型相关异常 | 模型没下或路径不对 | 跑 `download_model.py --check` |
| 索引读不出 | faiss 版本不符 | 确认 `faiss-cpu==1.14.2` |
| 修改代码后 `git_dirty=True` | 自指循环，见下 | 正常，非缺陷 |

> **关于 `meta.git_dirty` 恒为 True**：结果 JSON 每次生成都会改写自身，
> 因此记录 meta 的那一刻工作区必然非干净。**这不是缺陷**。
> 溯源是否可信以 `meta.git_commit` 是否与 HEAD 一致、以及数字能否独立复算为准。

---

## 1. 公网分享（ngrok）

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
