# 前端开发对接交接清单（HANDOFF）

> 给负责前端界面的同学。一句话任务：**参考本仓库的代码结构编写前端，并且所有接口必须严格对齐 `API.md`。**

---

## 0. 你的任务

- 阅读并理解本项目的目录结构与后端接口（重点是 `src/api/` 与 `API.md`）。
- 编写前端界面（技术栈不限，可参考现有 `frontend/` 的 React 实现）。
- **所有数据接口必须严格对齐 `API.md` 中的路径、字段名、数据类型**，不得擅自增减字段或改名。

---

## 1. 获取代码

```bash
git clone -b main https://github.com/Jerry518520/microsat-anomaly-analysis.git
cd microsat-anomaly-analysis
```

> ⚠️ 代码都在 **`main`** 分支，务必 `clone -b main` 或克隆后 `git checkout main`。

---

## 2. 准备数据（向队长单独要 `data_share.zip`）

仓库**不收录数据**（体积大、含实验产物），必须由队长私下发你 `data_share.zip`，解压到项目**根目录**：

- `data/raw/segments.csv` —— 遥测原始数据（接口读取）
- `data/results/*.json` —— 实验结果（看板/实验页读取）
- `data/vectorstore/*` —— 向量索引（RAG 检索用）

---

## 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入队长给你的 `VOLCENGINE_API_KEY`。其余保持默认即可（不填模型路径时，首次运行自动从 HuggingFace 镜像下载 `BAAI/bge-m3`）。

---

## 4. 后端运行（接口联调基准）

需要 **Python 3.11+** 与 **NVIDIA CUDA**（当前 `requirements.txt` 含 `faiss-gpu`，队友机器已确认有显卡）。

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
uvicorn src.api.main:app --port 8000
```

健康检查：

```bash
curl http://localhost:8000/api/health
# => {"status":"ok","version":"2.0.0"}
```

> 无显卡机器兜底：把 `requirements.txt` 的 `faiss-gpu` 改成 `faiss-cpu` 再 `pip install`。

---

## 5. 前端运行

```bash
cd frontend
npm install
npm run dev      # 默认 http://localhost:5180
```

Vite 已在 `frontend/vite.config.ts` 配置 `/api` 代理到 `http://localhost:8000`，前端直接请求 `/api/...` 即可，无需写完整域名。

> 现有 `frontend/src/App.tsx` 是 React 范例，通道映射、严重度配色都按 `API.md` 实现，可直接参考其处理方式。

---

## 6. 接口对齐硬性要求（必读 `API.md`）

- **通道编码**：9 个遥测通道见 `API.md` §2，前端 `label` 必须与之一致（如 `CADC0874` = 磁力计 Z轴）。
- **严重度 5 档**：`nominal / caution / warning / critical / offline`，阈值与配色见 `API.md` §2，不得擅自改阈值的数值或颜色。
- **接口清单**（全部以 `/api` 为前缀，响应均为 JSON）：
  - `GET /api/health`
  - `GET /api/dashboard/{metrics,channels,alerts,status}`
  - `GET /api/detection/{rag-config,experiments,channel-f1,system-params}`
  - `GET /api/explanation/{detail,waveform}` + `POST /api/explanation/query`
- **不要**在前端代码中传任何 API Key；Key 由服务端从环境变量读取。

---

## 7. 验收 Checklist

- [ ] 能 `clone -b dev` 并在本地起后端 + 前端
- [ ] 9 个通道 `label` 与配色和 `API.md` §2 一致
- [ ] 严重度 5 档判定逻辑与阈值一致
- [ ] 所有请求走 `/api/...`，返回字段名与 `API.md` 示例**完全一致**
- [ ] `GET /api/explanation/detail` 返回 `garbled=true` 时，前端给出友好提示而非乱码渲染
- [ ] 不向仓库提交 `.env`、模型文件、`data/`、`node_modules/`

---

## 8. 提交规范

- 基于 `dev` 新建功能分支：`git checkout -b feat/frontend-xxx`
- 提 PR 到 `dev`，描述清楚改动内容与接口对齐情况
- **严禁**提交 `.env`、模型权重、`data/`、`node_modules/`、`__pycache__/`

---

## 9. 常见问题

- **后端起不来** → 检查 CUDA / `faiss-gpu` 是否装好、`.env` 是否填了 key、`data/` 是否解压到位。
- **RAG 问答 `/query` 返回空或报错** → 需要 BGE-M3 模型（首次自动从镜像下载，或填 `EMBEDDING_MODEL_PATH` 指向本地模型）。
- **其余接口（dashboard / detection / detail / waveform）只需 `segments.csv + results + vectorstore`，无需模型即可联调**——可先把这些界面跑通，再处理 RAG 问答。
