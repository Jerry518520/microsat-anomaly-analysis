"""FastAPI 入口 — 封装现有 Python 逻辑为 REST API

安全模型（2026-10-04 上线加固）
--------------------------------
- **认证**：除 `/api/health` 外，所有接口需在 `X-API-Key` 头提供有效密钥。
  密钥由环境变量 `API_AUTH_KEYS` 配置（逗号分隔支持多 key 便于轮换）。
  **未配置密钥时默认拒绝全部**（返回 503），这是刻意的安全默认 ——
  运维忘记配置时应报错而非静默裸奔。本地开发可显式设
  `API_AUTH_DISABLED=true` 关闭。
  理由：`/api/explanation/query` 每次真实调用 DeepSeek API 消耗付费额度，
  `/api/stream/ingest` 任何人可推送伪造遥测污染检测结果。
- **限流**：令牌桶，按客户端 + 路径分桶。写接口严格（query 接口 0.2 次/秒），
  读接口宽松。超限返回 429 + `Retry-After`。可设 `API_RATE_LIMIT_DISABLED=true` 关闭。
  实现见 `src/api/security.py`。
"""
import sys
import os

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

# 必须在读取 API_AUTH_KEYS 之前加载 .env，否则密钥不生效
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import dashboard, detection, explanation, stream
from src.api.security import rate_limit, require_api_key, startup_warning

app = FastAPI(
    title="OPS-SAT Telemetry API",
    description="卫星遥测异常诊断系统 REST API",
    version="2.1.0",
    # 全局依赖：每个请求先过认证（/api/health 在 security 内豁免），
    # 再过限流。放在 dependencies 里而非逐路由装饰，是为了让新增路由
    # 自动受保护 —— 漏加装饰器就会裸奔。
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)

# CORS — 允许 React dev server 访问
# ⚠ allow_headers 需包含 X-API-Key，否则浏览器预检会拦掉带密钥的请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5180", "http://127.0.0.1:5180", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key", "X-Request-ID"],
)

# 注册路由
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(detection.router, prefix="/api/detection", tags=["Detection"])
app.include_router(explanation.router, prefix="/api/explanation", tags=["Explanation"])
app.include_router(stream.router, prefix="/api/stream", tags=["Stream"])


@app.on_event("startup")
def _print_security_banner():
    print(startup_warning())


@app.get("/api/health")
def health_check():
    """健康检查 —— 刻意免认证，供 K8s liveness probe / 负载均衡使用。

    注意：不要在此返回任何敏感信息（密钥、版本明细、依赖路径）。
    """
    return {"status": "ok", "version": "2.1.0"}
