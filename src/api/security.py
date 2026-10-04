"""
API 认证与限流
===============

**为什么必须有**：本系统有两个会花钱/被滥用的入口：
- ``POST /api/explanation/query`` 每次真实调用 DeepSeek API，消耗付费额度
- ``POST /api/stream/ingest`` 任何人可推送伪造遥测，污染检测结果

原实现 16 个路由**全部无认证**，接口地址一旦泄露即等于送钱给别人。

## 设计取舍

**认证方式**：API Key（``X-API-Key`` 头）。
不引入 JWT —— 这是内部运维系统，没有多用户体系，JWT 的签名/过期管理是过度设计。

**未配置密钥时的默认行为：拒绝全部**。
这是刻意的安全默认值。若默认放行，运维忘记配置 = 系统裸奔且无人察觉；
若默认拒绝，未配置时报错明确，反而会促使正确配置。
可通过 ``API_AUTH_DISABLED=true`` 显式关闭（仅建议本地开发用）。

**豁免路径**：``/api/health`` 免认证 —— 否则 K8s liveness probe
与负载均衡健康检查都会失败。

**限流**：令牌桶。写接口严格（尤其 ``/api/explanation/query`` 花钱），
读接口宽松。返回 429 并带 ``Retry-After``。

**不引入 slowapi 等依赖** —— 令牌桶约 60 行即可，避免为一个鉴权需求
增加供应链面。
"""
from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict  # noqa: F401  (预留)
from typing import Iterable

from fastapi import HTTPException, Request

# ---------------------------------------------------------------- 配置


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def configured_keys() -> list[str]:
    """从环境变量读允许的 API Key（逗号分隔，支持多 key 便于轮换）。"""
    raw = os.environ.get("API_AUTH_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def auth_enabled() -> bool:
    """是否启用认证。

    默认启用。若显式设置 ``API_AUTH_DISABLED=true`` 则关闭。
    **未配置任何 key 时认证仍算启用** —— 此时所有受保护路径都会被拒绝，
    这是刻意的安全默认（见模块 docstring）。
    """
    return not _env_flag("API_AUTH_DISABLED", default=False)


# ---------------------------------------------------------------- 认证

EXEMPT_PATHS = frozenset(
    {
        "/api/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/docs/oauth2-redirect",
    }
)


def _extract_key(request: Request) -> str:
    """按优先级提取 API Key。

    `X-API-Key` 头 > `Authorization: Bearer` > `?api_key=` query。

    query 形式是为 **WebSocket** 保留的 —— 浏览器 WebSocket API 无法设置
    自定义请求头，只能把密钥放进 URL。HTTP 请求应优先用头（密钥不会进
    访问日志）。
    """
    h = request.headers.get("x-api-key")
    if h and h.strip():
        return h.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    qk = request.query_params.get("api_key")
    if qk and qk.strip():
        return qk.strip()
    return ""


def require_api_key(request: Request) -> None:
    """FastAPI 依赖：校验 API Key。

    认证被显式关闭时直接放行（本地开发场景）。
    """
    if not auth_enabled():
        return
    if request.url.path in EXEMPT_PATHS:
        return
    keys = configured_keys()
    if not keys:
        raise HTTPException(
            status_code=503,
            detail=(
                "API 认证未配置：环境变量 API_AUTH_KEYS 为空。"
                "请设置逗号分隔的 key，或显式设 API_AUTH_DISABLED=true "
                "（仅建议本地开发）。"
            ),
        )
    given = _extract_key(request)
    # 用常数时间比较，避免逐字符计时侧信道
    import hmac

    ok = any(hmac.compare_digest(given, k) for k in keys)
    if not ok:
        raise HTTPException(
            status_code=401,
            detail="缺少或无效的 API Key。请在 X-API-Key 头提供。",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def startup_warning() -> str:
    """返回启动时的安全提示文本（由 main.py 打印）。"""
    if not auth_enabled():
        return (
            "[安全警告] API_AUTH_DISABLED=true —— 认证已关闭，"
            "所有接口裸奔。仅可用于本地开发，严禁公网部署。"
        )
    keys = configured_keys()
    if not keys:
        return (
            "[安全警告] API_AUTH_KEYS 未配置 —— 除健康检查外，"
            "所有接口将返回 503。这是有意的安全默认，请配置密钥。"
        )
    return f"[安全] API 认证已启用，已加载 {len(keys)} 个 key。/api/health 免认证。"


# ---------------------------------------------------------------- 限流

# (路径前缀, 窗口秒数, 窗口内允许请求数)
#
# ⚠ 用**固定窗口**而非令牌桶 —— 令牌桶需要 rate 与 burst 两个参数，
#   两者易自相矛盾（实测 /api/explanation/query 设 rate=0.2/s burst=3 时，
#   因每次真实 LLM 调用耗时 2~8 秒，6 次请求被分散到多个滑动窗口内，
#   限流形同虚设）。固定窗口只有「窗口时长 + 窗口内上限」两个量，
#   语义直观、可预测、易验收：钱花得快的接口就调短窗口调小上限。
DEFAULT_LIMITS: list[tuple[str, float, int]] = [
    # 花钱接口：10 秒内最多 2 次（约 0.2 次/秒）
    ("/api/explanation/query", 10.0, 2),
    ("/api/explanation", 10.0, 4),
    # 写接口：10 秒内最多 20 次
    ("/api/stream", 10.0, 20),
    # 读接口：10 秒内最多 200 次
    ("/api/", 10.0, 200),
]


def _match_limit(path: str) -> tuple[float, int]:
    """返回 (窗口秒数, 窗口内上限)。取第一个匹配的前缀。"""
    for prefix, window, cap in DEFAULT_LIMITS:
        if path.startswith(prefix):
            return window, cap
    return 10.0, 200


class FixedWindowLimiter:
    """按 (客户端, 路径) 分桶的固定窗口计数器。

    相比令牌桶的优势：只有「窗口时长」与「窗口上限」两个参数，
    不会像令牌桶那样因 rate/burst 配错而失效；且 Retry-After 可精确计算。

    多线程安全。内存随 (客户端 × 路径) 增长，为防内存耗尽，
    超过 ``_MAX_BUCKETS`` 时淘汰最久未使用的桶。
    """

    #: 最多保留多少个桶（防止被伪造大量不同 IP 撑爆内存）
    _MAX_BUCKETS = 10_000

    def __init__(self) -> None:
        # key -> (窗口起始时刻, 窗口内计数)
        self._hits: dict[tuple[str, str], list[float, int]] = {}
        self._lock = threading.Lock()

    def check(self, client: str, path: str) -> tuple[bool, int]:
        """返回 (是否放行, Retry-After 秒数)。"""
        window, cap = _match_limit(path)
        now = time.monotonic()
        key = (client, path.split("?")[0])
        with self._lock:
            if key not in self._hits and len(self._hits) >= self._MAX_BUCKETS:
                # 淘汰最久未访问的半个桶
                victims = sorted(self._hits.items(), key=lambda kv: kv[1][0])[
                    : self._MAX_BUCKETS // 2
                ]
                for k, _ in victims:
                    del self._hits[k]
            slot = self._hits.get(key)
            if slot is None or now - slot[0] >= window:
                slot = [now, 0]
                self._hits[key] = slot
            if slot[1] >= cap:
                return False, max(1, int(window - (now - slot[0])) + 1)
            slot[1] += 1
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


_limiter = FixedWindowLimiter()


def rate_limit(request: Request) -> None:
    """FastAPI 依赖：令牌桶限流。"""
    if _env_flag("API_RATE_LIMIT_DISABLED", default=False):
        return
    client = request.client.host if request.client else "unknown"
    ok, retry_after = _limiter.check(client, request.url.path)
    if not ok:
        raise HTTPException(
            status_code=429,
            detail=(
                "请求过于频繁。"
                "本接口会消耗 LLM 配额，请降低调用频率。"
            ),
            headers={"Retry-After": str(retry_after)},
        )
