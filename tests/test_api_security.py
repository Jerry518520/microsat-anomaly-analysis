"""
API 安全验收测试
================

覆盖上线前的两个安全项：**认证** 与 **限流**。

**为什么必须有这两个测试**：项目此前 16 个接口全部无认证，而
``/api/explanation/query`` 每次真实调用 DeepSeek API 消耗付费额度。
这类缺陷不会让任何测试失败，只会在上线后被人刷账单时暴露。

**测试用真实 uvicorn 子进程而非 TestClient** —— 实测 FastAPI 的
``TestClient`` **不会执行** ``app.dependencies`` 里的函数，
用 TestClient 测认证会得到「无 key 也能过」的假通过。
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KEYS = "test-key-alpha,test-key-beta"
PORT = 8899


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# 绕开本机代理，否则会返回 502
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _req(base, path, headers=None, method="GET", body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers=headers or {}, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        r = _OPENER.open(req, timeout=timeout)
        return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    env = dict(os.environ)
    env["API_AUTH_KEYS"] = KEYS
    env.pop("API_AUTH_DISABLED", None)
    env.pop("API_RATE_LIMIT_DISABLED", None)
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "src.api.main:app",
            "--host", "127.0.0.1", "--port", str(port),
            "--log-level", "warning",
        ],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(90):
        if proc.poll() is not None:
            out = proc.stdout.read().decode(errors="replace")
            pytest.fail(f"服务启动失败：\n{out[-1500:]}")
        try:
            code, _, _ = _req(base, "/api/health", timeout=3)
            if code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        proc.kill()
        pytest.fail("服务 90 秒内未就绪")
    yield base
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


PROTECTED = [
    "/api/dashboard/metrics",
    "/api/dashboard/channels",
    "/api/dashboard/status",
    "/api/dashboard/alerts",
    "/api/detection/rag-config",
    "/api/detection/experiments",
    "/api/detection/channel-f1",
    "/api/detection/system-params",
    "/api/stream/state",
]


# ---------------------------------------------------------------- 认证


def test_health_is_exempt_from_auth(server):
    """/api/health 必须免认证 —— 否则 K8s liveness probe 会失败。"""
    code, _, _ = _req(server, "/api/health")
    assert code == 200, f"健康检查应免认证，实得 {code}"


@pytest.mark.parametrize("path", PROTECTED)
def test_protected_endpoint_rejects_missing_key(server, path):
    """无 key 调受保护接口必须 401，不能放行。"""
    code, body, _ = _req(server, path)
    assert code == 401, f"{path} 无 key 应 401，实得 {code}"
    assert "API Key" in body.decode(errors="replace")


@pytest.mark.parametrize("path", PROTECTED)
def test_protected_endpoint_rejects_wrong_key(server, path):
    """错误 key 必须 401。"""
    code, _, _ = _req(server, path, headers={"X-API-Key": "totally-wrong"})
    assert code == 401, f"{path} 错误 key 应 401，实得 {code}"


@pytest.mark.parametrize("key", KEYS.split(","))
def test_each_configured_key_works(server, key):
    """逗号分隔的每个 key 都应可用（便于轮换）。"""
    code, _, _ = _req(server, "/api/dashboard/metrics", headers={"X-API-Key": key})
    assert code == 200, f"配置的 key {key} 应可用，实得 {code}"


def test_bearer_scheme_supported(server):
    """Authorization: Bearer 也要能用（便于网关统一鉴权）。"""
    code, _, _ = _req(server, "/api/dashboard/metrics",
                      headers={"Authorization": f"Bearer {KEYS.split(',')[0]}"})
    assert code == 200, f"Bearer 方式应可用，实得 {code}"


def test_api_key_via_query_supported(server):
    """?api_key= 形式须支持 —— 浏览器的 WebSocket API 无法设置自定义头。"""
    code, _, _ = _req(server, f"/api/dashboard/metrics?api_key={KEYS.split(',')[0]}")
    assert code == 200, f"query 传 key 应可用，实得 {code}"


def test_key_length_leak_not_returned(server):
    """401 响应体不应泄露密钥长度/内容等信息。"""
    code, body, _ = _req(server, "/api/dashboard/metrics")
    text = body.decode(errors="replace")
    assert KEYS.split(",")[0] not in text, "错误响应泄露了配置的 key"
    assert "test-key" not in text, "错误响应泄露了 key 前缀"


# ---------------------------------------------------------------- 限流


def test_limiter_limits_query_endpoint():
    """query 接口限流必须很严（10 秒内 2 次）。"""
    from src.api.security import FixedWindowLimiter, _match_limit

    window, cap = _match_limit("/api/explanation/query")
    assert cap <= 2, f"query 接口每次真实调用 LLM 耗钱，窗口上限应 ≤2，实得 {cap}"

    lim = FixedWindowLimiter()
    results = [lim.check("1.2.3.4", "/api/explanation/query")[0] for _ in range(5)]
    assert results[:2] == [True, True], f"前 2 次应放行，实得 {results}"
    assert results[2] is False, f"第 3 次应被限流，实得 {results[2]}"


def test_limiter_read_endpoint_is_looser():
    """读接口应比写接口宽松。"""
    from src.api.security import _match_limit

    _, q = _match_limit("/api/explanation/query")
    _, r = _match_limit("/api/dashboard/metrics")
    assert r > q * 10, f"读接口上限应远高于 query，实得 {r} vs {q}"


def test_limiter_retry_after_positive():
    """被限流时 Retry-After 必须是正整数秒。"""
    from src.api.security import FixedWindowLimiter

    lim = FixedWindowLimiter()
    lim.check("5.6.7.8", "/api/explanation/query")
    lim.check("5.6.7.8", "/api/explanation/query")
    ok, retry = lim.check("5.6.7.8", "/api/explanation/query")
    assert ok is False
    assert isinstance(retry, int) and retry >= 1, f"Retry-After 应为正整数秒，实得 {retry!r}"


def test_limiter_isolates_clients():
    """不同客户端互不影响 —— 否则一个内网用户能拖垮所有人。"""
    from src.api.security import FixedWindowLimiter

    lim = FixedWindowLimiter()
    for _ in range(2):
        lim.check("9.9.9.1", "/api/explanation/query")
    blocked_a, _ = lim.check("9.9.9.1", "/api/explanation/query")
    ok_b, _ = lim.check("9.9.9.2", "/api/explanation/query")
    assert blocked_a is False, "客户端 A 应已被限流"
    assert ok_b is True, "客户端 B 不应受 A 影响"


def test_limiter_bounded_memory():
    """桶数量必须有上限，否则伪造大量来源 IP 会撑爆内存。"""
    from src.api.security import FixedWindowLimiter

    assert FixedWindowLimiter._MAX_BUCKETS <= 100_000
    lim = FixedWindowLimiter()
    for i in range(FixedWindowLimiter._MAX_BUCKETS + 200):
        lim.check(f"10.0.{i // 256}.{i % 256}", "/api/dashboard/metrics")
    assert len(lim._hits) <= FixedWindowLimiter._MAX_BUCKETS, (
        f"桶数应受限，实得 {len(lim._hits)}"
    )


# ---------------------------------------------------------------- 兜底


def test_unconfigured_keys_denies_all():
    """未配置 API_AUTH_KEYS 时必须拒绝全部（503），不能静默放行。

    这是刻意的安全默认：运维忘记配置时应明确报错，
    而不是「看起来正常」地裸奔。
    """
    port = _free_port()
    env = dict(os.environ)
    env.pop("API_AUTH_KEYS", None)      # 关键：不配置
    env.pop("API_AUTH_DISABLED", None)  # 且未显式关闭认证
    env.pop("API_RATE_LIMIT_DISABLED", None)
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "src.api.main:app",
            "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning",
        ],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        ready = False
        for _ in range(60):
            if proc.poll() is not None:
                pytest.fail(f"服务启动失败：\n{proc.stdout.read().decode(errors='replace')[-1200:]}")
            try:
                if _req(base, "/api/health", timeout=3)[0] == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)
        assert ready, "服务未就绪"

        # 健康检查仍应免认证
        assert _req(base, "/api/health")[0] == 200, "健康检查必须始终免认证"
        # 但受保护接口应 503 拒绝
        code, body, _ = _req(base, "/api/dashboard/metrics")
        assert code == 503, f"未配置 key 时应 503 拒绝，实得 {code}"
        assert "API_AUTH_KEYS" in body.decode(errors="replace"), (
            "503 响应应指明缺少哪个环境变量"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_auth_can_be_explicitly_disabled():
    """API_AUTH_DISABLED=true 时应放行（仅本地开发用）。"""
    port = _free_port()
    env = dict(os.environ)
    env.pop("API_AUTH_KEYS", None)
    env["API_AUTH_DISABLED"] = "true"
    env.pop("API_RATE_LIMIT_DISABLED", None)
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "src.api.main:app",
            "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning",
        ],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        ready = False
        for _ in range(60):
            if proc.poll() is not None:
                pytest.fail(f"服务启动失败：\n{proc.stdout.read().decode(errors='replace')[-1200:]}")
            try:
                if _req(base, "/api/health", timeout=3)[0] == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)
        assert ready, "服务未就绪"
        code, _, _ = _req(base, "/api/dashboard/metrics")
        assert code == 200, (
            f"显式关闭认证后应放行（本地开发场景），实得 {code}"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
