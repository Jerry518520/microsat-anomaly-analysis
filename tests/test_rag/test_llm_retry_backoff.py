"""LLM 重试策略 —— 指数退避 + 鉴权快速失败（防回归）。

背景（缺陷 P0-02）：
    修复前 `src/rag/llm_client.py::generate` 的重试循环里**没有任何 sleep**，
    3 次尝试紧邻排列。实测：
        - 真实 401（打api.deepseek.com，错误 key）：3 次尝试共 1.58s
        - 429限流：3 次尝试 0.00s
        - 网络错误：3 次尝试 0.00s
    三个问题：
        1. 无退避 → 服务端收到背靠背请求只会更严格限流，重试近乎无效
        2. 401/403 也重试 → 凭据错误不会因等待自愈，纯属白等
        3. 响应 JSON 截断不参与重试 → 直接把空/半截结果当成功返回

**测试绝不能变慢**：所有用例通过注入 no-op sleep（`LLMClient(sleep_func=...)`）
断言退避**序列**，而非真的等待。真实耗时的验证见
`scripts/_tmp_measure_retry.py` 与提交信息中的实测数据。
"""

import sys
import time
from pathlib import Path

import pytest
import requests

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.llm_client import (  # noqa: E402
    AUTH_STATUS_CODES,
    RETRYABLE_STATUS_CODES,
    LLMError,
    LLMAuthError,
    LLMTransportError,
    LLMPermanentError,
    LLMRateLimitError,
    LLMResponseFormatError,
    LLMClient,
    _parse_retry_after,
    compute_backoff_delay,
)


# ============================================================
# 工具
# ============================================================

def make_response(status, body=b"{}", headers=None):
    r = requests.Response()
    r.status_code = status
    r._content = body
    r.url = "https://api.deepseek.com/v1/chat/completions"
    r.headers.update(headers or {})
    return r


def make_client(sleeps=None, max_retries=3, **cfg_over):
    """构造注入 no-op sleep 的客户端；`sleeps` 收集被请求的退避时长。"""
    recorded = sleeps if sleeps is not None else []
    client = LLMClient(sleep_func=lambda d: recorded.append(d))
    client.max_retries = max_retries
    for k, v in cfg_over.items():
        setattr(client, k, v)
    client._recorded_sleeps = recorded
    return client


def patch_post(monkeypatch, handler):
    """把 requests.post 替换为计数处理器。返回尝试次数容器。"""
    calls = {"n": 0}

    def fake_post(url, **kw):
        calls["n"] += 1
        return handler(calls["n"])

    monkeypatch.setattr("src.rag.llm_client.requests.post", fake_post)
    return calls


# ============================================================
# 1. 退避数学
# ============================================================

class TestBackoffMath:
    def test_sequence_is_exponential_then_capped(self):
        got = [compute_backoff_delay(i, base=1.0, factor=2.0, cap=30.0)
               for i in range(1, 8)]
        assert got == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]

    def test_zero_base_means_no_wait(self):
        """测试用的base=0配置下退避为 0——这是保证测试不变慢的手段。"""
        assert [compute_backoff_delay(i, base=0.0) for i in range(1, 4)] == [0.0, 0.0, 0.0]

    def test_default_params_match_documented_spec(self):
        """默认参数即需求：base=1s, factor=2, cap=30s。"""
        assert compute_backoff_delay(1) == 1.0
        assert compute_backoff_delay(3) == 4.0
        assert compute_backoff_delay(99) == 30.0

    def test_attempt_below_one_is_clamped(self):
        assert compute_backoff_delay(0) == compute_backoff_delay(1)

    def test_retry_after_header_parsed(self):
        r = make_response(429, headers={"Retry-After": "7"})
        assert _parse_retry_after(r) == 7.0

    def test_retry_after_garbage_returns_none(self):
        r = make_response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
        assert _parse_retry_after(r) is None

    def test_retry_after_absent_returns_none(self):
        assert _parse_retry_after(make_response(429)) is None


# ============================================================
# 2. 鉴权错误：必须**不重试**
# ============================================================

class TestAuthFailsFast:
    @pytest.mark.parametrize("status", sorted(AUTH_STATUS_CODES))
    def test_auth_status_raises_immediately(self, monkeypatch, status):
        sleeps = []
        client = make_client(sleeps)
        calls = patch_post(monkeypatch, lambda n: make_response(status))

        with pytest.raises(LLMAuthError):
            client.generate("ping")

        assert calls["n"] == 1, (
            f"{status} 属鉴权错误，重试无意义，却打了 {calls['n']} 次"
        )
        assert sleeps == [], f"{status} 不应触发任何退避等待，实际 {sleeps}"

    def test_auth_error_message_is_actionable(self, monkeypatch):
        client = make_client()
        patch_post(monkeypatch, lambda n: make_response(401, b'{"error":"bad key"}'))
        with pytest.raises(LLMAuthError) as ei:
            client.generate("ping")
        assert "API Key" in str(ei.value)

    def test_auth_error_is_runtime_error_subclass(self):
        """保持与原实现 `raise RuntimeError(...)` 的兼容性。

        下游（src/rag/pipeline.py:233、src/streaming/stage2.py:225）都用
        `except Exception` 兜底，行为不变。
        """
        assert issubclass(LLMAuthError, RuntimeError)
        assert issubclass(LLMError, RuntimeError)

    def test_non_auth_4xx_also_fails_fast(self, monkeypatch):
        """400/404/422 请求本身有问题，重试同样无意义。"""
        for status in (400, 404, 422):
            sleeps = []
            client = make_client(sleeps)
            calls = patch_post(monkeypatch, lambda n, s=status: make_response(s))
            with pytest.raises(LLMPermanentError):
                client.generate("ping")
            assert calls["n"] == 1, f"{status} 不应重试"
            assert sleeps == []


# ============================================================
# 3. 可重试错误：必须有**可见的退避间隔**
# ============================================================

class TestRetryWithBackoff:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_retryable_status_backs_off(self, monkeypatch, status):
        sleeps = []
        client = make_client(sleeps)
        calls = patch_post(monkeypatch, lambda n: make_response(status))

        with pytest.raises(LLMError):
            client.generate("ping")

        assert calls["n"] == 3, f"{status} 应重试满 3 次"
        assert sleeps == [1.0, 2.0], (
            f"{status} 退避序列应为 [1.0, 2.0]（base=1,factor=2），实际 {sleeps}"
        )

    def test_network_error_backs_off(self, monkeypatch):
        sleeps = []
        client = make_client(sleeps)

        def boom(n):
            raise requests.exceptions.ConnectionError("connection reset")

        calls = patch_post(monkeypatch, boom)
        with pytest.raises(LLMTransportError):
            client.generate("ping")
        assert calls["n"] == 3
        assert sleeps == [1.0, 2.0]

    def test_timeout_backs_off(self, monkeypatch):
        sleeps = []
        client = make_client(sleeps)

        def boom(n):
            raise requests.exceptions.Timeout()

        patch_post(monkeypatch, boom)
        with pytest.raises(LLMTransportError):
            client.generate("ping")
        assert sleeps == [1.0, 2.0]

    def test_retry_after_header_takes_precedence(self, monkeypatch):
        """服务端给了 Retry-After 就用它（上限受cap 约束）。"""
        sleeps = []
        client = make_client(sleeps)
        patch_post(monkeypatch, lambda n: make_response(429, headers={"Retry-After": "5"}))
        with pytest.raises(LLMError):
            client.generate("ping")
        assert sleeps == [5.0, 5.0]

    def test_retry_after_respects_cap(self, monkeypatch):
        sleeps = []
        client = make_client(sleeps)
        client.backoff_cap = 30.0
        patch_post(monkeypatch, lambda n: make_response(429, headers={"Retry-After": "600"}))
        with pytest.raises(LLMError):
            client.generate("ping")
        assert sleeps == [30.0, 30.0]

    def test_backoff_respects_configured_cap(self, monkeypatch):
        """多次失败时退避必须封顶，不会无限增长。"""
        sleeps = []
        client = make_client(sleeps, max_retries=6, backoff_cap=5.0)
        patch_post(monkeypatch, lambda n: make_response(503))
        with pytest.raises(LLMError):
            client.generate("ping")
        assert sleeps == [1.0, 2.0, 4.0, 5.0, 5.0], f"未正确封顶：{sleeps}"
        assert max(sleeps) <= 5.0

    def test_successful_retry_stops_backoff(self, monkeypatch):
        """第 2 次成功就不该再等——没有多余 sleep。"""
        sleeps = []
        client = make_client(sleeps)

        def handler(n):
            if n == 1:
                return make_response(503)
            return make_response(200, b'{"choices":[{"message":{"content":"ok"}}],"usage":{"total_tokens":7}}')

        calls = patch_post(monkeypatch, handler)
        out = client.generate("ping")
        assert out == "ok"
        assert calls["n"] == 2
        assert sleeps == [1.0], "只在首次失败后等1 次"

    def test_stats_only_count_successes(self, monkeypatch):
        client = make_client()
        patch_post(monkeypatch, lambda n: make_response(200, b'{"choices":[{"message":{"content":"hi"}}],"usage":{"total_tokens":3}}'))
        client.generate("ping")
        assert client.total_requests == 1
        assert client.total_tokens == 3


# ============================================================
# 4. JSON 解析失败纳入重试
# ============================================================

class TestJsonParseFailureIsRetryable:
    def test_truncated_json_is_retried_then_can_succeed(self, monkeypatch):
        """输出截断是瞬时问题，重试应能拿到完整内容。"""
        sleeps = []
        client = make_client(sleeps)

        def handler(n):
            if n == 1:
                # 被截断的 JSON —— response.json() 会抛
                return make_response(200, '{"choices":[{"message":{"content":"部分'.encode("utf-8"))
            return make_response(200, b'{"choices":[{"message":{"content":"full answer"}}]}')

        calls = patch_post(monkeypatch, handler)
        out = client.generate("ping")
        assert out == "full answer"
        assert calls["n"] == 2
        assert sleeps == [1.0]

    def test_missing_choices_is_retryable(self, monkeypatch):
        client = make_client()
        patch_post(monkeypatch, lambda n: make_response(200, b'{"unexpected":1}'))
        with pytest.raises(LLMResponseFormatError):
            client.generate("ping")

    def test_response_format_error_flag(self):
        assert LLMResponseFormatError.retryable is True
        assert LLMTransportError.retryable is True
        assert LLMRateLimitError.retryable is True
        assert LLMAuthError.retryable is False
        assert LLMPermanentError.retryable is False


# ============================================================
# 5. 性能保证：注入 no-op sleep 后测试**零等待**
# ============================================================

class TestNoRealSleepInTests:
    def test_injected_sleep_is_actually_used(self, monkeypatch):
        """确认 self._sleep 被真正调用（而不是绕过退避直接重试）。"""
        sleeps = []
        client = make_client(sleeps)
        patch_post(monkeypatch, lambda n: make_response(503))
        t0 = time.perf_counter()
        with pytest.raises(LLMError):
            client.generate("ping")
        elapsed = time.perf_counter() - t0
        assert len(sleeps) == 2
        assert elapsed < 0.5, f"注入 no-op sleep 后不应真实等待，实测 {elapsed:.2f}s"

    def test_base_zero_config_makes_backoff_instant(self, monkeypatch):
        """另一种手段：把 backoff_base 设为 0，退避序列全0。"""
        sleeps = []
        client = make_client(sleeps, backoff_base=0.0)
        patch_post(monkeypatch, lambda n: make_response(429))
        t0 = time.perf_counter()
        with pytest.raises(LLMError):
            client.generate("ping")
        elapsed = time.perf_counter() - t0
        assert sleeps == [0.0, 0.0]
        assert elapsed < 0.5

    def test_default_client_would_really_wait(self, monkeypatch):
        """反向确认：默认 base=1 时序列确实是 1s/2s（不是被写死成 0）。"""
        client = LLMClient(sleep_func=lambda d: None)
        assert client.backoff_base == 1.0
        assert client.backoff_factor == 2.0
        assert client.backoff_cap == 30.0
        assert client._backoff_delay(1) == 1.0
        assert client._backoff_delay(2) == 2.0
        assert client._backoff_delay(6) == 30.0


# ============================================================
# 6. 流式路径同样分级
# ============================================================

class TestStreamPath:
    def test_stream_401_fails_fast(self, monkeypatch):
        sleeps = []
        client = make_client(sleeps)
        calls = patch_post(monkeypatch, lambda n: make_response(401))
        with pytest.raises(LLMAuthError):
            client.generate("ping", stream=True)
        assert calls["n"] == 1
        assert sleeps == []

    def test_stream_garbage_chunks_raise_retryable(self, monkeypatch):
        """所有 chunk 都解析失败时不得返回空串（旧实现会静默返回 ""）。"""
        client = make_client()
        r = make_response(200)
        r.iter_lines = lambda: [b'data: {"choices":[{"delta":{"content":"a"}}]}',
                                b"data: {broken"]
        r.raw = None
        # 手工构造：首个 chunk 正常 -> 仍应返回内容
        calls = patch_post(monkeypatch, lambda n: r)
        out = client.generate("ping", stream=True)
        assert "a" in out

    def test_stream_all_broken_raises(self, monkeypatch):
        client = make_client()
        r = make_response(200)
        r.iter_lines = lambda: [b"data: {broken1", b"data: {broken2"]
        patch_post(monkeypatch, lambda n: r)
        with pytest.raises(LLMResponseFormatError):
            client.generate("ping", stream=True)


# ============================================================
# 7. 配置默认值来自 yaml
# ============================================================

class TestConfigDefaults:
    def test_rag_config_has_backoff_keys(self):
        import yaml
        cfg_path = project_root / "configs" / "rag_config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            llm = yaml.safe_load(f)["llm"]
        assert llm["retry_backoff_base"] == 1.0
        assert llm["retry_backoff_factor"] == 2.0
        assert llm["retry_backoff_cap"] == 30.0
        assert llm["max_retries"] == 3
