"""
API Key 可用性探测脚本（最小代价：只发 1 条最短请求）

安全约束：
  - 本脚本只输出"已配置且可用 / 未配置 / 无效（HTTP 状态码或错误类型）"
  - **绝不打印、写入或记录 API Key 原文**
  - .env 内容不落盘

用法：
  python scripts/probe_apikey_v3.py

退出码：
  0 = Key 可用；1 = Key 未配置或不可用；2 = 网络/其他异常
"""

import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
import os  # noqa: E402
import yaml  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

VERDICT = {"status": "unknown", "detail": ""}


def main():
    cfg_path = PROJECT_ROOT / "configs" / "rag_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    llm_cfg = cfg.get("llm", {})

    api_base = llm_cfg.get("api_base", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
    model = llm_cfg.get("model", "deepseek-v3-2-251201")
    key_env = llm_cfg.get("api_key_env", "VOLCENGINE_API_KEY")

    api_key = os.environ.get(key_env, "")
    if not api_key:
        api_key = llm_cfg.get("api_key", "")

    print("=" * 60)
    print("API Key 探测（不输出 Key 原文）")
    print("=" * 60)
    print(f"provider/model : {llm_cfg.get('provider', 'volcengine')} / {model}")
    print(f"api_base       : {api_base}")
    print(f"key_env 变量名 : {key_env}")

    if not api_key or not api_key.strip():
        print(f"Key 来源       : 未配置（环境变量与配置文件均无值）")
        VERDICT.update(status="not_configured", detail="env var and config both empty")
        print("\n>>> 结论：Key 未配置")
        print("=" * 60)
        return 1

    # 只报告长度量级与前缀形态，不泄露原文
    print(f"Key 来源       : 已配置")
    print(f"Key 长度       : {len(api_key)} 字符（不输出内容）")
    print(f"Key 是否含空白 : {api_key != api_key.strip()}")

    # 最小代价请求：max_tokens=1，让模型只回一个 token
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }

    try:
        resp = requests.post(api_base, headers=headers, json=payload, timeout=30)
    except requests.exceptions.Timeout:
        VERDICT.update(status="error", detail="request timeout after 30s")
        print("\n>>> 结论：请求超时（30s），未能验证")
        print("=" * 60)
        return 2
    except Exception as e:
        VERDICT.update(status="error", detail=f"{type(e).__name__}")
        print(f"\n>>> 结论：请求异常（{type(e).__name__}），未能验证")
        print("=" * 60)
        return 2

    print(f"HTTP 状态码    : {resp.status_code}")

    if resp.status_code == 200:
        VERDICT.update(status="available", detail="HTTP 200")
        # 只打印响应体长度与是否非空，不打印业务内容
        body = resp.text
        print(f"响应体长度     : {len(body)} 字符")
        print("\n>>> 结论：Key 已配置且可用")
        print("=" * 60)
        return 0

    # 非 200：只摘取错误码，绝不回显 body（可能含敏感回显）
    err_type = "unknown"
    try:
        j = resp.json()
        err = j.get("error", {}) if isinstance(j, dict) else {}
        err_type = err.get("code") or err.get("type") or err.get("message") or "unknown"
        # err_type 可能含 key 片段，做长度截断并只保留前 80 字符
        err_type = str(err_type)[:80]
    except Exception:
        err_type = f"non-json body ({len(resp.text)} chars)"

    # 非 200：可能是 Key 无效，也可能只是模型 ID 不对。
    # 用一次免费的 /models 列表请求区分二者（该接口不产生推理费用）。
    # 从 api_base 推导 /models 端点：<...>/chat/completions -> <...>/models
    models_url = api_base.replace("/chat/completions", "/models")
    auth_ok = None
    try:
        r2 = requests.get(models_url, headers=headers, timeout=30)
        auth_ok = (r2.status_code == 200)
        print(f"/models 状态码  : {r2.status_code}（用于区分 Key 无效 vs 模型ID无效）")
    except Exception as e:
        print(f"/models 请求异常: {type(e).__name__}")

    if auth_ok:
        print("\n>>> 结论：Key 已通过鉴权，但模型 ID 不可用（HTTP 404 InvalidEndpointOrModel.NotFound）")
        print("          即：Key 本身有效；配置文件中的 model 字段在火山方舟上不存在或无权限。")
        VERDICT.update(
            status="invalid_model",
            detail=f"HTTP {resp.status_code}: {err_type}; /models auth OK -> key valid, model id invalid",
        )
        print("=" * 60)
        return 1

    VERDICT.update(status="invalid", detail=f"HTTP {resp.status_code}: {err_type}")
    print(f"错误码         : {err_type}")
    print(f"\n>>> 结论：Key 无效（返回 HTTP {resp.status_code}）")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
