"""
LLM 客户端模块
封装 LLM API 调用（支持火山引擎/DeepSeek、NVIDIA等）
"""

import os
import sys
from typing import List, Dict, Any, Optional, Generator, Callable
from pathlib import Path
import yaml
import logging
import requests
import json
import time
from datetime import datetime

# 自动加载 .env 文件（项目根目录）
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# 配置日志（由应用入口统一配置 basicConfig）
logger = logging.getLogger(__name__)


# ============================================================
# 重试策略：可重试性分级 + 指数退避
# ============================================================

class LLMError(RuntimeError):
    """LLM 调用失败的基类。

    继承 RuntimeError 而非 Exception，是为了保持与原实现
    `raise RuntimeError(f"无法调用 LLM：{e}")` 的兼容性——
    下游（src/rag/pipeline.py:233、src/streaming/stage2.py:225 等）
    一律用 `except Exception` 兜底，行为不变；而需要精细处理的
    新代码可以只捕获 LLMError。
    """

    #: 该错误是否值得重试。子类覆盖。
    retryable = True


class LLMAuthError(LLMError):
    """401 / 403 鉴权失败。

    **不重试**（retryable=False）：凭据错误不会因为等待而自愈。
    原实现对 401 也连打 3 次，实测每次往返约 0.5s，纯粹白等；
    加上退避后更会变成 1+2=3s 的无谓等待。
    """

    retryable = False


class LLMRateLimitError(LLMError):
    """429 限流 / 5xx 服务端错误。可重试，且应遵循服务端 Retry-After。"""

    retryable = True

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMPermanentError(LLMError):
    """400/404/422 等其余 4xx：请求本身有问题，重试无意义。"""

    retryable = False


class LLMResponseFormatError(LLMError):
    """响应无法解析（JSON 截断 / choices 缺失 / 流式chunk 非法）。

    可重试：常由输出超长被截断、或网关偶发返回半截body 引起，
    重试一次通常能拿到完整内容。
    """

    retryable = True


class LLMTransportError(LLMError):
    """网络层失败（超时 / 连接重置 / DNS）。可重试。"""

    retryable = True


#: 鉴权类状态码——出现即判定「重试无意义」
AUTH_STATUS_CODES = frozenset({401, 403})
#: 明确可重试的状态码（限流 + 服务端错误）
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


def _parse_retry_after(response) -> Optional[float]:
    """从 429/503 响应的 Retry-After 头取秒数。

    只支持「整数秒」形式（RFC 7231 的 HTTP-date 形式在 OpenAI 兼容
    端点上基本不出现，且需要邮件头日期解析，收益不抵复杂度）。
    解析失败返回 None，调用方退回本地指数退避。
    """
    try:
        raw = (response.headers or {}).get("Retry-After")
        if raw is None:
            return None
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        return None


def compute_backoff_delay(
    attempt: int,
    base: float = 1.0,
    factor: float = 2.0,
    cap: float = 30.0,
) -> float:
    """指数退避时长（秒）。

    Args:
        attempt: 已失败次数，从 1 开始。第 1 次重试等base 秒。
        base: 首次退避时长（秒）
        factor: 每次翻倍因子
        cap: 单次退避上限（秒）

    Returns:
        float: 本次重试前应等待的秒数，序列为 base, base*factor,
        base*factor^2 ...，触顶后恒为 cap。

    例（base=1, factor=2, cap=30）：1, 2, 4, 8, 16, 30, 30 ...
    """
    if attempt < 1:
        attempt = 1
    delay = base * (factor ** (attempt - 1))
    return float(min(delay, cap))


def _safe_format(template: str, **values) -> str:
    """填充提示词模板，对调用方未提供的占位符用「未提供」兜底而非抛 KeyError。

    为什么需要（P0-A5）：
        `str.format` 遇到模板里存在、但本次调用未传的命名占位符会直接抛
        `KeyError`。项目里 `configs/rag_config.yaml` 的 `prompt.user_template`
        含 4 个占位符，而 `generate_with_context` 原实现只传了 2 个，
        一调用就崩。本函数让「模板与调用方参数不同步」这种漂移
        退化为可观测的占位文字，而不是线上异常。

    另兼容两种情况：
        - 模板带 str.format 的格式说明符（如 {score:.2f}）时，正则只取
          字段名，补齐后交给 format 处理，格式说明符不丢失；
        - 出现未知占位符时同样填「未提供」，不中断生成。
    """
    import re as _re

    # 注意：只对「兜底填充」的占位符做 str() 转换，调用方已提供的值一律
    # 原样保留。否则 score=1.5 会被转成 '1.5'，导致 {score:.2f} 这类带格式
    # 说明符的字段抛 ValueError: Unknown format code 'f'。
    known = dict(values)
    for name in _re.findall(r"\{(\w+)", template):
        if name not in known:
            known[name] = "未提供"
    return template.format(**known)


class LLMClient:
    """
    LLM 客户端（支持火山引擎/DeepSeek、NVIDIA等OpenAI兼容API）
    
    支持功能：
    1. 多平台API调用
    2. 流式响应
    3. 重试机制
    4. 错误处理
    5. 性能监控
    
    示例：
        >>> client = LLMClient(config_path="configs/rag_config.yaml")
        >>> response = client.generate("请分析卫星异常原因")
        >>> print(response)
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        sleep_func: Optional[Callable[[float], None]] = None,
    ):
        """
        初始化 LLM 客户端

        Args:
            config_path: 配置文件路径
            sleep_func: 退避等待的可注入实现，默认 `time.sleep`。
                测试注入 no-op（`lambda _: None`）即可断言退避序列
                而不真的等待，从而**测试绝对不变慢**。
        """
        # 加载配置
        self.config = self._load_config(config_path)
        self.llm_config = self.config.get("llm", {})
        
        # API 配置
        self.provider = self.llm_config.get("provider", "volcengine")
        self.api_base = self.llm_config.get("api_base", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
        self.model = self.llm_config.get("model", "deepseek-v3-2-251201")
        
        # API 认证（优先从环境变量读取，也支持配置文件直传）
        api_key_env = self.llm_config.get("api_key_env", "VOLCENGINE_API_KEY")
        self.api_key = os.environ.get(api_key_env, "")
        # 如果环境变量没有，尝试从配置文件直接读取
        if not self.api_key:
            self.api_key = self.llm_config.get("api_key", "")
        
        if not self.api_key:
            logger.warning(f"未找到 API Key，请设置环境变量：{api_key_env}")
        
        # 请求参数
        self.temperature = self.llm_config.get("temperature", 0.1)
        self.max_tokens = self.llm_config.get("max_tokens", 1024)
        self.top_p = self.llm_config.get("top_p", 0.9)
        
        # 超时设置
        self.timeout = self.llm_config.get("timeout", 30)
        self.max_retries = self.llm_config.get("max_retries", 3)

        # 退避设置（指数退避：base * factor^(n-1)，封顶 cap）
        self.backoff_base = float(self.llm_config.get("retry_backoff_base", 1.0))
        self.backoff_factor = float(self.llm_config.get("retry_backoff_factor", 2.0))
        self.backoff_cap = float(self.llm_config.get("retry_backoff_cap", 30.0))
        # 可注入的等待实现——测试传 no-op 即可零耗时验证退避序列
        self._sleep = sleep_func if sleep_func is not None else time.sleep
        
        # 性能监控
        self.total_requests = 0
        self.total_tokens = 0
        self.total_latency = 0.0
        
        logger.info(f"LLM 客户端初始化完成，平台：{self.provider}，模型：{self.model}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "rag_config.yaml"
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功：{config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"配置文件未找到：{config_path}，使用默认配置")
            return {}
        except Exception as e:
            logger.error(f"配置文件加载失败：{e}")
            return {}
    
    def _prepare_messages(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        准备消息列表
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            history: 历史对话
        
        Returns:
            List[Dict]: 消息列表
        """
        messages = []
        
        # 添加系统提示词
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 添加历史对话
        if history:
            messages.extend(history)
        
        # 添加用户提示词
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        生成响应
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            history: 历史对话
            temperature: 温度参数
            max_tokens: 最大 token 数
            stream: 是否流式响应
        
        Returns:
            str: 生成的响应文本
        """
        # 准备消息
        messages = self._prepare_messages(prompt, system_prompt, history)
        
        # 准备请求数据
        request_data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": self.top_p,
            "stream": stream
        }
        
        # 准备请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 发送请求（带重试 + 指数退避）
        #
        # 修复记录：原实现 3 次重试**紧邻排列、零 sleep**，实测 401 场景
        # 总耗时 1.58s、429/网络错误场景 0.00s——服务端在收到 3 次瞬时
        # 背靠背请求后只会更严格限流，退避缺失使重试近乎无效；而 401
        # 这类鉴权错误重试 100 次结果也一样，纯属白等。
        # 现在：
        #   - 401/403 -> LLMAuthError(retryable=False)，首次即失败
        #   - 429/5xx/超时/网络错误/JSON 截断 -> 指数退避后重试
        last_error: Optional[LLMError] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = datetime.now()

                if stream:
                    response = self._stream_request(request_data, headers)
                else:
                    response = self._normal_request(request_data, headers)

                # 计算延迟
                latency = (datetime.now() - start_time).total_seconds()

                # 更新统计
                self.total_requests += 1
                self.total_latency += latency

                logger.info(f"LLM 调用成功，延迟：{latency:.2f}s")
                return response

            except LLMError as e:
                last_error = e

                if not e.retryable:
                    # 鉴权失败：重试无意义，立即失败（不 sleep）
                    logger.error(
                        f"LLM 调用失败（不可重试错误 {type(e).__name__}，"
                        f"尝试 {attempt}/{self.max_retries}）：{e}"
                    )
                    raise

                if attempt == self.max_retries:
                    logger.error(f"LLM 调用最终失败（已重试 {attempt} 次）：{e}")
                    raise

                delay = self._backoff_delay(attempt, e)
                logger.warning(
                    f"LLM 调用失败（尝试 {attempt}/{self.max_retries}）：{e}"
                    f"；{delay:.2f}s 后重试"
                )
                self._sleep(delay)

            except Exception as e:
                # 兜底：非 LLMError 的意外异常不重试（语义未知，重试可能
                # 放大副作用），直接向上抛，调用方按原行为处理。
                logger.error(f"LLM 调用异常（不重试）：{type(e).__name__}: {e}")
                raise

        # 理论上不可达：循环内要么 return、要么 raise
        raise RuntimeError(f"无法调用 LLM：{last_error}")

    def _backoff_delay(self, attempt: int, error: Optional[LLMError] = None) -> float:
        """计算第attempt 次失败后的退避时长。

        若服务端给了 `Retry-After`（429/503 常见）且不超过 cap，
        优先采用服务端建议值——它比本地猜的更准。
        """
        server_hint = getattr(error, "retry_after", None)
        if server_hint is not None and server_hint > 0:
            return float(min(server_hint, self.backoff_cap))
        return compute_backoff_delay(
            attempt,
            base=self.backoff_base,
            factor=self.backoff_factor,
            cap=self.backoff_cap,
        )
    
    def _raise_for_status(self, response):
        """按状态码抛出**分级**异常，交给 generate() 决定是否退避重试。

        修复记录：原实现对所有非 200 一律 `raise RuntimeError`，而
        generate() 对所有异常一律重试 3 次。结果是 401（key 写错、
        余额不足）也会被连打 3 次——凭据问题不会因为等待而自愈。
        """
        status = response.status_code
        # 截断响应体，避免超长错误页灌满日志
        body = (response.text or "")[:500]
        message = f"API 调用失败，状态码：{status}，响应：{body}"

        if status in AUTH_STATUS_CODES:
            hint = "API Key 无效/无权限，请检查 .env 中的密钥与账户状态"
            logger.error(f"{message} | {hint}")
            raise LLMAuthError(f"{message} | {hint}")

        if status == 429:
            retry_after = _parse_retry_after(response)
            logger.error(f"{message} | 触发限流，Retry-After={retry_after}")
            raise LLMRateLimitError(message, retry_after=retry_after)

        if status in RETRYABLE_STATUS_CODES:
            retry_after = _parse_retry_after(response)
            logger.warning(f"{message} | 可重试状态码，Retry-After={retry_after}")
            raise LLMRateLimitError(message, retry_after=retry_after)

        # 4xx 其余（400 请求体错、404 模型名错、422 校验失败）等——重试
        # 同样无意义，按不可重试处理，避免白等 3 次。
        logger.error(f"{message} | 客户端错误，重试无意义")
        raise LLMPermanentError(f"{message} | 客户端错误，重试无意义")

    def _normal_request(self, request_data: Dict, headers: Dict) -> str:
        """
        普通请求（非流式）

        Args:
            request_data: 请求数据
            headers: 请求头

        Returns:
            str: 响应文本

        Raises:
            LLMAuthError: 401/403，不可重试
            LLMRateLimitError: 429，遵循 Retry-After
            LLMError: 其他 5xx / 4xx
            LLMResponseFormatError: 响应 JSON 无法解析（可重试）
            LLMTransportError: 超时 / 网络异常（可重试）
        """
        try:
            response = requests.post(
                self.api_base,
                headers=headers,
                json=request_data,
                timeout=self.timeout
            )

            # 检查响应状态（分级异常）
            if response.status_code != 200:
                self._raise_for_status(response)

            # 解析响应（JSON 截断 -> LLMResponseFormatError，可重试）
            try:
                response_data = response.json()
            except (ValueError, json.JSONDecodeError) as e:
                raise LLMResponseFormatError(
                    f"响应 JSON 解析失败（可能被截断）：{e}；"
                    f"响应前 200 字符：{(response.text or '')[:200]}"
                )

            # 提取响应文本
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0].get("message", {}).get("content", "")

                # 更新 token 统计
                usage = response_data.get("usage", {})
                self.total_tokens += usage.get("total_tokens", 0)

                return content
            else:
                raise LLMResponseFormatError(f"响应格式错误（缺choices）：{response_data}")

        except requests.exceptions.Timeout:
            raise LLMTransportError(f"API 调用超时（{self.timeout}s）")
        except requests.exceptions.RequestException as e:
            raise LLMTransportError(f"网络请求失败：{e}")
    
    def _stream_request(self, request_data: Dict, headers: Dict) -> str:
        """
        流式请求
        
        Args:
            request_data: 请求数据
            headers: 请求头
        
        Returns:
            str: 完整的响应文本

        Raises:
            LLMAuthError / LLMPermanentError: 不可重试
            LLMRateLimitError: 429/5xx，可重试
            LLMTransportError: 超时 / 网络异常，可重试
        """
        try:
            response = requests.post(
                self.api_base,
                headers=headers,
                json=request_data,
                timeout=self.timeout,
                stream=True
            )

            # 检查响应状态（分级异常）
            if response.status_code != 200:
                self._raise_for_status(response)
            
            # 处理流式响应
            full_content = ""
            bad_chunks = 0
            
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    
                    # 跳过空行
                    if not line.strip():
                        continue
                    
                    # 解析 SSE 数据
                    if line.startswith("data: "):
                        data = line[6:]
                        
                        # 检查结束标志
                        if data.strip() == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data)

                            # 提取内容
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")

                                if content:
                                    full_content += content
                                    # 可以在这里实现实时输出
                                    # print(content, end="", flush=True)

                        except json.JSONDecodeError as e:
                            # 修复记录：原实现 `continue` 静默丢弃损坏 chunk。
                            # 若首个 chunk 就被截断，full_content 恒为空串，
                            # generate() 会把 "" 当成功返回，LLM 静默失效。
                            # 现在记录并计数，收尾时若一个 chunk 都没解出
                            # 则抛可重试异常，让上层重试而非拿到空答案。
                            logger.warning(f"流式 chunk JSON 解析失败：{e}")
                            bad_chunks += 1
                            continue

            if not full_content and bad_chunks:
                raise LLMResponseFormatError(
                    f"流式响应解析失败：{bad_chunks} 个 chunk 全部无法解析，"
                    f"未获得任何内容（可能是响应被截断）"
                )

            return full_content

        except requests.exceptions.Timeout:
            raise LLMTransportError(f"API 调用超时（{self.timeout}s）")
        except requests.exceptions.RequestException as e:
            raise LLMTransportError(f"网络请求失败：{e}")
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        template: Optional[str] = None,
        channel_id: Optional[str] = None,
        anomaly_type: Optional[str] = None,
        anomaly_description: Optional[str] = None,
    ) -> str:
        """
        基于上下文生成响应（RAG 模式）

        Args:
            query: 用户查询
            context: 检索到的上下文
            system_prompt: 系统提示词
            template: 用户提示词模板
            channel_id: 异常所在遥测通道（如 CADC0874），填入模板的 {channel_id}
            anomaly_type: 异常类型（如 磁力计饱和），填入模板的 {anomaly_type}
            anomaly_description: 异常的文字描述，填入模板的 {anomaly_description}

        Returns:
            str: 生成的响应

        修复记录（P0-A5）：
            原实现只向模板传 context 与 query 两个变量，而
            configs/rag_config.yaml 的 prompt.user_template 实际含有 4 个占位符
            —— {context} / {channel_id} / {anomaly_type} / {anomaly_description}。
            调用本函数会直接抛 `KeyError: 'channel_id'`（实测复现）。
            根因是 str.format 对缺失的命名占位符抛 KeyError，与 query 无关
            （模板本身未使用 {query}，多传的 query 不会报错，少传的会报错）。

            现在按「模板实际用到什么就填什么」的原则补齐三个可选参数，并
            对仍缺失的占位符用占位文字兜底（见 _safe_format），保证任何
            模板组合下都不会再因占位符缺失而崩溃。
        """
        # 使用默认模板
        if template is None:
            template = self.config.get("prompt", {}).get("user_template", "")

        # 填充模板
        if template:
            prompt = _safe_format(
                template,
                context=context,
                query=query,
                channel_id=channel_id or "未提供",
                anomaly_type=anomaly_type or "未提供",
                anomaly_description=anomaly_description or "未提供",
            )
        else:
            # 默认提示词
            prompt = f"""请基于以下上下文回答问题：

上下文：
{context}

问题：{query}

请给出专业、准确的回答，并标注知识来源。"""

        # 使用默认系统提示词
        # 两级回落（修复 7 段式防幻觉在 RAG 路径上静默失效）：
        #   1. 调用方显式传入的非空 system_prompt
        #   2. 配置 prompt.system_prompt
        #   3. src/rag/prompts.py 的代码默认值（748 字符，含
        #      【通道定位】【来源】等 7 段式约束）
        # 原实现只做 1、2 两级，且用 get(k, "") 使「键存在但值为空串」时
        # 直接取空 -> 7 段式 system 消息被静默丢弃，LLM 退化为自由发挥。
        if not system_prompt:
            system_prompt = self.config.get("prompt", {}).get("system_prompt") or ""
        if not system_prompt:
            try:
                from .prompts import PromptTemplates
                system_prompt = PromptTemplates().get_system_prompt()
            except Exception as e:  # 提示词模块异常不应阻断生成
                logger.warning(f"未能加载默认 system_prompt（7 段式约束将缺失）：{e}")
                system_prompt = ""

        # 生成响应
        return self.generate(prompt, system_prompt)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            Dict: 统计信息
        """
        avg_latency = self.total_latency / self.total_requests if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_latency": self.total_latency,
            "average_latency": avg_latency,
            "model": self.model,
            "provider": self.provider
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.total_requests = 0
        self.total_tokens = 0
        self.total_latency = 0.0
        logger.info("统计信息已重置")
    
    def __repr__(self) -> str:
        """对象表示"""
        return (
            f"LLMClient("
            f"model={self.model}, "
            f"provider={self.provider}, "
            f"requests={self.total_requests})"
        )


# 全局实例（单例模式）
_global_llm_client = None

def get_llm_client(config_path: Optional[str] = None) -> LLMClient:
    """
    获取全局 LLM 客户端实例（单例模式）
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        LLMClient: LLM 客户端实例
    """
    global _global_llm_client
    if _global_llm_client is None:
        _global_llm_client = LLMClient(config_path)
    return _global_llm_client


def generate_response(prompt: str, **kwargs) -> str:
    """
    快速生成响应（使用全局实例）
    
    Args:
        prompt: 用户提示词
        **kwargs: 传递给 generate 方法的参数
    
    Returns:
        str: 生成的响应
    """
    client = get_llm_client()
    return client.generate(prompt, **kwargs)


if __name__ == "__main__":
    """模块测试"""
    print("=== NVIDIA LLM 客户端测试 ===")
    
    # 测试客户端
    client = LLMClient()
    print(f"LLM 客户端：{client}")
    
    # 测试统计信息
    stats = client.get_stats()
    print(f"统计信息：{stats}")
    
    print("测试完成！")
