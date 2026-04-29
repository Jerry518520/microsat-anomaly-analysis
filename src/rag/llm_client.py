"""
LLM 客户端模块
封装 LLM API 调用（支持火山引擎/DeepSeek、NVIDIA等）
"""

import os
import sys
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
import yaml
import logging
import requests
import json
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化 LLM 客户端
        
        Args:
            config_path: 配置文件路径
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
        
        # 发送请求（带重试）
        for attempt in range(self.max_retries):
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
                
            except Exception as e:
                logger.warning(f"LLM 调用失败（尝试 {attempt + 1}/{self.max_retries}）：{e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"LLM 调用最终失败：{e}")
                    raise RuntimeError(f"无法调用 LLM：{e}")
        
        return ""
    
    def _normal_request(self, request_data: Dict, headers: Dict) -> str:
        """
        普通请求（非流式）
        
        Args:
            request_data: 请求数据
            headers: 请求头
        
        Returns:
            str: 响应文本
        """
        try:
            response = requests.post(
                self.api_base,
                headers=headers,
                json=request_data,
                timeout=self.timeout
            )
            
            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"API 调用失败，状态码：{response.status_code}，响应：{response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 解析响应
            response_data = response.json()
            
            # 提取响应文本
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0].get("message", {}).get("content", "")
                
                # 更新 token 统计
                usage = response_data.get("usage", {})
                self.total_tokens += usage.get("total_tokens", 0)
                
                return content
            else:
                raise RuntimeError(f"响应格式错误：{response_data}")
                
        except requests.exceptions.Timeout:
            raise RuntimeError(f"API 调用超时（{self.timeout}s）")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"网络请求失败：{e}")
    
    def _stream_request(self, request_data: Dict, headers: Dict) -> str:
        """
        流式请求
        
        Args:
            request_data: 请求数据
            headers: 请求头
        
        Returns:
            str: 完整的响应文本
        """
        try:
            response = requests.post(
                self.api_base,
                headers=headers,
                json=request_data,
                timeout=self.timeout,
                stream=True
            )
            
            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"API 调用失败，状态码：{response.status_code}，响应：{response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 处理流式响应
            full_content = ""
            
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
                            logger.warning(f"JSON 解析失败：{e}")
                            continue
            
            return full_content
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"API 调用超时（{self.timeout}s）")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"网络请求失败：{e}")
    
    def generate_with_context(
        self, 
        query: str, 
        context: str,
        system_prompt: Optional[str] = None,
        template: Optional[str] = None
    ) -> str:
        """
        基于上下文生成响应（RAG 模式）
        
        Args:
            query: 用户查询
            context: 检索到的上下文
            system_prompt: 系统提示词
            template: 用户提示词模板
        
        Returns:
            str: 生成的响应
        """
        # 使用默认模板
        if template is None:
            template = self.config.get("prompt", {}).get("user_template", "")
        
        # 填充模板
        if template:
            prompt = template.format(
                context=context,
                query=query
            )
        else:
            # 默认提示词
            prompt = f"""请基于以下上下文回答问题：

上下文：
{context}

问题：{query}

请给出专业、准确的回答，并标注知识来源。"""
        
        # 使用默认系统提示词
        if system_prompt is None:
            system_prompt = self.config.get("prompt", {}).get("system_prompt", "")
        
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
