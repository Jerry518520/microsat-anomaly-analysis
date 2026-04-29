"""
RAG Pipeline 模块
将检索和生成串联起来，形成完整的 RAG 流程
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import yaml
import logging
from datetime import datetime

# 导入子模块
from .embedding import BGE_M3_Embedder, get_embedder
from .vectorstore import FAISSVectorStore, get_vectorstore
from .llm_client import LLMClient, get_llm_client
from .prompts import PromptTemplates, get_prompt_templates

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG Pipeline
    
    完整的 RAG 流程：
    1. 接收用户查询
    2. 检索相关文档
    3. 构建提示词
    4. 调用 LLM 生成响应
    5. 返回结果（含来源标注）
    
    支持功能：
    1. 端到端 RAG 流程
    2. 多种查询模式（异常分析、对比分析、通用查询）
    3. 性能监控
    4. 错误处理和降级
    
    示例：
        >>> pipeline = RAGPipeline(config_path="configs/rag_config.yaml")
        >>> result = pipeline.query("卫星遥测数据异常原因")
        >>> print(result["answer"])
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        embedder: Optional[BGE_M3_Embedder] = None,
        vectorstore: Optional[FAISSVectorStore] = None,
        llm_client: Optional[LLMClient] = None,
        prompt_templates: Optional[PromptTemplates] = None
    ):
        """
        初始化 RAG Pipeline
        
        Args:
            config_path: 配置文件路径
            embedder: 嵌入编码器实例
            vectorstore: 向量数据库实例
            llm_client: LLM 客户端实例
            prompt_templates: Prompt 模板实例
        """
        # 加载配置
        self.config = self._load_config(config_path)
        self.retrieval_config = self.config.get("retrieval", {})
        
        # 初始化组件（使用依赖注入或默认实例）
        self.embedder = embedder or get_embedder(config_path)
        self.vectorstore = vectorstore or get_vectorstore(config_path)
        self.llm_client = llm_client or get_llm_client(config_path)
        self.prompt_templates = prompt_templates or get_prompt_templates(config_path)
        
        # 检索参数
        self.top_k = self.retrieval_config.get("top_k", 5)
        self.score_threshold = self.retrieval_config.get("score_threshold", 0.7)
        self.max_tokens = self.retrieval_config.get("max_tokens", 2000)
        
        # 性能监控
        self.total_queries = 0
        self.total_retrieval_time = 0.0
        self.total_generation_time = 0.0
        self.total_latency = 0.0
        
        logger.info("RAG Pipeline 初始化完成")
    
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
    
    def query(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        query_type: str = "general",
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行 RAG 查询
        
        Args:
            query: 用户查询
            top_k: 返回的文档数量
            score_threshold: 相似度阈值
            query_type: 查询类型（general/anomaly/compare）
            **kwargs: 其他参数
        
        Returns:
            Dict[str, Any]: 查询结果，包含：
                - answer: 生成的答案
                - sources: 来源文档列表
                - context: 检索到的上下文
                - metadata: 查询元数据
        """
        start_time = datetime.now()
        
        try:
            # 参数处理
            top_k = top_k or self.top_k
            score_threshold = score_threshold or self.score_threshold
            
            logger.info(f"开始 RAG 查询：{query[:50]}...")
            
            # 步骤 1：检索相关文档
            retrieval_start = datetime.now()
            context, sources = self.vectorstore.search_with_context(
                query, 
                top_k=top_k,
                max_tokens=self.max_tokens
            )
            retrieval_time = (datetime.now() - retrieval_start).total_seconds()
            
            # 检查是否有检索结果
            if not context:
                logger.warning("未检索到相关文档")
                return {
                    "answer": "抱歉，知识库中未找到相关信息。请尝试更具体的查询。",
                    "sources": [],
                    "context": "",
                    "metadata": {
                        "query": query,
                        "query_type": query_type,
                        "retrieval_time": retrieval_time,
                        "generation_time": 0,
                        "total_time": (datetime.now() - start_time).total_seconds(),
                        "sources_count": 0
                    }
                }
            
            # 步骤 2：构建提示词
            if query_type == "anomaly":
                # 异常分析模式
                prompts = self.prompt_templates.get_anomaly_analysis_prompt(
                    channel_id=kwargs.get("channel_id", ""),
                    anomaly_type=kwargs.get("anomaly_type", ""),
                    anomaly_description=kwargs.get("anomaly_description", context),
                    context=context,
                    additional_info=kwargs.get("additional_info")
                )
            elif query_type == "compare":
                # 对比分析模式
                prompts = self.prompt_templates.get_comparison_prompt(
                    query=query,
                    context=context,
                    comparison_type=kwargs.get("comparison_type", "similar")
                )
            else:
                # 通用查询模式
                prompts = self.prompt_templates.get_rag_prompt(
                    query=query,
                    context=context
                )
            
            # 步骤 3：调用 LLM 生成响应
            generation_start = datetime.now()
            answer = self.llm_client.generate(
                prompt=prompts["user"],
                system_prompt=prompts["system"]
            )
            generation_time = (datetime.now() - generation_start).total_seconds()
            
            # 计算总时间
            total_time = (datetime.now() - start_time).total_seconds()
            
            # 更新统计
            self.total_queries += 1
            self.total_retrieval_time += retrieval_time
            self.total_generation_time += generation_time
            self.total_latency += total_time
            
            logger.info(f"RAG 查询完成，总时间：{total_time:.2f}s")
            
            # 返回结果
            return {
                "answer": answer,
                "sources": sources,
                "context": context,
                "metadata": {
                    "query": query,
                    "query_type": query_type,
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "sources_count": len(sources)
                }
            }
            
        except Exception as e:
            logger.error(f"RAG 查询失败：{e}")
            
            # 返回错误信息
            return {
                "answer": f"抱歉，查询过程中出现错误：{str(e)}",
                "sources": [],
                "context": "",
                "metadata": {
                    "query": query,
                    "query_type": query_type,
                    "error": str(e),
                    "total_time": (datetime.now() - start_time).total_seconds()
                }
            }
    
    def analyze_anomaly(
        self,
        channel_id: str,
        anomaly_type: str,
        anomaly_description: str,
        additional_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        异常分析专用方法
        
        Args:
            channel_id: 通道 ID
            anomaly_type: 异常类型
            anomaly_description: 异常描述
            additional_info: 额外信息
            **kwargs: 其他参数
        
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 构建查询
        query = f"通道 {channel_id} 出现 {anomaly_type} 异常：{anomaly_description}"
        
        # 执行查询
        return self.query(
            query=query,
            query_type="anomaly",
            channel_id=channel_id,
            anomaly_type=anomaly_type,
            anomaly_description=anomaly_description,
            additional_info=additional_info,
            **kwargs
        )
    
    def compare_topics(
        self,
        topic: str,
        comparison_type: str = "similar",
        **kwargs
    ) -> Dict[str, Any]:
        """
        对比分析专用方法
        
        Args:
            topic: 对比主题
            comparison_type: 对比类型（similar/different/evolution）
            **kwargs: 其他参数
        
        Returns:
            Dict[str, Any]: 对比结果
        """
        return self.query(
            query=topic,
            query_type="compare",
            comparison_type=comparison_type,
            **kwargs
        )
    
    def batch_query(
        self, 
        queries: List[str], 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        批量查询
        
        Args:
            queries: 查询列表
            **kwargs: 传递给 query 方法的参数
        
        Returns:
            List[Dict[str, Any]]: 查询结果列表
        """
        results = []
        
        for i, query in enumerate(queries):
            logger.info(f"批量查询进度：{i + 1}/{len(queries)}")
            result = self.query(query, **kwargs)
            results.append(result)
        
        return results
    
    def load_knowledge_base(self, pdf_paths: Union[str, List[str]], **kwargs) -> int:
        """
        加载知识库
        
        Args:
            pdf_paths: PDF 文件路径或路径列表
            **kwargs: 其他参数
        
        Returns:
            int: 添加的文档块数量
        """
        return self.vectorstore.add_documents(pdf_paths, **kwargs)
    
    def load_existing_knowledge_base(self, directory: Optional[str] = None) -> bool:
        """
        加载已存在的知识库
        
        Args:
            directory: 知识库目录
        
        Returns:
            bool: 是否加载成功
        """
        return self.vectorstore.load(directory)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        avg_retrieval = self.total_retrieval_time / self.total_queries if self.total_queries > 0 else 0
        avg_generation = self.total_generation_time / self.total_queries if self.total_queries > 0 else 0
        avg_latency = self.total_latency / self.total_queries if self.total_queries > 0 else 0
        
        return {
            "total_queries": self.total_queries,
            "total_retrieval_time": self.total_retrieval_time,
            "total_generation_time": self.total_generation_time,
            "total_latency": self.total_latency,
            "average_retrieval_time": avg_retrieval,
            "average_generation_time": avg_generation,
            "average_latency": avg_latency,
            "vectorstore_stats": self.vectorstore.get_stats(),
            "llm_stats": self.llm_client.get_stats()
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.total_queries = 0
        self.total_retrieval_time = 0.0
        self.total_generation_time = 0.0
        self.total_latency = 0.0
        self.llm_client.reset_stats()
        logger.info("统计信息已重置")
    
    def __repr__(self) -> str:
        """对象表示"""
        return (
            f"RAGPipeline("
            f"queries={self.total_queries}, "
            f"vectorstore={self.vectorstore}, "
            f"llm={self.llm_client})"
        )


# 全局实例（单例模式）
_global_pipeline = None

def get_rag_pipeline(config_path: Optional[str] = None) -> RAGPipeline:
    """
    获取全局 RAG Pipeline 实例（单例模式）
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        RAGPipeline: RAG Pipeline 实例
    """
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = RAGPipeline(config_path)
    return _global_pipeline


def query_knowledge(query: str, **kwargs) -> Dict[str, Any]:
    """
    快速查询知识库（使用全局实例）
    
    Args:
        query: 用户查询
        **kwargs: 传递给 query 方法的参数
    
    Returns:
        Dict[str, Any]: 查询结果
    """
    pipeline = get_rag_pipeline()
    return pipeline.query(query, **kwargs)


def analyze_anomaly(
    channel_id: str,
    anomaly_type: str,
    anomaly_description: str,
    **kwargs
) -> Dict[str, Any]:
    """
    快速异常分析（使用全局实例）
    
    Args:
        channel_id: 通道 ID
        anomaly_type: 异常类型
        anomaly_description: 异常描述
        **kwargs: 其他参数
    
    Returns:
        Dict[str, Any]: 分析结果
    """
    pipeline = get_rag_pipeline()
    return pipeline.analyze_anomaly(
        channel_id=channel_id,
        anomaly_type=anomaly_type,
        anomaly_description=anomaly_description,
        **kwargs
    )


if __name__ == "__main__":
    """模块测试"""
    print("=== RAG Pipeline 测试 ===")
    
    # 测试 Pipeline
    pipeline = RAGPipeline()
    print(f"RAG Pipeline：{pipeline}")
    
    # 测试统计信息
    stats = pipeline.get_stats()
    print(f"统计信息：{stats}")
    
    print("测试完成！")
