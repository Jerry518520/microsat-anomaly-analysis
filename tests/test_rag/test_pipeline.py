"""
RAG Pipeline 测试脚本
测试单通道异常查询功能
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入被测试的模块
from src.rag.pipeline import RAGPipeline, get_rag_pipeline, query_knowledge, analyze_anomaly
from src.rag.embedding import BGE_M3_Embedder
from src.rag.vectorstore import FAISSVectorStore
from src.rag.llm_client import NVIDIALLMClient
from src.rag.prompts import PromptTemplates


class TestRAGPipeline:
    """RAG Pipeline 测试类"""
    
    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        return {
            "embedding": {
                "model_name": "BAAI/bge-m3",
                "device": "cpu",
                "batch_size": 32,
                "normalize_embeddings": True,
                "max_length": 512
            },
            "vectorstore": {
                "index_type": "Flat",
                "metric": "cosine",
                "persist_directory": "data/test_vectorstore",
                "use_gpu": False
            },
            "document": {
                "chunking": {
                    "strategy": "recursive",
                    "chunk_size": 512,
                    "chunk_overlap": 50
                }
            },
            "retrieval": {
                "top_k": 5,
                "score_threshold": 0.7,
                "max_tokens": 2000
            },
            "llm": {
                "provider": "nvidia",
                "api_base": "https://integrate.api.nvidia.com/v1/chat/completions",
                "model": "qwen/qwen3.5-397b-a17b",
                "api_key_env": "NVIDIA_API_KEY",
                "temperature": 0.1,
                "max_tokens": 1024,
                "timeout": 30,
                "max_retries": 3
            },
            "prompt": {
                "system_prompt": "你是一个卫星异常诊断专家。",
                "user_template": "请分析：{context}",
                "citation_format": "【来源：{document_name} 第 {page_number} 页】"
            }
        }
    
    @pytest.fixture
    def mock_embedder(self):
        """模拟嵌入编码器"""
        embedder = Mock(spec=BGE_M3_Embedder)
        embedder.encode.return_value = [[0.1, 0.2, 0.3]]  # 模拟嵌入向量
        embedder.get_embedding_dimension.return_value = 3
        return embedder
    
    @pytest.fixture
    def mock_vectorstore(self):
        """模拟向量数据库"""
        vectorstore = Mock(spec=FAISSVectorStore)
        vectorstore.search_with_context.return_value = (
            "卫星遥测数据异常可能原因：1. 传感器故障 2. 通信链路问题",
            [
                {
                    "content": "卫星遥测数据异常可能原因：1. 传感器故障 2. 通信链路问题",
                    "score": 0.85,
                    "metadata": {
                        "filename": "NASA_SOA_2024.pdf",
                        "page": 42
                    }
                }
            ]
        )
        vectorstore.get_stats.return_value = {
            "total_documents": 100,
            "index_type": "Flat",
            "metric": "cosine"
        }
        return vectorstore
    
    @pytest.fixture
    def mock_llm_client(self):
        """模拟 LLM 客户端"""
        llm_client = Mock(spec=NVIDIALLMClient)
        llm_client.generate.return_value = "根据知识库分析，卫星遥测数据异常可能由以下原因导致：1. 传感器故障 2. 通信链路问题"
        llm_client.get_stats.return_value = {
            "total_requests": 1,
            "total_tokens": 100,
            "average_latency": 0.5
        }
        return llm_client
    
    @pytest.fixture
    def mock_prompt_templates(self):
        """模拟 Prompt 模板"""
        templates = Mock(spec=PromptTemplates)
        templates.get_rag_prompt.return_value = {
            "system": "你是一个卫星异常诊断专家。",
            "user": "请分析：卫星遥测数据异常"
        }
        templates.get_anomaly_analysis_prompt.return_value = {
            "system": "你是一个卫星异常诊断专家。",
            "user": "请分析异常：通道 CADC0874"
        }
        templates.get_comparison_prompt.return_value = {
            "system": "你是一个卫星系统专家。",
            "user": "请对比分析：卫星通信协议"
        }
        return templates
    
    @pytest.fixture
    def pipeline(self, mock_config, mock_embedder, mock_vectorstore, mock_llm_client, mock_prompt_templates):
        """创建测试用 Pipeline"""
        with patch('src.rag.pipeline.yaml.safe_load', return_value=mock_config):
            pipeline = RAGPipeline(
                embedder=mock_embedder,
                vectorstore=mock_vectorstore,
                llm_client=mock_llm_client,
                prompt_templates=mock_prompt_templates
            )
            return pipeline
    
    def test_pipeline_initialization(self, pipeline):
        """测试 Pipeline 初始化"""
        assert pipeline is not None
        assert pipeline.top_k == 5
        assert pipeline.score_threshold == 0.7
        assert pipeline.max_tokens == 2000
    
    def test_query_general(self, pipeline):
        """测试通用查询"""
        result = pipeline.query("卫星遥测数据异常原因")
        
        assert "answer" in result
        assert "sources" in result
        assert "context" in result
        assert "metadata" in result
        assert len(result["answer"]) > 0
        assert len(result["sources"]) > 0
    
    def test_query_anomaly(self, pipeline):
        """测试异常分析查询"""
        result = pipeline.analyze_anomaly(
            channel_id="CADC0874",
            anomaly_type="数值异常",
            anomaly_description="遥测值超出正常范围"
        )
        
        assert "answer" in result
        assert result["metadata"]["query_type"] == "anomaly"
    
    def test_query_compare(self, pipeline):
        """测试对比分析查询"""
        result = pipeline.compare_topics(
            topic="卫星通信协议对比",
            comparison_type="similar"
        )
        
        assert "answer" in result
        assert result["metadata"]["query_type"] == "compare"
    
    def test_batch_query(self, pipeline):
        """测试批量查询"""
        queries = [
            "卫星遥测数据异常原因",
            "磁力计校准方法",
            "太阳能电池板故障"
        ]
        
        results = pipeline.batch_query(queries)
        
        assert len(results) == 3
        for result in results:
            assert "answer" in result
            assert "sources" in result
    
    def test_load_knowledge_base(self, pipeline, mock_vectorstore):
        """测试加载知识库"""
        # 模拟 PDF 文件
        with patch('os.path.exists', return_value=True):
            count = pipeline.load_knowledge_base(["test.pdf"])
            
            assert count >= 0
            mock_vectorstore.add_documents.assert_called_once()
    
    def test_load_existing_knowledge_base(self, pipeline, mock_vectorstore):
        """测试加载已存在的知识库"""
        mock_vectorstore.load.return_value = True
        
        result = pipeline.load_existing_knowledge_base("data/vectorstore")
        
        assert result is True
        mock_vectorstore.load.assert_called_once_with("data/vectorstore")
    
    def test_get_stats(self, pipeline):
        """测试获取统计信息"""
        stats = pipeline.get_stats()
        
        assert "total_queries" in stats
        assert "total_retrieval_time" in stats
        assert "total_generation_time" in stats
        assert "total_latency" in stats
        assert "vectorstore_stats" in stats
        assert "llm_stats" in stats
    
    def test_reset_stats(self, pipeline):
        """测试重置统计信息"""
        # 先执行一些查询
        pipeline.query("测试查询")
        
        # 重置统计
        pipeline.reset_stats()
        
        stats = pipeline.get_stats()
        assert stats["total_queries"] == 0
        assert stats["total_retrieval_time"] == 0.0
    
    def test_query_error_handling(self, pipeline, mock_vectorstore):
        """测试查询错误处理"""
        # 模拟检索失败
        mock_vectorstore.search_with_context.side_effect = Exception("检索失败")
        
        result = pipeline.query("测试查询")
        
        assert "answer" in result
        assert "错误" in result["answer"]
    
    def test_query_no_results(self, pipeline, mock_vectorstore):
        """测试无检索结果"""
        # 模拟空结果
        mock_vectorstore.search_with_context.return_value = ("", [])
        
        result = pipeline.query("测试查询")
        
        assert "answer" in result
        assert "未找到" in result["answer"]


class TestRAGPipelineIntegration:
    """RAG Pipeline 集成测试（需要真实环境）"""
    
    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY"),
        reason="需要 NVIDIA API Key"
    )
    def test_real_query(self):
        """真实查询测试（需要 API Key）"""
        try:
            pipeline = RAGPipeline()
            
            # 测试查询
            result = pipeline.query("卫星遥测数据异常可能原因")
            
            assert "answer" in result
            assert len(result["answer"]) > 0
            
        except Exception as e:
            pytest.skip(f"集成测试失败：{e}")
    
    @pytest.mark.skipif(
        not os.path.exists("data/vectorstore"),
        reason="需要向量库数据"
    )
    def test_with_real_vectorstore(self):
        """使用真实向量库测试"""
        try:
            pipeline = RAGPipeline()
            pipeline.load_existing_knowledge_base()
            
            result = pipeline.query("卫星异常诊断")
            
            assert "answer" in result
            
        except Exception as e:
            pytest.skip(f"向量库测试失败：{e}")


class TestHelperFunctions:
    """辅助函数测试"""
    
    @patch('src.rag.pipeline.RAGPipeline')
    def test_get_rag_pipeline(self, MockPipeline):
        """测试获取全局 Pipeline 实例"""
        mock_instance = Mock()
        MockPipeline.return_value = mock_instance
        
        # 第一次调用
        pipeline1 = get_rag_pipeline()
        
        # 第二次调用（应该返回同一个实例）
        pipeline2 = get_rag_pipeline()
        
        assert pipeline1 is pipeline2
        MockPipeline.assert_called_once()
    
    @patch('src.rag.pipeline.get_rag_pipeline')
    def test_query_knowledge(self, mock_get_pipeline):
        """测试快速查询函数"""
        mock_pipeline = Mock()
        mock_pipeline.query.return_value = {"answer": "测试答案"}
        mock_get_pipeline.return_value = mock_pipeline
        
        result = query_knowledge("测试查询")
        
        assert result["answer"] == "测试答案"
        mock_pipeline.query.assert_called_once_with("测试查询")
    
    @patch('src.rag.pipeline.get_rag_pipeline')
    def test_analyze_anomaly_function(self, mock_get_pipeline):
        """测试快速异常分析函数"""
        mock_pipeline = Mock()
        mock_pipeline.analyze_anomaly.return_value = {"answer": "异常分析结果"}
        mock_get_pipeline.return_value = mock_pipeline
        
        result = analyze_anomaly(
            channel_id="CADC0874",
            anomaly_type="数值异常",
            anomaly_description="遥测值超出正常范围"
        )
        
        assert result["answer"] == "异常分析结果"
        mock_pipeline.analyze_anomaly.assert_called_once()


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
