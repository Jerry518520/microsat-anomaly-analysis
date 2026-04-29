"""
简单的 RAG 系统测试脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试模块导入"""
    print("=== 测试模块导入 ===")
    
    try:
        from src.rag.embedding import BGE_M3_Embedder
        print("✓ 嵌入模块导入成功")
    except Exception as e:
        print(f"✗ 嵌入模块导入失败: {e}")
        return False
    
    try:
        from src.rag.vectorstore import FAISSVectorStore
        print("✓ 向量数据库模块导入成功")
    except Exception as e:
        print(f"✗ 向量数据库模块导入失败: {e}")
        return False
    
    try:
        from src.rag.llm_client import NVIDIALLMClient
        print("✓ LLM 客户端模块导入成功")
    except Exception as e:
        print(f"✗ LLM 客户端模块导入失败: {e}")
        return False
    
    try:
        from src.rag.prompts import PromptTemplates
        print("✓ Prompt 模块导入成功")
    except Exception as e:
        print(f"✗ Prompt 模块导入失败: {e}")
        return False
    
    try:
        from src.rag.pipeline import RAGPipeline
        print("✓ Pipeline 模块导入成功")
    except Exception as e:
        print(f"✗ Pipeline 模块导入失败: {e}")
        return False
    
    return True

def test_cuda():
    """测试 CUDA 加速"""
    print("\n=== 测试 CUDA 加速 ===")
    
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU 数量: {torch.cuda.device_count()}")
            print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
            print(f"CUDA 版本: {torch.version.cuda}")
            return True
        else:
            print("CUDA 不可用，将使用 CPU")
            return False
    except Exception as e:
        print(f"CUDA 测试失败: {e}")
        return False

def test_embedding():
    """测试嵌入编码器"""
    print("\n=== 测试嵌入编码器 ===")
    
    try:
        from src.rag.embedding import BGE_M3_Embedder
        
        # 初始化嵌入编码器
        embedder = BGE_M3_Embedder()
        print(f"嵌入编码器初始化成功")
        print(f"设备: {embedder.device}")
        print(f"嵌入维度: {embedder.get_embedding_dimension()}")
        
        # 测试编码
        test_texts = ["卫星遥测数据异常", "磁力计校准失败"]
        embeddings = embedder.encode(test_texts)
        print(f"编码成功，嵌入形状: {embeddings.shape}")
        
        # 测试相似度
        similarity = embedder.similarity(test_texts[0], test_texts[1])
        print(f"相似度计算成功: {similarity:.4f}")
        
        return True
    except Exception as e:
        print(f"嵌入编码器测试失败: {e}")
        return False

def test_vectorstore():
    """测试向量数据库"""
    print("\n=== 测试向量数据库 ===")
    
    try:
        from src.rag.vectorstore import FAISSVectorStore
        
        # 初始化向量数据库
        vectorstore = FAISSVectorStore()
        print(f"向量数据库初始化成功")
        
        # 测试统计信息
        stats = vectorstore.get_stats()
        print(f"统计信息: {stats}")
        
        return True
    except Exception as e:
        print(f"向量数据库测试失败: {e}")
        return False

def test_llm_client():
    """测试 LLM 客户端"""
    print("\n=== 测试 LLM 客户端 ===")
    
    try:
        from src.rag.llm_client import NVIDIALLMClient
        
        # 初始化 LLM 客户端
        client = NVIDIALLMClient()
        print(f"LLM 客户端初始化成功")
        print(f"模型: {client.model}")
        print(f"提供商: {client.provider}")
        
        # 测试统计信息
        stats = client.get_stats()
        print(f"统计信息: {stats}")
        
        return True
    except Exception as e:
        print(f"LLM 客户端测试失败: {e}")
        return False

def test_prompts():
    """测试 Prompt 模板"""
    print("\n=== 测试 Prompt 模板 ===")
    
    try:
        from src.rag.prompts import PromptTemplates
        
        # 初始化 Prompt 模板
        templates = PromptTemplates()
        print(f"Prompt 模板初始化成功")
        
        # 测试系统提示词
        system_prompt = templates.get_system_prompt()
        print(f"系统提示词长度: {len(system_prompt)} 字符")
        
        # 测试用户提示词
        user_prompt = templates.get_user_prompt(
            channel_id="CADC0874",
            anomaly_type="数值异常",
            anomaly_description="遥测值超出正常范围",
            context="测试上下文"
        )
        print(f"用户提示词长度: {len(user_prompt)} 字符")
        
        return True
    except Exception as e:
        print(f"Prompt 模板测试失败: {e}")
        return False

def test_pipeline():
    """测试 RAG Pipeline"""
    print("\n=== 测试 RAG Pipeline ===")
    
    try:
        from src.rag.pipeline import RAGPipeline
        
        # 初始化 Pipeline
        pipeline = RAGPipeline()
        print(f"RAG Pipeline 初始化成功")
        
        # 测试统计信息
        stats = pipeline.get_stats()
        print(f"统计信息: {stats}")
        
        return True
    except Exception as e:
        print(f"RAG Pipeline 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始 RAG 系统测试...\n")
    
    # 运行所有测试
    tests = [
        ("模块导入", test_imports),
        ("CUDA 加速", test_cuda),
        ("嵌入编码器", test_embedding),
        ("向量数据库", test_vectorstore),
        ("LLM 客户端", test_llm_client),
        ("Prompt 模板", test_prompts),
        ("RAG Pipeline", test_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"测试 {test_name} 出现异常: {e}")
            results.append((test_name, False))
    
    # 打印测试结果汇总
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    
    passed = 0
    failed = 0
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("=" * 50)
    print(f"总计: {len(results)} 项测试")
    print(f"通过: {passed} 项")
    print(f"失败: {failed} 项")
    print("=" * 50)
    
    if failed == 0:
        print("\n🎉 所有测试通过！RAG 系统准备就绪。")
        return 0
    else:
        print(f"\n⚠️  有 {failed} 项测试失败，请检查相关模块。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
