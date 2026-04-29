import sys
import os

# 设置标准输出编码
sys.stdout.reconfigure(encoding='utf-8')

# 测试嵌入编码器
try:
    from src.rag.embedding import BGE_M3_Embedder
    
    print("=== 测试嵌入编码器 ===")
    e = BGE_M3_Embedder()
    print(f"设备: {e.device}")
    print(f"维度: {e.get_embedding_dimension()}")
    
    # 测试编码
    emb = e.encode(['测试'])
    print(f"编码成功，形状: {emb.shape}")
    
    print("嵌入编码器测试通过！")
    
except Exception as ex:
    print(f"嵌入编码器测试失败: {ex}")
    import traceback
    traceback.print_exc()

# 测试向量数据库
try:
    from src.rag.vectorstore import FAISSVectorStore
    
    print("\n=== 测试向量数据库 ===")
    v = FAISSVectorStore()
    stats = v.get_stats()
    print(f"统计信息: {stats}")
    
    print("向量数据库测试通过！")
    
except Exception as ex:
    print(f"向量数据库测试失败: {ex}")
    import traceback
    traceback.print_exc()

# 测试 LLM 客户端
try:
    from src.rag.llm_client import NVIDIALLMClient
    
    print("\n=== 测试 LLM 客户端 ===")
    c = NVIDIALLMClient()
    print(f"模型: {c.model}")
    print(f"提供商: {c.provider}")
    
    print("LLM 客户端测试通过！")
    
except Exception as ex:
    print(f"LLM 客户端测试失败: {ex}")
    import traceback
    traceback.print_exc()

# 测试 Prompt 模板
try:
    from src.rag.prompts import PromptTemplates
    
    print("\n=== 测试 Prompt 模板 ===")
    t = PromptTemplates()
    system_prompt = t.get_system_prompt()
    print(f"系统提示词长度: {len(system_prompt)} 字符")
    
    print("Prompt 模板测试通过！")
    
except Exception as ex:
    print(f"Prompt 模板测试失败: {ex}")
    import traceback
    traceback.print_exc()

# 测试 Pipeline
try:
    from src.rag.pipeline import RAGPipeline
    
    print("\n=== 测试 RAG Pipeline ===")
    p = RAGPipeline()
    stats = p.get_stats()
    print(f"统计信息: {stats}")
    
    print("RAG Pipeline 测试通过！")
    
except Exception as ex:
    print(f"RAG Pipeline 测试失败: {ex}")
    import traceback
    traceback.print_exc()

print("\n=== 所有测试完成 ===")
