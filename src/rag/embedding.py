"""
嵌入编码模块
使用 BAAI/bge-m3 模型进行文本嵌入编码，支持 CUDA 加速
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np
import yaml
import logging

# 设置 HuggingFace 镜像源（国内加速）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 修复 PyTorch 2.10+ 在 Windows 上的 getpass.getuser() → pwd 模块缺失问题
import getpass
if not hasattr(getpass, '_original_getuser'):
    getpass._original_getuser = getpass.getuser
def _win_getuser():
    return os.environ.get('USERNAME', os.environ.get('USER', 'default'))
getpass.getuser = _win_getuser

# 尝试导入相关库，提供友好的错误提示
try:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
except ImportError as e:
    print(f"[ERROR] 缺少依赖包：{e}")
    print("请安装：pip install torch sentence-transformers transformers")
    sys.exit(1)

# 配置日志（由应用入口统一配置 basicConfig）
logger = logging.getLogger(__name__)


class BGE_M3_Embedder:
    """
    BGE-M3 嵌入编码器
    
    支持功能：
    1. CUDA 加速（自动检测并回退到 CPU）
    2. 批量编码（优化 GPU 利用率）
    3. 嵌入向量归一化（cosine 相似度优化）
    4. 缓存机制（避免重复编码）
    
    示例：
        >>> embedder = BGE_M3_Embedder(config_path="configs/rag_config.yaml")
        >>> embeddings = embedder.encode(["卫星遥测数据异常", "磁力计校准失败"])
    """
    
    def __init__(self, config_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化嵌入编码器
        
        Args:
            config_path: 配置文件路径，默认为 configs/rag_config.yaml
            device: 指定设备（cuda/cpu），默认为自动检测
        """
        # 加载配置
        self.config = self._load_config(config_path)
        self.embedding_config = self.config.get("embedding", {})
        
        # 设备检测
        self.device = device or self._detect_device()
        logger.info(f"初始化 BGE-M3 嵌入编码器，设备：{self.device}")
        
        # 模型参数
        model_name = self.embedding_config.get("model_name", "BAAI/bge-m3")
        # 如果是相对路径且存在，解析为绝对路径
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / model_name
        if model_path.is_dir():
            self.model_name = str(model_path)
        else:
            self.model_name = model_name
        self.batch_size = self.embedding_config.get("batch_size", 32)
        self.normalize_embeddings = self.embedding_config.get("normalize_embeddings", True)
        self.max_length = self.embedding_config.get("max_length", 512)
        self.cache_dir = self.embedding_config.get("cache_dir", "data/embeddings_cache")
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 加载模型
        self.model = self._load_model()
        
        # 缓存字典（内存缓存）
        self.cache: Dict[str, np.ndarray] = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            # 默认路径：项目根目录下的 configs/rag_config.yaml
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
    
    def _detect_device(self) -> str:
        """自动检测最佳设备"""
        # 优先使用配置中的设备
        config_device = self.embedding_config.get("device", "").lower()
        if config_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif config_device == "cpu":
            return "cpu"
        
        # 自动检测
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"检测到 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            return "cuda"
        else:
            logger.warning("CUDA 不可用，将使用 CPU（性能较差）")
            return "cpu"
    
    def _load_model(self) -> SentenceTransformer:
        """加载 SentenceTransformer 模型"""
        try:
            logger.info(f"加载模型：{self.model_name}")
            
            # 检查缓存目录
            cache_dir = self.cache_dir if self.embedding_config.get("use_cache", True) else None
            
            # 加载模型
            model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=cache_dir
            )
            
            # 设置模型参数
            model.max_seq_length = self.max_length
            
            # 测试模型
            test_text = ["测试文本"]
            test_embedding = model.encode(test_text, normalize_embeddings=self.normalize_embeddings)
            logger.info(f"模型测试成功，嵌入维度：{test_embedding.shape[1]}")
            
            return model
            
        except Exception as e:
            logger.error(f"模型加载失败：{e}")
            raise RuntimeError(f"无法加载模型 {self.model_name}: {e}")
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: Optional[int] = None,
        show_progress_bar: Optional[bool] = None,
        normalize: Optional[bool] = None
    ) -> np.ndarray:
        """
        编码文本为嵌入向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小，默认为配置中的值
            show_progress_bar: 是否显示进度条
            normalize: 是否归一化嵌入向量
        
        Returns:
            numpy.ndarray: 嵌入向量，形状为 (n_texts, embedding_dim)
        """
        # 参数处理
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = batch_size or self.batch_size
        show_progress_bar = show_progress_bar or self.embedding_config.get("show_progress_bar", True)
        normalize = normalize if normalize is not None else self.normalize_embeddings
        
        # 检查缓存
        if len(texts) == 1 and texts[0] in self.cache:
            logger.debug(f"从缓存获取嵌入：{texts[0][:50]}...")
            return self.cache[texts[0]].reshape(1, -1)
        
        # 批量编码
        try:
            logger.debug(f"编码 {len(texts)} 个文本，batch_size={batch_size}")
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            
            # 更新缓存（仅缓存单个文本）
            if len(texts) == 1:
                self.cache[texts[0]] = embeddings[0]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"文本编码失败：{e}")
            # 返回零向量作为降级
            if len(texts) == 1:
                return np.zeros((1, self.model.get_sentence_embedding_dimension()))
            else:
                return np.zeros((len(texts), self.model.get_sentence_embedding_dimension()))
    
    def encode_with_metadata(
        self, 
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        metadata_fields: List[str] = None
    ) -> Dict[str, Any]:
        """
        编码文档（带元数据）
        
        Args:
            documents: 文档列表，每个文档是字典，包含文本和元数据
            text_field: 文本字段名
            metadata_fields: 需要保留的元数据字段
        
        Returns:
            dict: 包含嵌入向量和元数据的结果
        """
        if metadata_fields is None:
            metadata_fields = []
        
        # 提取文本
        texts = [doc.get(text_field, "") for doc in documents]
        
        # 提取元数据
        metadata = []
        for doc in documents:
            meta = {field: doc.get(field, "") for field in metadata_fields}
            metadata.append(meta)
        
        # 编码文本
        embeddings = self.encode(texts)
        
        return {
            "embeddings": embeddings,
            "metadata": metadata,
            "texts": texts
        }
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的余弦相似度
        
        Args:
            text1: 文本1
            text2: 文本2
        
        Returns:
            float: 余弦相似度 (0-1)
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        # 计算余弦相似度
        similarity = np.dot(emb1[0], emb2[0]) / (
            np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0])
        )
        
        return float(similarity)
    
    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        return self.model.get_embedding_dimension()
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("嵌入缓存已清空")
    
    def __repr__(self) -> str:
        """对象表示"""
        return f"BGE_M3_Embedder(model={self.model_name}, device={self.device}, dim={self.get_embedding_dimension()})"


# 全局实例（单例模式）
_global_embedder = None

def get_embedder(config_path: Optional[str] = None) -> BGE_M3_Embedder:
    """
    获取全局嵌入编码器实例（单例模式）
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        BGE_M3_Embedder: 嵌入编码器实例
    """
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = BGE_M3_Embedder(config_path)
    return _global_embedder


def encode_texts(texts: List[str], **kwargs) -> np.ndarray:
    """
    快速编码文本（使用全局实例）
    
    Args:
        texts: 文本列表
        **kwargs: 传递给 encode 方法的参数
    
    Returns:
        np.ndarray: 嵌入向量
    """
    embedder = get_embedder()
    return embedder.encode(texts, **kwargs)


if __name__ == "__main__":
    """模块测试"""
    print("=== BGE-M3 嵌入编码器测试 ===")
    
    # 测试嵌入编码器
    embedder = BGE_M3_Embedder()
    print(f"嵌入编码器: {embedder}")
    
    # 测试编码
    test_texts = [
        "卫星遥测数据异常",
        "磁力计校准失败可能导致测量偏差",
        "太阳能电池板输出功率下降"
    ]
    
    embeddings = embedder.encode(test_texts)
    print(f"嵌入向量形状: {embeddings.shape}")
    print(f"嵌入维度: {embedder.get_embedding_dimension()}")
    
    # 测试相似度
    sim = embedder.similarity(test_texts[0], test_texts[1])
    print(f"相似度 ('{test_texts[0]}' vs '{test_texts[1]}'): {sim:.4f}")
    
    print("测试完成！")