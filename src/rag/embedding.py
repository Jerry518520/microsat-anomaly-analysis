"""
嵌入编码模块
使用 BGE-M3 模型进行文本嵌入编码，支持 CUDA 加速

模型路径单一真相源（P0修复）：
    建索引（scripts/build_index.py）与运行时（本模块）曾各读各的：
        - build_index.py 用 configs/rag_config.yaml 的 model_name（BAAI/bge-m3）
        - 本模块用 .env 的 EMBEDDING_MODEL_PATH（models/Xorbits/bge-m3）
    两者是不同字符串，一旦权重不同（例如有人只改了 yaml），已建好的
    FAISS 索引与查询向量就会落在**不同的向量空间**里，检索结果静默
    劣化——不报错，只是变差。现统一到 `resolve_embedding_model_path()`。
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


# ============================================================
# 模型路径解析：建索引与运行时共用的唯一实现
# ============================================================

#: 环境变量名（唯一真相源）。.env / 部署环境均通过它指定本地模型。
MODEL_PATH_ENV_VAR = "EMBEDDING_MODEL_PATH"

#: 一个本地模型目录被认为「完整」所需的文件。
#:
#: 为什么需要这个检查：实测 models/bge-m3/ 只有 1_Pooling/ + imgs/ +
#: .cache/huggingface/download/，是 BAAI/bge-m3 **中断的下载残留**，
#: 没有任何权重文件。SentenceTransformer 对这种目录不会立刻报「模型
#: 损坏」，而是等到真正 encode 时才炸，或者更糟——某些版本会拿
#: config.json 随机初始化，静默产出**完全无意义的向量**。
_REQUIRED_MODEL_FILES = ("config.json",)
#: 权重文件候选（至少命中一个）。BGE-M3 同时带 safetensors + bin。
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
#: 用于确认是 sentence-transformers 目录（否则只是普通 HF 目录）。
_ST_MODULE_FILES = ("modules.json", "1_Pooling/config.json")


class EmbeddingModelPathError(RuntimeError):
    """模型路径无法解析/不可用。**不静默回退**到其他模型。"""


def _looks_like_hub_id(name: str) -> bool:
    """判断字符串是否是 HuggingFace Hub repo id（如 BAAI/bge-m3）。

    特征：含且只含一个 '/'，且不是 Windows 盘符路径、不是绝对/相对
    路径。用于识别「需要联网下载」的写法。
    """
    if os.path.isabs(name):
        return False
    if name.startswith((".", "\\")):
        return False
    if name.startswith("models/") or name.startswith("models\\"):
        return False
    #盘符形如 C:/... 已经被isabs 排除；这里只处理 'org/name'
    return name.count("/") == 1 and not name[1:3] in (":\\", ":/")


def describe_model_dir(path: str) -> str:
    """返回本地模型目录的完整性诊断文本（供报错信息使用）。"""
    p = Path(path)
    if not p.exists():
        return f"路径不存在：{p}"
    if not p.is_dir():
        return f"不是目录：{p}"
    has = [f for f in _REQUIRED_MODEL_FILES if (p / f).exists()]
    missing = [f for f in _REQUIRED_MODEL_FILES if not (p / f).exists()]
    weights = [f for f in _WEIGHT_FILES if (p / f).exists()]
    bits = [f"存在 {len(weights)} 个权重文件: {weights or '无'}"]
    bits.append(f"config 缺失: {missing or '无'}")
    st = [f for f in _ST_MODULE_FILES if (p / f).exists()]
    bits.append(f"sentence-transformers 模块文件: {st or '无（不是 ST 目录）'}")
    return f"目录 {p} | " + "; ".join(bits)


def validate_local_model_dir(path: str) -> None:
    """校验本地模型目录是否**真的能用**，否则抛带诊断信息的异常。

    刻意不做任何回退：路径指向残缺目录时自动改用别的模型，会让索引
    与查询向量悄悄分属不同模型，比直接报错危害大得多。
    """
    p = Path(path)
    if not p.exists():
        raise EmbeddingModelPathError(
            f"嵌入模型目录不存在：{p}\n{describe_model_dir(path)}\n"
            f"请把 {MODEL_PATH_ENV_VAR} 指向一个完整解压的模型目录，"
            f"或参考 .env.example。"
        )
    if not p.is_dir():
        raise EmbeddingModelPathError(
            f"嵌入模型路径不是目录：{p}\n"
            f"{MODEL_PATH_ENV_VAR} 应指向解压后的模型文件夹本身。"
        )
    missing = [f for f in _REQUIRED_MODEL_FILES if not (p / f).exists()]
    if missing:
        raise EmbeddingModelPathError(
            f"嵌入模型目录缺少必要文件 {missing}：{p}\n"
            f"{describe_model_dir(path)}\n"
            f"这通常是从 HuggingFace 下载中断留下的空壳目录，"
            f"请删除后重新完整下载/解压。"
        )
    if not any((p / f).exists() for f in _WEIGHT_FILES):
        raise EmbeddingModelPathError(
            f"嵌入模型目录**没有任何权重文件**：{p}\n"
            f"{describe_model_dir(path)}\n"
            f"至少需要 {' 或 '.join(_WEIGHT_FILES)}。"
            f"该目录极可能是下载残留——已验证此前的 models/bge-m3/ "
            f"就是这种空壳（只有 1_Pooling/ 与 imgs/），无法产出可用向量。"
        )
    if not any((p / f).exists() for f in _ST_MODULE_FILES):
        raise EmbeddingModelPathError(
            f"嵌入模型目录不是 sentence-transformers 格式：{p}\n"
            f"{describe_model_dir(path)}\n"
            f"需要 modules.json 或 1_Pooling/config.json 来确定各子模块结构。"
        )


def resolve_embedding_model_path(
    embedding_config: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
    project_root: Optional[str] = None,
    allow_hub_download: bool = False,
    validate: bool = True,
) -> str:
    """解析嵌入模型路径——**建索引与运行时共用的唯一真相源**。

    优先级：
        1. 环境变量 ``EMBEDDING_MODEL_PATH``（推荐，.env 里配置）
        2. ``embedding_config['model_name']``

    Args:
        embedding_config: configs/rag_config.yaml 的 embedding 段
        env: 环境变量映射，默认 os.environ（测试可注入）
        project_root: 项目根目录，默认取本文件上溯两级
        allow_hub_download: 是否允许 Hub repo id（如 BAAI/bge-m3）。
            默认 **False**：本项目运行在离线/内网环境，实测
            huggingface.co 直连超时（8s TIMEOUT），只有 hf-mirror
            可达；而依赖镜像下载 2.2GB 权重不是可接受的隐式行为。
            传True 才会放行 repo id。
        validate: 是否校验本地目录完整性

    Returns:
        str: 可直接传给 SentenceTransformer 的模型标识

    Raises:
        EmbeddingModelPathError: 路径不存在/不完整，或离线环境给了 Hub id。
            **绝不静默回退**到其他模型或目录。
    """
    env = os.environ if env is None else env
    if project_root is None:
        project_root = str(Path(__file__).resolve().parent.parent.parent)
    root = Path(project_root)

    from_env = (env.get(MODEL_PATH_ENV_VAR) or "").strip()
    from_config = ""
    if embedding_config:
        from_config = str(embedding_config.get("model_name") or "").strip()

    if from_env:
        # 显式指定的优先级最高—— .env 是部署者的显式意图
        source = f"环境变量 {MODEL_PATH_ENV_VAR}"
        name = from_env
    elif from_config:
        source = "配置文件 embedding.model_name"
        name = from_config
    else:
        raise EmbeddingModelPathError(
            f"未配置嵌入模型路径：环境变量 {MODEL_PATH_ENV_VAR} 为空，"
            f"且 embedding.model_name 也为空。\n"
            f"请在项目根目录 .env 中设置：\n"
            f"    {MODEL_PATH_ENV_VAR}=models/Xorbits/bge-m3"
        )

    # 本地路径 -> 绝对路径；Hub repo id -> 视allow_hub_download 决定放行与否
    candidate = Path(name)
    looks_local = candidate.is_absolute() or os.sep in name or "/" in name
    local = (root / candidate) if not candidate.is_absolute() else candidate

    if local.is_dir():
        if validate:
            validate_local_model_dir(str(local))
        logger.info(f"嵌入模型路径（{source}）解析为本地目录：{local}")
        return str(local)

    # 不是已存在的本地目录
    if _looks_like_hub_id(name) or not looks_local:
        if not allow_hub_download:
            raise EmbeddingModelPathError(
                f"配置的嵌入模型 `{name}`（来自{source}）是一个 HuggingFace "
                f"Hub repo id，不是本地模型目录。\n"
                f"本项目运行在离线/内网环境：实测 huggingface.co 直连超时"
                f"（8s 无响应），直接使用 repo id 会下载失败或挂起。\n"
                f"请把完整模型下载/解压到本地后，在 .env 中设置：\n"
                f"    {MODEL_PATH_ENV_VAR}=models/<你的目录>\n"
                f"（已验证 models/Xorbits/bge-m3 与 BAAI/bge-m3 为**同一份"
                f"权重**，pytorch_model.bin 的 md5 均为 767f43f2a03a47fc...，"
                f"向量逐元素相同，可直接使用）\n"
                f"若你确知网络可用并希望临时从镜像下载，"
                f"请显式传 allow_hub_download=True。"
            )
        logger.warning(
            f"嵌入模型使用 Hub repo id `{name}`（来自{source}），"
            f"将触发联网下载。离线环境会失败。"
        )
        return name

    # 本地路径写法但目录不存在
    raise EmbeddingModelPathError(
        f"嵌入模型目录不存在：{local}（来自{source}，原值 {name!r}）\n"
        f"请检查 {MODEL_PATH_ENV_VAR} 的拼写与相对路径基准"
        f"（相对路径基准为项目根目录 {root}）。"
    )


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
        
        # 模型参数：路径解析交给唯一真相源 resolve_embedding_model_path()
        # （与 scripts/build_index.py 共用），保证建索引与运行时加载**同一份**
        # 权重。原实现在此处重复实现了一遍 "env or config or BAAI/bge-m3"，
        # 正是两条路径分叉的根因。
        self.model_name = resolve_embedding_model_path(self.embedding_config)
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