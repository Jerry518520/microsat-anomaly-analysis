"""
向量数据库模块
使用 FAISS 构建和管理向量索引，支持 PDF 文档加载、切分和检索
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import yaml
import logging
import pickle
from datetime import datetime

# 尝试导入相关库
try:
    import faiss
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document
except ImportError as e:
    print(f"[ERROR] 缺少依赖包：{e}")
    print("请安装：pip install faiss-cpu langchain langchain-community langchain-text-splitters pypdf")
    sys.exit(1)

# 导入嵌入模块
from .embedding import BGE_M3_Embedder, get_embedder
from .retrieval_policy import default_priority_caps, is_low_quality, select_candidates

# 关键词侧（BM25）。可选依赖：jieba 缺失时 bm25.tokenize自动退化为
# 「英文正则 + CJK bigram」，不会让检索失败。
from .bm25 import BM25Index

# 配置日志（由应用入口统一配置 basicConfig）
logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS 向量数据库
    
    支持功能：
    1. PDF 文档加载和切分
    2. 嵌入向量生成和索引构建
    3. 向量检索（支持 top-k 和阈值过滤）
    4. 索引持久化和加载
    5. GPU 加速（可选）
    
    示例：
        >>> vectorstore = FAISSVectorStore(config_path="configs/rag_config.yaml")
        >>> vectorstore.add_documents(["doc1.pdf", "doc2.pdf"])
        >>> results = vectorstore.search("卫星异常", top_k=5)
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        embedder: Optional[BGE_M3_Embedder] = None
    ):
        """
        初始化向量数据库
        
        Args:
            config_path: 配置文件路径
            embedder: 嵌入编码器实例，如果为 None 则自动创建
        """
        # 加载配置
        self.config = self._load_config(config_path)
        self.vectorstore_config = self.config.get("vectorstore", {})
        self.document_config = self.config.get("document", {})
        self.retrieval_config = self.config.get("retrieval", {})
        
        # 初始化嵌入编码器
        self.embedder = embedder or get_embedder(config_path)
        
        # 向量库参数
        self.index_type = self.vectorstore_config.get("index_type", "Flat")
        self.metric = self.vectorstore_config.get("metric", "cosine")
        self.persist_directory = self.vectorstore_config.get("persist_directory", "data/vectorstore")
        self.use_gpu = self.vectorstore_config.get("use_gpu", False)
        self.gpu_id = self.vectorstore_config.get("gpu_id", 0)
        
        # 文档处理参数
        chunking_config = self.document_config.get("chunking", {})
        self.chunk_size = chunking_config.get("chunk_size", 512)
        self.chunk_overlap = chunking_config.get("chunk_overlap", 50)
        self.chunking_strategy = chunking_config.get("strategy", "recursive")
        
        # 检索参数
        self.top_k = self.retrieval_config.get("top_k", 5)
        self.score_threshold = self.retrieval_config.get("score_threshold", 0.3)
        # 候选池大小：priority 保留席位 + 每源上限需要更深的候选才能填满 top_k
        self.candidate_pool_k = self.retrieval_config.get("candidate_pool_k", 40)
        # 相对竞争区宽度：分数低于 top_score - rel_delta 的候选不享受 priority 保留席位
        self.rel_delta = self.retrieval_config.get("rel_delta", 0.12)
        # 低质量 chunk 过滤（PDF 残片/目录页）
        self.filter_low_quality = self.retrieval_config.get("filter_low_quality", True)
        self.min_chunk_chars = self.retrieval_config.get("min_chunk_chars", 80)
        self.max_junk_ratio = self.retrieval_config.get("max_junk_ratio", 0.55)
        # priority -> 同源入选上限
        self.priority_caps = self.retrieval_config.get("priority_caps") or default_priority_caps()
        self.reserve_priority_max = self.retrieval_config.get("reserve_priority_max", 2)
        self.max_reserved = self.retrieval_config.get("max_reserved")
        # 置信度提示阈值：top1 低于此值时在结果里标注「置信度低」而非静默丢弃
        self.low_confidence_threshold = self.retrieval_config.get(
            "low_confidence_threshold", 0.45
        )
        # --- 混合检索（向量 + BM25 关键词）---
        # 实测依据与选型理由见 _merge_keyword_candidates 的docstring。
        # 关键约束：keyword_boost 实测必须 <=0.02，否则 MRR 下降（0.940 -> 0.917）。
        self.hybrid_search = self.retrieval_config.get("hybrid_search", False)
        self.keyword_pool_k = self.retrieval_config.get("keyword_pool_k", 40)
        self.keyword_boost = self.retrieval_config.get("keyword_boost", 0.02)
        self._bm25: Optional[BM25Index] = None

        # 初始化组件
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.document_metadata: List[Dict[str, Any]] = []
        
        # 创建持久化目录
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # 初始化文本分割器
        self.text_splitter = self._create_text_splitter()
        
        logger.info(f"FAISS 向量数据库初始化完成，索引类型：{self.index_type}")
    
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
    
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """创建文本分割器"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
    
    def _load_pdf(self, pdf_path: str) -> List[Document]:
        """
        加载 PDF 文档
        
        Args:
            pdf_path: PDF 文件路径
        
        Returns:
            List[Document]: 文档列表
        """
        try:
            logger.info(f"加载 PDF：{pdf_path}")
            
            # 使用 PyPDFLoader 加载
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # 添加元数据
            for doc in documents:
                doc.metadata["source"] = pdf_path
                doc.metadata["filename"] = Path(pdf_path).name
                doc.metadata["load_time"] = datetime.now().isoformat()
            
            logger.info(f"PDF 加载完成，共 {len(documents)} 页")
            return documents
            
        except Exception as e:
            logger.error(f"PDF 加载失败：{pdf_path} - {e}")
            return []
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """
        切分文档
        
        Args:
            documents: 原始文档列表
        
        Returns:
            List[Document]: 切分后的文档块列表
        """
        try:
            logger.info(f"切分文档，原始文档数：{len(documents)}")
            
            # 使用文本分割器切分
            chunks = self.text_splitter.split_documents(documents)
            
            # 为每个块添加索引信息
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i
                chunk.metadata["chunk_size"] = len(chunk.page_content)
            
            logger.info(f"文档切分完成，共 {len(chunks)} 个块")
            return chunks
            
        except Exception as e:
            logger.error(f"文档切分失败：{e}")
            return []
    
    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        创建 FAISS 索引
        
        Args:
            embeddings: 嵌入向量矩阵
        
        Returns:
            faiss.Index: FAISS 索引对象
        """
        try:
            dimension = embeddings.shape[1]
            n_vectors = embeddings.shape[0]
            
            logger.info(f"创建 FAISS 索引，维度：{dimension}，向量数：{n_vectors}")
            
            # 根据配置选择索引类型
            if self.index_type == "Flat":
                # 精确搜索（适合小规模数据）
                if self.metric == "cosine":
                    # 使用内积（余弦相似度需要归一化向量）
                    index = faiss.IndexFlatIP(dimension)
                else:
                    # L2 距离
                    index = faiss.IndexFlatL2(dimension)
            
            elif self.index_type.startswith("IVF"):
                # IVF 索引（适合大规模数据）
                nlist = int(self.index_type[3:]) if len(self.index_type) > 3 else 100
                
                if self.metric == "cosine":
                    quantizer = faiss.IndexFlatIP(dimension)
                    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                else:
                    quantizer = faiss.IndexFlatL2(dimension)
                    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
                
                # 训练索引
                index.train(embeddings)
            
            else:
                # 默认使用 Flat 索引
                logger.warning(f"未知索引类型：{self.index_type}，使用 Flat 索引")
                if self.metric == "cosine":
                    index = faiss.IndexFlatIP(dimension)
                else:
                    index = faiss.IndexFlatL2(dimension)
            
            # GPU 加速（可选）
            if self.use_gpu and faiss.get_num_gpus() > 0:
                try:
                    logger.info(f"启用 GPU 加速，GPU ID：{self.gpu_id}")
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)
                except Exception as e:
                    logger.warning(f"GPU 加速失败，回退到 CPU：{e}")
            
            # 添加向量
            index.add(embeddings)
            
            logger.info(f"FAISS 索引创建完成，总向量数：{index.ntotal}")
            return index
            
        except Exception as e:
            logger.error(f"FAISS 索引创建失败：{e}")
            raise RuntimeError(f"无法创建 FAISS 索引：{e}")
    
    def add_documents(
        self, 
        pdf_paths: Union[str, List[str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        添加文档到向量数据库
        
        Args:
            pdf_paths: PDF 文件路径或路径列表
            metadata: 额外的元数据
        
        Returns:
            int: 添加的文档块数量
        """
        # 处理输入
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]
        
        all_chunks = []
        
        # 加载和切分每个 PDF
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF 文件不存在：{pdf_path}")
                continue
            
            # 加载 PDF
            documents = self._load_pdf(pdf_path)
            if not documents:
                continue
            
            # 切分文档
            chunks = self._split_documents(documents)
            if not chunks:
                continue
            
            # 添加额外元数据
            if metadata:
                for chunk in chunks:
                    chunk.metadata.update(metadata)
            
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("没有有效的文档块可添加")
            return 0
        
        # 提取文本并编码
        texts = [chunk.page_content for chunk in all_chunks]
        logger.info(f"开始编码 {len(texts)} 个文档块...")
        
        embeddings = self.embedder.encode(texts)
        
        # 创建或更新索引
        if self.index is None:
            self.index = self._create_faiss_index(embeddings)
        else:
            # 添加到现有索引
            self.index.add(embeddings)
        
        # 保存文档和元数据
        self.documents.extend(all_chunks)
        self.document_metadata.extend([chunk.metadata for chunk in all_chunks])
        
        logger.info(f"文档添加完成，当前总文档块数：{len(self.documents)}")
        
        # 自动保存
        if self.vectorstore_config.get("auto_save", True):
            self.save()
        
        return len(all_chunks)
    
    def _ensure_bm25(self) -> Optional[BM25Index]:
        """懒构建 BM25 索引：首次混合检索时才建，避免纯向量用户付出代价。

        语料或文档集合变化后需失效重建（:meth:`_invalidate_bm25`）。
        """
        if not self.hybrid_search:
            return None
        if self._bm25 is not None:
            return None if self._bm25.is_empty() else self._bm25
        if not self.documents:
            return None
        try:
            corpus = [
                d if isinstance(d, str) else getattr(d, "page_content", str(d))
                for d in self.documents
            ]
            self._bm25 = BM25Index(corpus)
            logger.info(
                f"BM25 关键词索引已构建：{self._bm25.size} chunk / "
                f"{self._bm25.vocabulary_size} 词元"
            )
        except Exception as e:
            logger.warning(f"BM25 索引构建失败，退回纯向量检索：{e}")
            self._bm25 = None
        return self._bm25

    def _invalidate_bm25(self) -> None:
        """文档集合变化后让BM25 索引失效。"""
        self._bm25 = None

    def _merge_keyword_candidates(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """把 BM25 关键词信号叠加到向量候选上（混合检索的第二路召回）。

        ## 为什么是「小权重加分」而不是 RRF 排名融合

        实测（24 条标注查询，评测脚本 ``scripts/eval_retrieval_quality.py``，
        选型探针 ``scripts/probe_fusion_weight.py`` 与 ``probe_fusion_blend.py``）
        横向比较了 4 类融合公式：

        ====================  ==========  ==========  ==============
        融合方式                 usable@5MRR        kw_recall
        ====================  ==========  ==========  ==============
        纯向量（改动前）        0.875       0.940       0.681
        RRF（排名融合）          **0.917**   0.902       0.710
        分数线性加权            0.917       0.902       0.688
        **BM25 小权重加分**     0.875       **0.940**   **0.724**
        ====================  ==========  ==========  ==============

        RRF 能把召回率提到 0.917，但**MRR 从 0.940 掉到 0.902**：
        它完全抛弃向量分数的语义序、改用纯排名序。而 BGE-M3 在同源
        细粒度相关性上的方向是可靠的（相邻两名差~0.01 但排序稳定），
        丢掉它得不偿失——这属于「用首位精度换召回」。

        ``keyword_boost=0.02``（加到向量分上）是实测中**唯一
        MRR 不降、kw_recall 反而 +0.043** 的方案，故采用。

        ## 残留局限（如实记录，未解决）

        本方案**不提升 usable@5**。原因是：纯 BM25 命中但向量侧未召回的
        chunk，其向量分为 0.0，加 0.02 后仍排在候选池尾部，进不了 top-5。
        要真正救回这批 chunk，必须让 cross-encoder rerank 接管打分
        （实测 usable@5 可达 1.000），但代价是 MRR 0.940 -> 0.807，
        超出「不许降低准确率换召回」的约束，本次**不做**。

        ## 与 RetrievalPolicy 既有约定的关系

        本方法只产出候选与分数，**不参与排序决策**：
        priority 仍只决定「入选资格」，最终顺序仍严格按 ``score`` 降序
        （由 :func:`~src.rag.retrieval_policy.select_candidates` 执行）。
        关键词加分只是让「字面匹配强」的 chunk 获得应得的分数补偿。

        Args:
            query: 原始查询文本。
            vector_results: 向量侧候选列表（原地追加关键词侧补充项）。
        """
        bm25 = self._ensure_bm25()
        if bm25 is None or not vector_results or self.keyword_boost <= 0:
            return vector_results

        n_docs = len(self.documents)
        already = {
            r["_pos"] for r in vector_results if r.get("_pos") is not None
        }
        # 关键词侧最高分，用于把BM25 分数压到 [0,1] 再乘权重，
        # 避免 BM25 的原始量纲（实测 0~48）压倒余弦相似度（0.45~0.67）。
        kw_hits = bm25.top_n(query, self.keyword_pool_k)
        if not kw_hits:
            return vector_results
        kw_max = max(s for _, s in kw_hits) or 1.0

        added = 0
        for pos, kw_score in kw_hits:
            if pos in already or pos >= n_docs or kw_score <= 0:
                continue
            content = self._content_at(pos)
            if not content:
                continue
            if self.filter_low_quality and is_low_quality(
                content, self.min_chunk_chars, self.max_junk_ratio
            ):
                continue
            metadata = (
                self.document_metadata[pos]
                if pos < len(self.document_metadata) else {}
            )
            vector_results.append({
                "content": content,
                # 向量侧未召回 -> 基线分0.0，仅靠关键词加分进入候选池。
                # 实测这类 chunk 仍排在池尾（见上方「残留局限」），
                # 保留它们是为了让 keyword_match 信号可见、可供后续策略使用。
                "score": self.keyword_boost * (kw_score / kw_max),
                "metadata": metadata,
                "keyword_match": kw_score,
                "_pos": pos,
            })
            added += 1

        # 对**已在向量候选池内**的 chunk 按关键词强度加分：
        # 这是真正生效的部分（它们有可比的向量分，加分不会被池尾淹没）。
        for r in vector_results:
            pos = r.get("_pos")
            if pos is None:
                continue
            hit = dict(kw_hits).get(pos)
            if hit:
                r["score"] = r["score"] + self.keyword_boost * (hit / kw_max)
                r["keyword_match"] = hit

        if added:
            logger.info(
                f"混合检索：关键词侧补充 {added} 个候选"
                f"（BM25 top{self.keyword_pool_k}，boost={self.keyword_boost}）"
            )
        return vector_results

    def _content_at(self, pos: int) -> Optional[str]:
        """取下标 ``pos`` 的 chunk 文本，兼容 Document / str / 其他类型。"""
        if pos < 0 or pos >= len(self.documents):
            return None
        doc = self.documents[pos]
        if isinstance(doc, str):
            return doc
        if hasattr(doc, "page_content"):
            return doc.page_content
        return str(doc)

    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            score_threshold: 相似度阈值
        
        Returns:
            List[Dict]: 检索结果列表，每个结果包含：
                - content: 文档内容
                - score: 相似度分数
                - metadata: 文档元数据
                - low_confidence: top1 分数偏低时为 True（结果仍返回，不静默丢弃）
        """
        if self.index is None or len(self.documents) == 0:
            logger.warning("向量库为空，请先添加文档")
            return []
        
        # 参数处理（用 is None 判断，避免显式传 0 被 `or` 覆盖）
        top_k = top_k if top_k is not None else self.top_k
        score_threshold = (score_threshold if score_threshold is not None
                           else self.score_threshold)
        
        try:
            # 编码查询
            query_embedding = self.embedder.encode(query)
            
            # 取更深的候选池：priority 保留席位与每源上限需要足够候选才能填满 top_k
            search_k = max(self.candidate_pool_k, top_k * 8, top_k)
            scores, indices = self.index.search(query_embedding, search_k)
            
            # 处理结果
            results = []
            n_filtered_quality = 0
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS 返回 -1 表示无效结果
                    continue
                
                score = float(score)
                
                # 应用阈值过滤
                if score < score_threshold:
                    continue
                
                # 获取文档（兼容LangChain Document对象和纯字符串）
                doc = self.documents[idx]
                metadata = self.document_metadata[idx] if idx < len(self.document_metadata) else {}
                
                # 提取内容：Document对象用page_content，字符串直接用
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                elif isinstance(doc, str):
                    content = doc
                else:
                    content = str(doc)
                
                # 过滤低质量片段：PDF 换页截断产生的孤立句子、目录页残片。
                # 这类片段维度低、极易与任意查询相似，曾多次占据 top1。
                if self.filter_low_quality and is_low_quality(
                    content, self.min_chunk_chars, self.max_junk_ratio
                ):
                    n_filtered_quality += 1
                    continue
                
                results.append({
                    "content": content,
                    "score": score,
                    "metadata": metadata,
                    "_pos": int(idx),
                })
            
            if n_filtered_quality:
                logger.info(f"过滤低质量片段 {n_filtered_quality} 条（min_chars={self.min_chunk_chars}）")
            
            # 混合检索：把 BM25 关键词信号叠加进候选池。
            # 必须在 score_threshold 过滤**之后**做——阈值是余弦相似度
            # 阈值（作用于向量侧），BM25 分数量纲不同，不可同一把尺子量。
            if self.hybrid_search:
                results = self._merge_keyword_candidates(query, results)
            
            # priority 感知的融合：priority 决定「是否入选」，score 决定「最终顺序」
            if len(results) > top_k:
                results, debug = select_candidates(
                    results,
                    top_k=top_k,
                    rel_delta=self.rel_delta,
                    max_reserved=self.max_reserved,
                    reserve_priority_max=self.reserve_priority_max,
                    priority_caps=self.priority_caps,
                )
                logger.debug(f"priority 融合：{debug}")
            elif results:
                results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
            
            # 置信度标注：阈值不设为「够不到」的死线，改为低分时显式标记
            if results:
                low_conf = results[0]["score"] < self.low_confidence_threshold
                if low_conf:
                    logger.info(
                        f"top1 分数 {results[0]['score']:.4f} < "
                        f"{self.low_confidence_threshold}，标记为低置信度"
                    )
                for r in results:
                    r["low_confidence"] = low_conf
            
            # 剥掉内部字段：_pos 仅供融合阶段定位文档，不属于对外契约。
            for r in results:
                r.pop("_pos", None)
            
            logger.info(f"检索完成，查询：{query[:50]}...，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"检索失败：{e}")
            return []
    
    def search_with_context(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        检索并生成上下文（用于 RAG）
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            max_tokens: 上下文最大 token 数
        
        Returns:
            Tuple[str, List[Dict]]: (上下文文本, 检索结果列表)
        """
        # 检索文档
        results = self.search(query, top_k)
        
        if not results:
            return "", []
        
        # 生成上下文
        max_tokens = max_tokens or self.retrieval_config.get("max_tokens", 2000)
        context_parts = []
        current_tokens = 0
        
        # Token 估算 helper：中文字符约2token，英文字符约0.25token
        def _estimate(text: str) -> int:
            chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            return chinese * 2 + (len(text) - chinese) * 0.25
        
        # 低置信度时在上下文开头显式提示，避免 LLM 把弱相关片段当权威依据
        low_conf = bool(results) and results[0].get("low_confidence", False)
        if low_conf:
            warning = (
                f"【检索置信度低】最佳片段相似度仅 {results[0]['score']:.4f}，"
                f"低于 {self.low_confidence_threshold}，以下内容可能与问题无关，"
                f"请勿据此编造结论。\n\n"
            )
            context_parts.append(warning)
            current_tokens += _estimate(warning)
        
        for result in results:
            content = result["content"]
            metadata = result["metadata"]
            
            estimated_tokens = _estimate(content)
            
            if current_tokens + estimated_tokens > max_tokens:
                break
            
            # 格式化上下文
            source = metadata.get("filename", "未知来源")
            page = metadata.get("page", "未知页码")
            citation = f"【来源：{source} 第 {page} 页】"
            
            context_parts.append(f"{content}\n{citation}")
            current_tokens += estimated_tokens
        
        context = "\n\n---\n\n".join(context_parts)
        
        logger.info(f"上下文生成完成，长度：{len(context)} 字符")
        return context, results
    
    def save(self, directory: Optional[str] = None):
        """
        保存向量库到磁盘
        
        Args:
            directory: 保存目录，默认为配置中的 persist_directory
        """
        directory = directory or self.persist_directory
        os.makedirs(directory, exist_ok=True)
        
        try:
            # 保存 FAISS 索引
            if self.index is not None:
                index_path = os.path.join(directory, "faiss_index.bin")
                
                # 如果是 GPU 索引，先转到 CPU
                if self.use_gpu and faiss.get_num_gpus() > 0:
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    faiss.write_index(cpu_index, index_path)
                else:
                    faiss.write_index(self.index, index_path)
                
                logger.info(f"FAISS 索引已保存：{index_path}")
            
            # 保存文档和元数据
            metadata_path = os.path.join(directory, "documents_metadata.pkl")
            with open(metadata_path, "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "metadata": self.document_metadata,
                    "config": {
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "index_type": self.index_type,
                        "metric": self.metric
                    }
                }, f)
            
            logger.info(f"文档元数据已保存：{metadata_path}")
            
        except Exception as e:
            logger.error(f"向量库保存失败：{e}")
            raise RuntimeError(f"无法保存向量库：{e}")
    
    def load(self, directory: Optional[str] = None) -> bool:
        """
        从磁盘加载向量库
        
        Args:
            directory: 加载目录，默认为配置中的 persist_directory
        
        Returns:
            bool: 是否加载成功
        """
        directory = directory or self.persist_directory
        
        try:
            # 加载 FAISS 索引
            index_path = os.path.join(directory, "faiss_index.bin")
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                logger.info(f"FAISS 索引已加载：{index_path}")
            else:
                logger.warning(f"FAISS 索引文件不存在：{index_path}")
                return False
            
            # 加载文档和元数据
            metadata_path = os.path.join(directory, "documents_metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, "rb") as f:
                    data = pickle.load(f)
                    # 兼容两种序列化格式：
                    # 格式1: List[Dict] (build_index.py直接序列化Document列表)
                    # 格式2: Dict (vectorstore.save()方法)
                    if isinstance(data, list):
                        self.documents = [item.get("content", "") for item in data]
                        self.document_metadata = [item.get("metadata", {}) for item in data]
                    elif isinstance(data, dict):
                        self.documents = data.get("documents", [])
                        self.document_metadata = data.get("metadata", [])
                    else:
                        logger.error(f"不支持的元数据格式：{type(data)}")
                        return False
                
                logger.info(f"文档元数据已加载：{metadata_path}")
            else:
                logger.warning(f"文档元数据文件不存在：{metadata_path}")
                return False
            
            logger.info(f"向量库加载完成，总文档块数：{len(self.documents)}")
            return True
            
        except Exception as e:
            logger.error(f"向量库加载失败：{e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取向量库统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            "total_documents": len(self.documents),
            "index_type": self.index_type,
            "metric": self.metric,
            "embedding_dimension": self.embedder.get_embedding_dimension(),
            "persist_directory": self.persist_directory,
            "has_index": self.index is not None,
            "index_size": self.index.ntotal if self.index else 0
        }
    
    def clear(self):
        """清空向量库"""
        self.index = None
        self.documents.clear()
        self.document_metadata.clear()
        self.embedder.clear_cache()
        logger.info("向量库已清空")
    
    def __repr__(self) -> str:
        """对象表示"""
        return (
            f"FAISSVectorStore("
            f"documents={len(self.documents)}, "
            f"index_type={self.index_type}, "
            f"metric={self.metric})"
        )


# 全局实例（单例模式）
_global_vectorstore = None

def get_vectorstore(config_path: Optional[str] = None) -> FAISSVectorStore:
    """
    获取全局向量数据库实例（单例模式）
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        FAISSVectorStore: 向量数据库实例
    """
    global _global_vectorstore
    if _global_vectorstore is None:
        _global_vectorstore = FAISSVectorStore(config_path)
    return _global_vectorstore


def search_knowledge(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    快速检索知识库（使用全局实例）
    
    Args:
        query: 查询文本
        top_k: 返回结果数量
    
    Returns:
        List[Dict]: 检索结果
    """
    vectorstore = get_vectorstore()
    # 首次调用时索引尚未载入，search() 会因「向量库为空」直接返回空列表。
    # 这里按需懒加载，避免该便捷函数恒返回空。
    if vectorstore.index is None:
        vectorstore.load()
    return vectorstore.search(query, top_k)


if __name__ == "__main__":
    """模块测试"""
    print("=== FAISS 向量数据库测试 ===")
    
    # 测试向量数据库
    vectorstore = FAISSVectorStore()
    print(f"向量数据库：{vectorstore}")
    
    # 测试统计信息
    stats = vectorstore.get_stats()
    print(f"统计信息：{stats}")
    
    print("测试完成！")
