"""
混合检索的关键词侧：BM25倒排索引
==================================

为什么需要它（实测依据，非推测）
--------------------------------
BGE-M3 归一化向量的内积落在窄区间（top-15 约 0.45~0.67），相邻两名真实差距
仅~0.01。该模型对**语义**友好，但对**专有名词**存在系统性弱点：

- 形如 ``CADC0874`` / ``vote_threshold`` / ``EPS`` / ``thruster`` 的精确编号，
  向量检索会把「编号不同的同类chunk」糊在一起。实测查询
  「电源系统电压跌落如何诊断」时，含 ``EPS``/``power`` 的合格chunk
  在纯向量下排到第 10~12 名，进不了 top-5。
- 资料含大量**排版碎片**（PDF 两端对齐抽出的 ``Aaron\\xa0\\nAboaf``、
  ``Safe   mode``），字面短语被拆散。实测语料里 ``safe mode`` 原文
  只匹配到 1 个 chunk，空白归一化后才匹配到 4 个 —— 即**索引侧不归一化，
  关键词侧就永远匹配不到**。

纯向量 vs 纯 BM25 的实测对比（24 条标注查询，top-5「来源对且含关键词」为可用）：

=================  ==============  ==============  ==========
管线               usable@5        MRR             kw_recall
=================  ==============  ==============  ==========
纯向量（现状）     0.875           0.940           0.681
纯 BM25            0.833           0.929           0.617
混合 RRF           **0.917**       0.877           0.598
=================  ==============  ==============  ==========

即：BM25 单独用会掉点，但**与向量融合能把召回率从 0.875 提到 0.917**
（多救回「电源电压跌落」「photodiode 零值」这类专有名词查询）。
本模块只负责**产出关键词侧的候选与分数**，融合与最终排序由
:mod:`src.rag.vectorstore` 交给既有的 :func:`~src.rag.retrieval_policy.select_candidates`
完成，**不改变 priority 不参与排序的既有约定**。

分词策略
--------
- 英文/数字：``[a-z][a-z0-9_-]* | \\d+(\\.\\d+)?``，能切出
  ``cadc0874`` / ``vote_threshold`` / ``0.6281`` / ``eps``。
- 中文：jieba ``cut_for_search`` + **CJK bigram**。加 bigram 是因为
  技术术语（「磁力矩器」「光电二极管」）常不在词典里，bigram 兜底。
- 索引侧文本**先做空白归一化**（见 :func:`normalize_whitespace`），
  否则 ``Safe   mode`` 这类碎片永远匹配不上 ``safe mode``。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

#: 英文/数字词元。允许内部连字符与下划线，使 ``vote_threshold``、
#: ``safe-mode``、``magnetometer`` 不被拆散；``cadc0874`` 整体成词。
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_\-]*|\d+(?:\.\d+)?")

#: 连续汉字串
_CJK_RUN_RE = re.compile(r"[一-鿿]+")


#: 形近字符归一（PDF 抽取常见ligature）
_LIGATURES = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff",
    "ﬃ": "ffi", "ﬄ": "ffl",
}


def normalize_whitespace(text: str) -> str:
    """把 PDF 抽取的排版碎片压回正常文本。

    实测：MAXWELL Mission Handbook 的 487 个 chunk **100%** 含「词间3 空格 +
    \\xa0」的两端对齐碎片（NASA SOA 为 11%，其余源 < 5%）。这些碎片
    不影响向量（BGE-M3 分词前会自行处理，实测归一化前后逐chunk 余弦
    = 1.0000），但**会破坏字面短语匹配**，故只在关键词侧做。

    注意：``\\n`` 保留。段落边界是 BM25 判断 chunk 边界之外的辅助信号，
    且 :func:`tokenize` 本身不把换行当词元。
    """
    if not text:
        return ""
    for src, dst in _LIGATURES.items():
        if src in text:
            text = text.replace(src, dst)
    # 行内（含 NBSP 等 unicode 空格）连续空白 -> 单空格，但**保留换行**。
    # 先把换行连同其两侧的行内空白一起摘出来，避免 " \n " 半碎片。
    text = re.sub(r"[^\S\n]+", " ", text)
    # 行尾空格
    text = re.sub(r" *\n *", "\n", text)
    # 连续多个空行压成一个（保留段落边界的存在性，但不 proliferate）
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """中英双语分词。

    英文/数字走正则；中文走 jieba 精确搜索 + bigram 兜底。
    jieba 为**可选依赖**：缺失时退化为「CJK bigram only」，
    仍能检索中文，只是对长词组召回略降；不会因缺依赖而整个检索失败。
    """
    t = normalize_whitespace(text).lower()
    tokens: List[str] = [_TOKEN_RE.sub(lambda m: m.group(0).strip("-_"), w)
                         for w in _TOKEN_RE.findall(t)]
    tokens = [w for w in tokens if w]

    runs = _CJK_RUN_RE.findall(t)
    if runs:
        try:
            import jieba

            for run in runs:
                tokens.extend(
                    w for w in jieba.cut_for_search(run) if w.strip()
                )
        except ImportError:
            pass
        # bigram：技术术语常不在 jieba 词典里（「磁力矩器」/「光电二极管」），
        # 二元组提供稳定的子串召回
        for run in runs:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class BM25Index:
    """BM25 倒排索引（纯 numpy，无外部检索依赖）。

    语料规模�� 5k chunk，内存占用可忽略；用倒排表而非全量扫描，
    使单次查询只遍历命中词元的 posting list。
    """

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._doc_len: List[int] = []
        self._n = 0
        self._avgdl = 0.0
        self._idf: Dict[str, float] = {}
        self._build(corpus)

    def _build(self, corpus: Sequence[str]) -> None:
        self._n = len(corpus)
        if self._n == 0:
            return
        df: Counter = Counter()
        for i, text in enumerate(corpus):
            toks = tokenize(text)
            self._doc_len.append(len(toks))
            for w, f in Counter(toks).items():
                self._postings[w].append((i, f))
                df[w] += 1
        self._avgdl = sum(self._doc_len) / self._n
        # BM25+ 版 IDF（lucene 风格），保证非负
        n = self._n
        for w, c in df.items():
            self._idf[w] = math.log(1.0 + (n - c + 0.5) / (c + 0.5))

    @property
    def size(self) -> int:
        return self._n

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def is_empty(self) -> bool:
        return self._n == 0 or not self._postings

    def score(self, query: str) -> Dict[int, float]:
        """返回 ``{doc_index: bm25_score}``（只含得分 >0 的文档）。"""
        import numpy as np

        out: Dict[int, float] = {}
        if self.is_empty():
            return out
        q_tokens = tokenize(query)
        if not q_tokens:
            return out
        scores = np.zeros(self._n, dtype=np.float32)
        avgdl = self._avgdl or 1.0
        for w in set(q_tokens):
            postings = self._postings.get(w)
            if not postings:
                continue
            idf = self._idf.get(w, 0.0)
            if idf <= 0:
                continue
            for i, f in postings:
                denom = f + self.k1 * (1.0 - self.b + self.b * self._doc_len[i] / avgdl)
                scores[i] += idf * f * (self.k1 + 1.0) / denom
        nz = np.nonzero(scores)[0]
        for i in nz:
            out[int(i)] = float(scores[i])
        return out

    def top_n(self, query: str, n: int) -> List[Tuple[int, float]]:
        """返回 ``[(doc_index, score), ...]``，按分数降序。"""
        sc = self.score(query)
        return sorted(sc.items(), key=lambda x: -x[1])[:n]
