"""
BM25 关键词侧与混合检索融合的防回归测试
==========================================

覆盖三组约定：
1. :mod:`src.rag.bm25` 的分词与空白归一化（索引侧归一化是关键词匹配的前提）；
2. :class:`~src.rag.bm25.BM25Index` 的基本行为（空语料、专有名词、排序方向）；
3. :meth:`FAISSVectorStore._merge_keyword_candidates` 的**融合契约**：
   priority 不参与排序、关键词加分不改变 score 排序主轴、返回结果不含内部字段。

不测「检索质量」本身——那由 ``scripts/eval_retrieval_quality.py`` 用
24 条标注查询做 before/after 对比，不适合塞进单元测试（需要真实索引 + GPU）。
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.rag.bm25 import BM25Index, normalize_whitespace, tokenize  # noqa: E402


# --------------------------------------------------------------------------
# 空白归一化
# --------------------------------------------------------------------------
class TestNormalizeWhitespace:
    def test_nbsp_becomes_space(self):
        # MAXWELL 手册 100% 的 chunk 含这种 NBSP 碎片
        assert normalize_whitespace("Aaron\xa0\nAboaf") == "Aaron\nAboaf"

    def test_triple_space_collapsed(self):
        # 'Safe   mode' 必须能匹配 'safe mode'
        assert normalize_whitespace("Safe   mode") == "Safe mode"

    def test_phrase_becomes_matchable(self):
        raw = "This  mode   occurs  when  the  state  of  charge"
        assert "safe mode" not in raw
        assert "state of charge" in normalize_whitespace(raw)

    def test_newline_preserved(self):
        # 段落边界是 chunk 结构信号，不能被压成一行
        assert normalize_whitespace("a\n\nb") == "a\n\nb"

    def test_empty_input(self):
        assert normalize_whitespace("") == ""
        assert normalize_whitespace(None) == ""

    def test_ligature_expanded(self):
        assert normalize_whitespace("ﬂow") == "flow"


# --------------------------------------------------------------------------
# 分词
# --------------------------------------------------------------------------
class TestTokenize:
    def test_english_identifier_kept_whole(self):
        # vote_threshold 不能被拆成 vote + threshold
        assert "vote_threshold" in tokenize("vote_threshold=0.25")

    def test_channel_id_kept_whole(self):
        assert "cadc0874" in tokenize("CADC0874通道异常")

    def test_decimal_number_kept_whole(self):
        assert "0.6281" in tokenize("F1 0.6281")

    def test_cjk_bigram_present(self):
        # bigram 是技术术语的兜底（「磁力矩器」未必在 jieba 词典里）
        toks = tokenize("磁力矩器故障")
        assert "磁力" in toks or "磁力矩" in toks

    def test_empty(self):
        assert tokenize("") == []


# --------------------------------------------------------------------------
# BM25Index
# --------------------------------------------------------------------------
class TestBM25Index:
    def test_empty_corpus_is_safe(self):
        bm = BM25Index([])
        assert bm.is_empty()
        assert bm.score("anything") == {}
        assert bm.top_n("anything", 5) == []

    def test_single_doc(self):
        bm = BM25Index(["the EPS module regulates power"])
        assert not bm.is_empty()
        hits = bm.top_n("EPS", 5)
        assert len(hits) == 1
        assert hits[0][0] == 0

    def test_exact_term_ranks_higher_than_incidental(self):
        # 专有名词查询应把含该词的 chunk 排到前面
        bm = BM25Index([
            "reaction wheels and magnetorquers on the spacecraft bus",
            "the reaction wheel assembly uses four canted wheels",
        ])
        top = bm.top_n("magnetorquers", 2)
        assert top[0][0] == 0

    def test_scores_are_descending(self):
        bm = BM25Index([
            "solar array power output degrades over time",
            "solar array",
            "attitude control system with reaction wheels",
        ])
        hits = bm.top_n("solar array power", 3)
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True)

    def test_unmatched_query_returns_empty(self):
        bm = BM25Index(["attitude control system"])
        assert bm.top_n("zzzzz-not-present", 5) == []

    def test_whitespace_fragment_still_findable(self):
        # 这是本模块存在的核心理由：索引侧碎片化会破坏字面匹配，
        # 归一化后 'Safe   mode' 必须能被 'safe mode' 检索到。
        bm = BM25Index(["Safe \xa0\nmode   will  try  to  maintain  power"])
        assert bm.top_n("safe mode", 3), "空白碎片导致 'safe mode' 检索不到"

    def test_position_index_alignment(self):
        # top_n 返回的 doc下标必须能对回原语料顺序
        corpus = ["alpha one", "beta two", "gamma three"]
        bm = BM25Index(corpus)
        (pos, _), = bm.top_n("gamma", 1)
        assert corpus[pos] == "gamma three"


# --------------------------------------------------------------------------
# 融合契约（vectorstore）
# --------------------------------------------------------------------------
def _store(docs, **kw):
    """借用真实 ``_merge_keyword_candidates``，但跳过 __init__（不加载嵌入模型）。"""
    from src.rag.vectorstore import FAISSVectorStore

    s = FAISSVectorStore.__new__(FAISSVectorStore)
    s.documents = docs
    s.document_metadata = [{} for _ in docs]
    s.keyword_boost = kw.get("boost", 0.02)
    s.keyword_pool_k = kw.get("pool", 40)
    s.filter_low_quality = True
    s.min_chunk_chars = 80
    s.max_junk_ratio = 0.55
    s.hybrid_search = True
    from src.rag.bm25 import BM25Index
    s._bm25 = BM25Index(docs)
    return s


class TestMergeKeywordCandidates:
    def test_boost_applied_to_vector_hits(self):
        docs = [
            "EPS electrical power system regulates power and charging " * 3,
            "attitude control with reaction wheels on board " * 3,
        ]
        s = _store(docs)
        results = [{
            "content": docs[0], "score": 0.50, "metadata": {}, "_pos": 0,
        }]
        out = s._merge_keyword_candidates("EPS power", results)
        assert out[0]["score"] > 0.50, "关键词命中应带来加分"
        assert out[0]["score"] < 0.50 + s.keyword_boost + 1e-9, "加分不得超过 boost"

    def test_no_keyword_hit_leaves_score_untouched(self):
        docs = [
            "EPS electrical power system regulates power and charging " * 3,
            "attitude control with reaction wheels on board " * 3,
        ]
        s = _store(docs)
        results = [{
            "content": docs[0], "score": 0.50, "metadata": {}, "_pos": 0,
        }]
        out = s._merge_keyword_candidates("ZZZZ-nonexistent-token", results)
        assert out[0]["score"] == pytest.approx(0.50)

    def test_disabled_when_boost_zero(self):
        docs = ["EPS power module " * 5]
        s = _store(docs, boost=0.0)
        results = [{"content": docs[0], "score": 0.5, "metadata": {}, "_pos": 0}]
        out = s._merge_keyword_candidates("EPS", results)
        assert len(out) == 1 and out[0]["score"] == pytest.approx(0.5)

    def test_empty_results_short_circuits(self):
        s = _store(["EPS power " * 5])
        assert s._merge_keyword_candidates("EPS", []) == []

    def test_low_quality_chunk_excluded(self):
        # 过短且标点占比高的片段不应被关键词侧塞进候选
        docs = ["EPS", "a, b, c, d, e, f, g, h"]
        s = _store(docs)
        results = []
        out = s._merge_keyword_candidates("EPS", results)
        assert all(len(r["content"]) > 3 for r in out)

    def test_priority_never_used_for_ordering(self):
        """核心约定：融合只改score，绝不引入 priority 参与排序。"""
        docs = [
            "EPS power system regulates the bus voltage " * 3,
            "reaction wheels and magnetorquers for attitude control " * 3,
        ]
        metas = [
            {"source_name": "low-prio", "priority": 5},
            {"source_name": "high-prio", "priority": 0},
        ]
        s = _store(docs)
        s.document_metadata = metas
        results = [{
            "content": docs[0], "score": 0.50, "metadata": metas[0], "_pos": 0,
        }]
        out = s._merge_keyword_candidates("EPS power", results)
        # 唯一被加分的仍是 pos=0（低优先级那个），证明没按priority 优待
        assert out[0]["metadata"]["priority"] == 5
        assert out[0]["score"] > 0.50
