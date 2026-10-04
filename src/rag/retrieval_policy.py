"""
RAG 检索策略
=============

集中定义「低质量 chunk 判定」与「priority 感知的候选筛选」规则，
供 :mod:`scripts.build_index`（建库时打标/剔除）与
:mod:`src.rag.vectorstore`（检索时过滤/融合）共用，避免两处规则漂移。

设计要点（为何 priority 不做简单加权）
--------------------------------------
BGE-M3 归一化向量的内积落在窄区间（实测 top-15 约0.45~0.67），
若用 ``score + w * priority``直接加权，priority 的数值尺度会淹没
score 的有效差异（相邻两名真实差距仅 ~0.01），等于把语义排序改成
按文档来源硬排序。

因此这里采用 **「priority 决定入选资格，score 决定最终顺序」** 的两段式：

1. *保留席位*：仅当候选的分数落在 ``top_score - rel_delta`` 的相对竞争区内，
   才允许高 priority 源占用有限席位（``max_reserved``，默认不超过 top_k 的一半）。
   绝对分数低但相对竞争力强的内容仍可入选；
   分数明显掉队的内容即便 priority 高也进不来。
2. *每源上限*：priority 越高（如p=0 的项目实验结论）允许同源占更多席位，
   防止NASA SOA / ITU 这类大体量源把top_k 全部占满。
3. *最终排序*：严格按 score 降序，priority 不参与排序，
   以免提示词中的第一条片段与其真实相关度不符。
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

# 判定「无信息量片段」时忽略的字符（标点/空白/常见排版符号）
_PUNCT_CHARS = frozenset(
    " .,:;-–—_/\\()[]{}<>\"'`~@#$%^&*+=|\t\n"
)

# 文本被切断后残留的常见页眉页脚碎片（全部小写且不含中文/数字实体时才判为噪声）
_HEADER_ONLY_HINTS = (
    "returned due to non-completion",
)


def junk_ratio(text: str) -> float:
    """标点与空白字符占比，用于识别目录页/表格残片。"""
    if not text:
        return 1.0
    return sum(1 for c in text if c in _PUNCT_CHARS) / len(text)


def effective_length(text: str) -> int:
    """
    信息量长度：汉字按 2 计，其余字符按 1 计。

    中文字符的信息密度约为英文的两倍，若按原始字符数统一设下限，
    会误杀大量有实质内容的中文短chunk（实测 ITU CN 中有意义的段落
    常在 60~80 字之间）。加权后可对中英文用同一阈值。
    """
    if not text:
        return 0
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return cjk * 2 + (len(text) - cjk)


def is_low_quality(
    text: str,
    min_chars: int = 80,
    max_junk_ratio: float = 0.55,
) -> bool:
    """
    判断chunk 是否为低质量噪声片段。

    三类判定：
    1. 有效信息量过少（PDF 换页截断产生的孤立句子/表格单元格）；
    2. 标点空白占比过高（目录页、图表索引）；
    3. 命中已知的页眉/页脚残片。

    ``min_chars`` 比较的是 :func:`effective_length`（汉字按 2 计），
    因此中英文可用同一阈值。
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    if effective_length(stripped) < min_chars:
        return True
    if junk_ratio(stripped) > max_junk_ratio:
        return True
    low = stripped.lower()
    return any(h in low for h in _HEADER_ONLY_HINTS)


def default_priority_caps(max_priority: int = 5) -> Dict[int, int]:
    """
    priority -> 同源入选上限。

    priority 数字越小越优先（0 为项目自身实验结论）。
    高优先源允许占更多席位，保证对口知识不会被大体量通用手册淹没。
    """
    caps = {0: 3, 1: 3, 2: 2, 3: 2}
    for p in range(4, max_priority + 1):
        caps[p] = 1
    return caps


def select_candidates(
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    rel_delta: float = 0.12,
    max_reserved: Optional[int] = None,
    reserve_priority_max: int = 2,
    priority_caps: Optional[Dict[int, int]] = None,
    source_key: str = "source_name",
    priority_key: str = "priority",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    priority 感知的候选筛选（两段式：席位保留 -> 分数补齐 -> 分数排序）。

    Args:
        candidates: 候选字典列表，需含``score`` / metadata 中的
            ``source_key`` 与 ``priority_key``。
        top_k: 最终返回条数。
        rel_delta: 相对竞争区宽度。分数低于 ``top_score - rel_delta``
            的候选不参与保留席位（相对掉队者靠priority 也进不来）。
        max_reserved: 保留席位上限，默认 ``max(1, top_k // 2)``——
            保证 priority 最多影响一半结果，不主导整个结果集。
        reserve_priority_max: 仅priority <= 该值的源可占用保留席位。
        priority_caps: priority -> 同源上限，缺省见 :func:`default_priority_caps`。
        source_key / priority_key: metadata 中的字段名。

    Returns:
        (选中列表, 调试信息)。选中列表按 score 降序。
    """
    from collections import Counter, defaultdict

    caps = priority_caps or default_priority_caps()
    if max_reserved is None:
        max_reserved = max(1, top_k // 2)
    debug: Dict[str, Any] = {
        "n_candidates": len(candidates),
        "max_reserved": max_reserved,
        "reserve_priority_max": reserve_priority_max,
    }
    if not candidates:
        debug["top_score"] = None
        debug["rel_floor"] = None
        debug["n_eligible"] = 0
        debug["reserved_used"] = 0
        return [], debug

    def _src(c: Dict[str, Any]) -> str:
        return c.get("metadata", {}).get(source_key, "unknown")

    def _prio(c: Dict[str, Any]) -> int:
        val = c.get("metadata", {}).get(priority_key)
        return val if isinstance(val, int) else 99

    top_score = max(c["score"] for c in candidates)
    rel_floor = top_score - rel_delta
    eligible = [c for c in candidates if c["score"] >= rel_floor]
    debug.update({
        "top_score": top_score,
        "rel_floor": rel_floor,
        "n_eligible": len(eligible),
    })

    # 阶段 1：按 (priority 升序, 该源最高分降序) 分配保留席位
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in eligible:
        by_source[_src(c)].append(c)

    selected: List[Dict[str, Any]] = []
    chosen_idx = set()
    reserved_used = 0
    for src in sorted(
        by_source,
        key=lambda s: (min(_prio(c) for c in by_source[s]),
                       -max(c["score"] for c in by_source[s])),
    ):
        if reserved_used >= max_reserved:
            break
        pool = by_source[src]
        if min(_prio(c) for c in pool) > reserve_priority_max:
            continue
        best = max(pool, key=lambda c: c["score"])
        selected.append(best)
        chosen_idx.add(id(best))
        reserved_used += 1
    debug["reserved_used"] = reserved_used

    # 阶段 2：按 score 补齐。
    # 分级放宽，保证「补不满top_k」和「分数掉队候选混入」不会同时发生：
    #   a) 竞争区内 + 遵守每源上限   —— 正常路径，兼顾多样性与相关性
    #   b) 竞争区内 + 放宽每源上限   —— 候选源太少时仍能填满 top_k
    #   c) 竞争区外 + 遵守每源上限   —— 仅当a/b 仍填不满才启用
    #   d) 竞争区外 + 放宽每源上限   —— 最后兜底，保证不返回少于 top_k 条
    def _fill(pool: List[Dict[str, Any]], enforce_caps: bool) -> None:
        src_count = Counter(_src(c) for c in selected)
        for c in sorted(pool, key=lambda x: -x["score"]):
            if len(selected) >= top_k:
                return
            if id(c) in chosen_idx:
                continue
            if enforce_caps and src_count[_src(c)] >= caps.get(_prio(c), 1):
                continue
            selected.append(c)
            chosen_idx.add(id(c))
            src_count[_src(c)] += 1

    outside = [c for c in candidates if c["score"] < rel_floor]
    _fill(eligible, enforce_caps=True)
    debug["n_after_strict"] = len(selected)
    _fill(eligible, enforce_caps=False)
    debug["n_after_zone_relaxed"] = len(selected)
    if len(selected) < top_k:
        _fill(outside, enforce_caps=True)
    debug["n_after_outside"] = len(selected)
    _fill(outside, enforce_caps=False)
    debug["n_after_full_relax"] = len(selected)

    # 阶段 3：严格按 score 排序（priority 不参与排序）
    selected = selected[:top_k]
    selected.sort(key=lambda c: -c["score"])
    return selected, debug
