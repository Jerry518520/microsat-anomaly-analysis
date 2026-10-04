"""
RAG 检索策略回归测试

覆盖 2026-10 上线验收发现的检索质量问题：
1. 低质量片段（PDF 残片/ 目录页）被识别并过滤
2. priority 参与「入选」但不参与「排序」，排序仍严格按 score 降序
3. priority 保留席位受限，不会把对口知识顶掉（防止 9chunk 的实验库反客为主）
4. score_threshold 不被调成检索不到东西的死线，低分结果标记 low_confidence

这些测试全部使用构造数据，不需要加载 FAISS 索引或嵌入模型。
"""

import os
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.retrieval_policy import (  # noqa: E402
    default_priority_caps,
    effective_length,
    is_low_quality,
    junk_ratio,
    select_candidates,
)


# ============================================================
# 低质量片段识别
# ============================================================

def test_orphan_sentence_is_low_quality():
    """实测占据 top1 的孤立句子必须被判为噪声"""
    junk = "returned due to non-completion of coordination."
    assert is_low_quality(junk) is True


def test_too_short_chunk_is_low_quality():
    """有效字符数低于下限 -> 噪声"""
    assert is_low_quality("OK") is True
    assert is_low_quality("   \n  ") is True


def test_table_of_contents_page_is_low_quality():
    """目录页：字符几乎全是标点/空白/页码"""
    toc = "1.2 Thermal Control ....................................... 45\n" \
          "2.1  Introduction .......................................... 7\n"
    assert junk_ratio(toc) > 0.55
    assert is_low_quality(toc) is True


def test_meaningful_chunk_is_not_low_quality():
    """真实技术内容不能被误杀"""
    good = (
        "OPS-SAT 姿态确定与控制系统包含陀螺仪、加速度计、三轴磁力计与反作用飞轮。"
        "磁力计异常通常源于磁干扰或磁力矩器故障，会直接影响姿态解算精度。"
    )
    assert is_low_quality(good) is False


def test_min_chars_boundary_respected():
    """刚好达到 min_chars 的内容不应因长度被过滤"""
    text = "磁力计异常通常源于磁干扰或磁力矩器故障"  # 19 个汉字 -> effective_length=38
    assert effective_length(text) == 38
    assert is_low_quality(text, min_chars=38) is False
    assert is_low_quality(text, min_chars=39) is True


def test_dense_chinese_survives_min_chars():
    """中文信息密度高：72 字的中文技术内容不应被 min_chars=80 误杀。
    原始字符数 72 < 80，但 effective_length=131 >= 80。
    """
    text = (
        "OPS-SAT 姿态确定与控制系统包含陀螺仪、加速度计、三轴磁力计与反作用飞轮。"
        "磁力计异常通常源于磁干扰或磁力矩器故障，会直接影响姿态解算精度。"
    )
    assert len(text) < 80, "前提：该内容原始字符数确实低于 80"
    assert is_low_quality(text, min_chars=80) is False


def test_english_junk_still_filtered_under_same_threshold():
    """中英文用同一阈值：英文残片仍应被过滤"""
    assert is_low_quality("2 plus Irradiation and Gyro", min_chars=80) is True


# ============================================================
# priority 融合：入选 vs 排序
# ============================================================

def _cand(idx, score, source, priority):
    return {
        "content": f"chunk-{idx}",
        "score": score,
        "metadata": {"source_name": source, "priority": priority},
    }


def test_priority_does_not_break_score_ordering():
    """核心约束：priority 决定入选，但最终顺序必须严格按 score 降序。

    低优先源拿到更高 score 时仍应排在前面 —— 否则 priority 就变成了
    硬排序，会破坏向量相似度语义。
    """
    candidates = [
        _cand(0, 0.90, "LowPriorityManual", 5),
        _cand(1, 0.60, "ExperimentKB", 0),
        _cand(2, 0.80, "ExperimentKB", 0),
    ]
    selected, _ = select_candidates(candidates, top_k=2)
    assert [c["score"] for c in selected] == sorted(
        [c["score"] for c in selected], reverse=True
    )
    # 低分但高 priority 的 chunk 即使入选，也不能排到高分chunk 前面
    assert selected[0]["score"] >= selected[-1]["score"]


def test_high_priority_source_gets_reserved_slot():
    """高优先源分数略低但仍在竞争区内时，应获得保留席位"""
    candidates = [
        _cand(i, 0.90 - i * 0.001, "HugeManual", 5) for i in range(20)
    ]
    candidates.append(_cand(99, 0.885, "ExperimentKB", 0))
    selected, debug = select_candidates(
        candidates, top_k=5, rel_delta=0.12
    )
    assert any(c["metadata"]["source_name"] == "ExperimentKB" for c in selected)
    assert debug["reserved_used"] >= 1


def test_priority_cannot_rescue_out_of_zone_candidate():
    """分数明显掉队的候选，即便 priority 最高也不该靠保留席位进来。
    这是防止「9chunk 实验库反客为主」的关键闸门。

    注意：候选池极小时可能因「填不满 top_k」而在最后兜底阶段被带上，
    因此这里构造充足的竞争区候选，确保它只能落选而不会挤掉别人。
    """
    candidates = [
        _cand(i, 0.90 - i * 0.001, f"Manual{i%5}", 4 + (i % 2)) for i in range(30)
    ]
    # priority=0 但分数比 top 低 0.30，远超 rel_delta=0.12
    candidates.append(_cand(99, 0.60, "ExperimentKB", 0))
    selected, debug = select_candidates(
        candidates, top_k=5, rel_delta=0.12
    )
    assert debug["n_eligible"] == 30, "竞争区应只含分数靠前的候选"
    assert not any(
        c["metadata"]["source_name"] == "ExperimentKB" for c in selected
    ), "分数掉队的实验库 chunk 不应进入结果"
    assert debug["rel_floor"] > 0.60


def test_out_of_zone_only_enters_when_pool_too_small():
    """候选池不足时允许兜底带入落选候选，但绝不减少返回条数。

    每源上限是「偏好」而非硬限制——否则候选源不足时结果会少于 top_k，
    这比多一条低分结果更糟。
    """
    candidates = [
        _cand(i, 0.90 - i * 0.001, "HugeManual", 5) for i in range(20)
    ]
    candidates.append(_cand(99, 0.60, "ExperimentKB", 0))
    selected, debug = select_candidates(
        candidates, top_k=5, rel_delta=0.12
    )
    assert len(selected) == 5, "兜底阶段应保证填满 top_k"
    assert debug["reserved_used"] == 0, \
        "落选候选不得占用保留席位（保留席位只在竞争区内分配）"


def test_reserved_slots_are_capped():
    """保留席位不超过 top_k 的一半，priority 不得主导整个结果集"""
    candidates = []
    for src, prio in [("A", 0), ("B", 1), ("C", 2), ("D", 0), ("E", 1)]:
        candidates.append(_cand(len(candidates), 0.90 - len(candidates) * 0.001,
                                src, prio))
    for i in range(30):
        candidates.append(_cand(100 + i, 0.70, "Filler", 5))
    selected, debug = select_candidates(candidates, top_k=4)
    assert debug["max_reserved"] == 2
    assert debug["reserved_used"] <= 2


def test_per_source_cap_limits_dominant_source():
    """候选源充足时（多源竞争），大体量源不能独占 top_k（priority 5 限1 条）"""
    candidates = [
        _cand(i, 0.90 - i * 0.001, "DominantManual", 5) for i in range(20)
    ]
    # 另有 4 个来源各1 条高分候选，保证放宽上限也能填满 top_k
    for j, prio in enumerate([1, 2, 3, 4]):
        candidates.append(_cand(80 + j, 0.87 - j * 0.001, f"Other{j}", prio))
    for i in range(5):
        candidates.append(_cand(50 + i, 0.86 - i * 0.001, "ExperimentKB", 0))
    selected, debug = select_candidates(candidates, top_k=5)
    assert debug["n_after_strict"] == 5, "源充足时不应需要放宽每源上限"
    n_dom = sum(1 for c in selected
                if c["metadata"]["source_name"] == "DominantManual")
    assert n_dom == 1, "priority=5 的源每源上限应为 1"


def test_high_priority_source_may_take_multiple_slots():
    """priority=0 的对口源允许占多条（上限 3），避免被单源挤掉"""
    candidates = [
        _cand(i, 0.90 - i * 0.001, "DominantManual", 4) for i in range(20)
    ]
    for i in range(5):
        candidates.append(_cand(50 + i, 0.86 - i * 0.001, "ExperimentKB", 0))
    selected, _ = select_candidates(candidates, top_k=5)
    n_exp = sum(1 for c in selected
                if c["metadata"]["source_name"] == "ExperimentKB")
    assert n_exp >= 2, "priority=0 的源应能占据多条席位"


def test_top_k_respected_and_results_sorted():
    candidates = [
        _cand(i, 0.90 - i * 0.001, f"Src{i%4}", i % 6) for i in range(40)
    ]
    selected, _ = select_candidates(candidates, top_k=5)
    assert len(selected) <= 5
    scores = [c["score"] for c in selected]
    assert scores == sorted(scores, reverse=True)


def test_empty_candidates_returns_empty():
    selected, debug = select_candidates([], top_k=5)
    assert selected == []
    assert debug["n_candidates"] == 0


def test_single_candidate_survives():
    selected, _ = select_candidates([_cand(0, 0.5, "X", 0)], top_k=5)
    assert len(selected) == 1


# ============================================================
# 配置：阈值不能调成检索不到东西的死线
# ============================================================

def test_score_threshold_in_config_is_reachable():
    """configs/rag_config.yaml 的 score_threshold 必须 <= 实测最低 top1。

    BGE-M3 在本库实测 top-15 落在 0.45~0.67；若阈值设成 0.7，检索恒返回空。
    """
    import yaml
    cfg = yaml.safe_load(
        (project_root / "configs" / "rag_config.yaml").read_text(encoding="utf-8")
    )
    retrieval = cfg["retrieval"]
    assert retrieval["score_threshold"] <= 0.3, \
        "score_threshold 超过 0.3 会让实测 top1(0.45~0.67) 的检索返回空"
    # 低置信度阈值必须高于 score_threshold，否则标记形同虚设
    assert retrieval["low_confidence_threshold"] > retrieval["score_threshold"]


def test_pipeline_default_threshold_matches_vectorstore():
    """pipeline.py 与 vectorstore.py 的缺省阈值必须一致（都曾是 0.7 的死线）"""
    pipeline_src = (project_root / "src" / "rag" / "pipeline.py").read_text(
        encoding="utf-8"
    )
    vs_src = (project_root / "src" / "rag" / "vectorstore.py").read_text(
        encoding="utf-8"
    )
    assert 'score_threshold", 0.7' not in pipeline_src, \
        "pipeline.py 的 score_threshold 缺省值仍为 0.7（死线）"
    assert 'score_threshold", 0.7' not in vs_src, \
        "vectorstore.py 的 score_threshold 缺省值仍为 0.7（死线）"


def test_default_priority_caps_shape():
    caps = default_priority_caps()
    assert caps[0] >= caps[5], "priority 越小的源允许席位不应更少"
    assert caps[5] == 1, "最低优先源应限制为 1 条"
