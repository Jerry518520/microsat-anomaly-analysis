"""
生产管线与论文配方一致性测试
=============================
生产管线 `src/integration/anomaly_rag_pipeline.py` 曾把 9 个通道的
contamination 全部写死、融合固定为「强通道 IF+OR / 弱通道段级 baseline」，
与实验代码 `scripts/fusion_v3.py` 选出的 gate_perchannel 配方不一致。
本文件锁定「生产读实验产出的配方」这一约定，防止回退。

不需要模型文件、不调用 LLM，全部是纯逻辑与文件一致性检查。
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FUSION_JSON = ROOT / "data" / "results" / "v3" / "fusion.json"
PIPELINE_PY = ROOT / "src" / "integration" / "anomaly_rag_pipeline.py"


def test_fusion_json_exists():
    """配方来源文件必须存在。缺失时生产管线会回退硬编码，需要能发现。"""
    assert FUSION_JSON.exists(), f"配方文件不存在：{FUSION_JSON}"


def test_recipe_loads_and_has_all_channels():
    """load_recipe() 能读出 9 个通道的算子与 contamination。"""
    from src.integration.anomaly_rag_pipeline import AnomalyRAGPipeline

    recipe = AnomalyRAGPipeline.load_recipe()
    assert recipe is not None, "load_recipe() 返回 None，生产会回退硬编码"

    gate = recipe["gate_choice"]
    assert len(gate) == 9, f"配方应含 9 个通道，实得 {len(gate)}"
    for ch, op in gate.items():
        assert op in {"rule", "if", "AND", "OR"}, f"{ch} 算子非法：{op}"
    assert isinstance(recipe["best_k"], int)


def test_pipeline_init_uses_recipe_not_hardcoded():
    """实例化后 best_contamination 应来自配方，不是内置字典。"""
    from src.integration.anomaly_rag_pipeline import AnomalyRAGPipeline

    p = AnomalyRAGPipeline()
    assert p.recipe is not None, "实例化后未载入配方"
    # 内置回退值对这几个通道与配方不同，断言已被覆盖
    assert p.best_contamination["CADC0872"] != 0.5 or p.recipe["contamination"]["CADC0872"] == 0.5
    # 逐一核对
    for ch, v in p.recipe["contamination"].items():
        expected = v if v is not None else 0.5
        assert p.best_contamination[ch] == pytest.approx(expected), (
            f"{ch} contamination 未用配方值：生产={p.best_contamination[ch]} 配方={expected}"
        )


def test_rule_predicate_uses_best_k_not_ratio():
    """规则层判定须用「违规数 >= k」，不得回退到「违规比例 >= 0.2」。

    两种口径差异很大：18 个特征时比例 0.2 ≈ 违规数 >= 4，
    而 val 选出的 k=2，明显更宽。混用会大幅降低 rule 通道召回。
    """
    src = PIPELINE_PY.read_text(encoding="utf-8")
    # 配方分支必须存在
    assert "n_violated >= k_best" in src, "规则层未使用 best_k"
    # 旧口径只能出现在 else 回退分支里
    assert "n_violated / n_total.replace(0, 1) >= 0.2" in src, "回退分支被删，会导致无配方时崩溃"


def test_gate_fusion_has_all_four_operators():
    """门控融合须实现 rule / if / AND / OR 四种算子。"""
    src = PIPELINE_PY.read_text(encoding="utf-8")
    for op in ['op == "rule"', 'op == "if"', 'op == "AND"', 'op == "OR"']:
        assert op in src, f"门控融合缺算子分支：{op}"


def test_recipe_contamination_matches_fusion_json():
    """生产用的 contamination 必须与 fusion.json 逐通道一致（防手工改错）。"""
    from src.integration.anomaly_rag_pipeline import AnomalyRAGPipeline

    recipe = AnomalyRAGPipeline.load_recipe()
    raw = json.loads(FUSION_JSON.read_text(encoding="utf-8"))["results"]["selection"]
    assert recipe["best_k"] == raw["best_k"]
    for ch, v in raw["best_contamination_per_channel"].items():
        if ch in recipe["contamination"]:
            assert recipe["contamination"][ch] == v


def test_no_anomaly_channel_has_no_positive_in_any_split():
    """被排除的通道必须在 fit/val/test 三个集合上全无正类样本。

    这是排除它的唯一正当理由。若某个「无异常通道」在 fit 上其实有异常样本，
    说明排除依据不成立，属用测试集信息做决策。
    """
    from src.experiments.framework import load_split
    from src.utils.constants import NO_ANOMALY_CHANNELS

    if not NO_ANOMALY_CHANNELS:
        pytest.skip("当前没有排除通道")

    fit, val, test, _ = load_split()
    for ch in NO_ANOMALY_CHANNELS:
        for name, df in (("fit", fit), ("val", val), ("test", test)):
            sub = df[df.channel == ch]
            assert int(sub.anomaly.sum()) == 0, (
                f"{ch} 在 {name} 上有 {int(sub.anomaly.sum())} 个异常段，"
                f"不满足排除条件（排除依据应为全集合无正类）"
            )


def test_hardcoded_params_not_claimed_as_optimal():
    """两个硬编码常量不得被注释或文档说成「最优」。

    `vote_threshold=0.25` 与段级基线 `contamination=0.25` 都**没有 val 选优过程**：
    - fusion_v3.py 中不存在 vote_threshold 这个参数，被 val 选出的是各软融合
      策略的 tau（语义为「归一化分数阈值」），与本值语义不同；
    - select_if_contamination 是分通道搜索，产出的
      best_contamination_per_channel 里没有「全局段级 IF」这一项。

    两者数值同为 0.25 属巧合。论文与注释不得称其为最优。
    """
    src = PIPELINE_PY.read_text(encoding="utf-8")

    # 「最优投票阈值」「最优子采样参数」不得作为结论性描述。
    # 允许出现在纠错说明里（如「原文写『最优投票阈值』不准确」），
    # 判据：该行须带 ⚠ 标记或含否定词。
    for i, line in enumerate(src.splitlines()):
        if "最优投票阈值" in line or "最优子采样参数" in line:
            assert ("⚠" in line) or ("不准确" in line) or ("非" in line), (
                f"第 {i + 1} 行把无选优过程的参数说成最优：{line.strip()[:70]}"
            )

    # 必须有工程设定值的标注
    assert "工程设定值" in src, "缺少「工程设定值」标注"
    # 段级基线 contamination 必须说明无选优过程
    assert "无选优过程的工程设定值" in src, "段级基线 contamination 缺来源说明"
    # 文件头 docstring 不得再宣称 Scheme G + 0.25 是唯一流程
    head = src[: src.index('"""', 3)]
    assert "Scheme G + threshold 0.25" not in head, (
        "docstring 仍把 Scheme G + 0.25 写成唯一流程，未说明配方优先"
    )


def test_production_prediction_does_not_read_labels():
    """生产推理路径不得读取标签来影响预测。

    允许：把真值放进 last_seg_results 供离线评估、API 返回真值做可视化。
    禁止：y_test/y_true 参与 _build_scheme_g 或 is_anomaly 的计算。
    """
    src = PIPELINE_PY.read_text(encoding="utf-8")
    # 死代码已删
    assert 'y_test = dataset_df.loc[~train_mask, "anomaly"].values' not in src, (
        "y_test 死代码仍在，线上代码不应出现标签变量"
    )
    # is_anomaly 的判定只能依赖 anomaly_score 与 threshold
    assert 'seg["is_anomaly"] = seg["anomaly_score"] >= threshold' in src, (
        "is_anomaly 判定口径被改动"
    )
    # 门控融合不得引用任何标签
    gate_src = src[src.index("def _build_scheme_g"):src.index("def _segment_aggregate")]
    for bad in ["y_test", "y_true", '["anomaly"]']:
        assert bad not in gate_src, f"门控融合引用了标签：{bad}"


def test_segment_baseline_is_dead_when_recipe_present():
    """有配方时，9 通道算子不含 seg_baseline，段级基线 IF 结果不应生效。

    若将来给某通道选了 seg_baseline 算子，此测试会失败——那时
    contamination=0.25 才会真正影响结果，必须先给它补上选优过程。
    """
    from src.integration.anomaly_rag_pipeline import AnomalyRAGPipeline

    recipe = AnomalyRAGPipeline.load_recipe()
    if recipe is None:
        pytest.skip("无配方文件")
    ops = set(recipe["gate_choice"].values())
    assert "seg_baseline" not in ops, (
        f"配方含 seg_baseline 算子，段级基线 IF 的 contamination=0.25 将生效，"
        f"而该值无选优过程 —— 须先补选优"
    )

