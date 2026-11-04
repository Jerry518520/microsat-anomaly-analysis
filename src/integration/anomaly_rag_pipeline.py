"""Frozen A02 engineering detection plus optional RAG explanation.

The engineering candidate uses canonical A01 features and only the fit1275
population. Historical fusion parameters are transferred as fixed engineering
choices. This is not a reproduction of historical test scores or a new-feature
validation winner. Missing or incompatible packages fail without training.
Formal labels equal the selected gate output; display scores never relabel.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.data_loader import load_segments, load_config, get_project_root
from src.utils.constants import (
    CHANNEL_PHYSICS,
    FEATURE_SUMMARY_KEYS, TYPE_THRESHOLDS, ANOMALY_TYPE_LABELS,
)
from src.rag.pipeline import RAGPipeline, get_rag_pipeline


@dataclass
class AnomalyResult:
    """单条异常检测结果"""
    segment: int | str
    channel: str
    anomaly_score: float  # 段级异常率 (0-1)
    anomaly_type: str  # intermittent_spikes/high_freq_jitter/low_activity/unusual_shapes
    feature_summary: Dict[str, float]  # 关键特征统计
    is_anomaly: bool = True
    rule_hits: int = 0
    rule_pred: Optional[bool] = None
    if_pred: Optional[bool] = None
    if_score: Optional[float] = None
    gate_op: str = ""
    gate_pred: bool = True
    final_pred: bool = True
    postprocess: str = "identity"
    display_score: float = 0.
    detector_version: str = ""
    feature_version: str = ""


@dataclass  
class RAGExplanation:
    """RAG解释结果"""
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_time: float
    generation_time: float


@dataclass
class CombinedResult:
    """联合结果：检测 + 解释"""
    segment: int | str
    channel: str
    anomaly_score: float
    anomaly_type: str
    feature_summary: Dict[str, float]
    rag_explanation: str
    rag_sources: List[Dict[str, Any]]
    retrieval_time: float
    generation_time: float
    rule_hits: int = 0
    rule_pred: Optional[bool] = None
    if_pred: Optional[bool] = None
    if_score: Optional[float] = None
    gate_op: str = ""
    gate_pred: bool = True
    final_pred: bool = True
    postprocess: str = "identity"
    display_score: float = 0.
    detector_version: str = ""
    feature_version: str = ""


class AnomalyRAGPipeline:
    """One frozen engineering detector; RAG is initialized separately."""
    STRONG_CHANNELS = {"CADC0872", "CADC0873", "CADC0874"}

    @classmethod
    def load_recipe(cls):
        from src.detection.frozen import load_recipe
        return load_recipe()

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.config_path = config_path
        self.rag_pipeline = None
        self.recipe = self.load_recipe()
        self.best_contamination = self.recipe["contamination"].copy()
        self.psi = self.recipe["isolation_forest"]["max_samples"]
        self.last_seg_results = pd.DataFrame()

    def _init_rag(self):
        """延迟初始化RAG Pipeline（避免启动时加载大模型）"""
        if self.rag_pipeline is None:
            print("[AnomalyRAGPipeline] 初始化RAG Pipeline...")
            self.rag_pipeline = get_rag_pipeline(self.config_path)
            # 加载已有FAISS索引
            if self.rag_pipeline.vectorstore.index is None:
                loaded = self.rag_pipeline.load_existing_knowledge_base()
                if loaded:
                    print(f"[AnomalyRAGPipeline] FAISS索引加载成功，文档数: {self.rag_pipeline.vectorstore.index.ntotal}")
                else:
                    print("[AnomalyRAGPipeline] ⚠️ FAISS索引加载失败")
    
    # ===== Step 1: 异常检测 =====
    
    def detect(self, segments_df: Optional[pd.DataFrame] = None, progress_callback=None) -> List[AnomalyResult]:
        """Detect only supplied points. None explicitly redetects official test points."""
        from src.detection import frozen
        judges = frozen.load_bundle()
        if segments_df is None:
            split = json.loads((project_root / frozen.SPLIT_JSON).read_text(encoding="utf-8"))
            raw = pd.read_csv(project_root / "data/raw/segments.csv", float_precision="round_trip")
            segments_df = raw[raw.segment.isin(split["test_ids"])].copy()
            self.last_input_mode = "official_test_redetection"
        else:
            self.last_input_mode = "supplied_points"
        if progress_callback:
            progress_callback(10, 100, "冻结检测包校验完成")
        features = frozen.point_features(segments_df)
        results = frozen.predict_features(features, judges)
        self.last_seg_results = results
        # Input identity is independent of the frozen training-data identity.
        required = ["channel", "segment", "timestamp", "value", "sampling"]
        import hashlib
        input_json = json.dumps(segments_df[required].to_numpy(dtype=object).tolist(),
                                default=str, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        self.last_input_sha256 = hashlib.sha256(input_json.encode()).hexdigest()
        anomalies = []
        for idx, row in results.iterrows():
            if not row.final_pred:
                continue
            summary = self._extract_feature_summary(features.iloc[[idx]], frozen.FEATURE_COLS)
            anomalies.append(AnomalyResult(
                segment=row.segment, channel=row.channel, anomaly_score=float(row.display_score),
                anomaly_type=self._classify_anomaly_type(summary, row.channel), feature_summary=summary,
                is_anomaly=bool(row.final_pred), rule_hits=int(row.rule_hits), rule_pred=row.rule_pred,
                if_pred=row.if_pred, if_score=row.score, gate_op=row.gate_op,
                gate_pred=bool(row.gate_pred), final_pred=bool(row.final_pred), postprocess=row.postprocess,
                display_score=float(row.display_score), detector_version=row.detector_version,
                feature_version=row.feature_version))
        if progress_callback:
            progress_callback(100, 100, "正式门控判定完成")
        return anomalies

    def _extract_feature_summary(self, seg_windows, feature_cols):
        """提取该段的关键特征统计。

        ⚠ 历史 bug：本方法曾读 `n_peaks_1.0` / `crest_factor` / `impulse_factor` /
        `min` / `max` / `zero_crossing_rate`，这些列在数据集里**全部不存在**
        （实际列见 framework.FEATURE_COLS，共 18 个），因此除 mean/std 外
        全部取不到值、summary 里只剩两项，其余键在下游表现为「缺失」。
        现改为按 FEATURE_SUMMARY_KEYS 取真实存在的列。
        """
        summary = {}
        # 段级数据每段只有一行；窗口级数据按段内均值聚合，两种都能用
        for feat in FEATURE_SUMMARY_KEYS:
            if feat in seg_windows.columns:
                vals = pd.to_numeric(seg_windows[feat], errors="coerce").dropna()
                if len(vals) > 0:
                    summary[feat] = float(vals.mean())

        return summary

    def _classify_anomaly_type(self, feat_summary: Dict, channel: str) -> str:
        """根据段级特征统计判定异常形态。

        Returns:
            str: intermittent_spikes / high_freq_jitter / low_activity / unusual_shapes

        规则依据（阈值与富集度见 utils/constants.py，均由 fit 集标定、val 集验证）：
          - intermittent_spikes：n_peaks>=3（一阶差分峰<=2）
              实测「n_peaks>=3 & diff_peaks<=2」象限异常率 1.000（fit 16/16、
              val 1/1），富集度 4.96 —— 数据中唯一强判别形态。
          - high_freq_jitter：diff_peaks>=10
              异常段一阶差分峰数 p80=13，正常段中位仅 6。
          - low_activity：n_peaks<=1 且 diff2_peaks<=5
              异常段二阶差分峰数 p20=1，峰数与差分同时压低= 信号近乎静止。
          - 其余一律 fallback，不编造数据不支持的类型。
            （原 zero_values / gaps 已被移除：实测 var<1e-10 的段异常率 0.217
              与基准 0.201 无差异；gaps_squared 全体 AUC 0.452，弱于随机，
              不足以支撑「数据间隙」这一类型。）

        ⚠ 只用跨通道可比的计数特征。std/var/mean 跨通道差4 个数量级
          （磁力计 ≈2e-5 vs 光电二极管 ≈0.3），不做绝对阈值判定。
          channel 参数保留给下游按通道物理含义二次加工，当前规则不依赖它。
        """
        th = TYPE_THRESHOLDS

        n_peaks = feat_summary.get("n_peaks")
        diff_peaks = feat_summary.get("diff_peaks")
        diff2_peaks = feat_summary.get("diff2_peaks")

        # 特征缺失时无法判定，直接 fallback（不做任何猜测）
        if n_peaks is None or diff_peaks is None or diff2_peaks is None:
            return "unusual_shapes"

        # 1) 间歇性尖峰：峰数高于正常段中位，且帧间差分极低
        if (n_peaks >= th["intermittent_spikes_n_peaks_min"]
                and diff_peaks <= th["intermittent_spikes_diff_peaks_max"]):
            return "intermittent_spikes"

        # 2) 持续高频抖动：一阶差分峰数超过异常段 p80
        if diff_peaks >= th["high_freq_jitter_diff_peaks_min"]:
            return "high_freq_jitter"

        # 3) 低活动：峰数与二阶差分同时处于低分位
        if (n_peaks <= th["low_activity_n_peaks_max"]
                and diff2_peaks <= th["low_activity_diff2_peaks_max"]):
            return "low_activity"

        # 4) fallback
        return "unusual_shapes"

    # ===== Step 2: RAG解释 =====
    
    def explain(self, anomaly: AnomalyResult) -> RAGExplanation:
        """
        为单个异常生成RAG解释
        
        Args:
            anomaly: 异常检测结果
        
        Returns:
            RAGExplanation: RAG生成的解释
        """
        self._init_rag()
        
        # 构建异常描述
        channel_desc = CHANNEL_PHYSICS.get(anomaly.channel, anomaly.channel)
        anomaly_desc = self._build_anomaly_description(anomaly)
        
        # 调用RAG pipeline
        print(f"\n[RAG] 分析异常: Segment {anomaly.segment}, {anomaly.channel}")
        result = self.rag_pipeline.analyze_anomaly(
            channel_id=anomaly.channel,
            anomaly_type=anomaly.anomaly_type,
            anomaly_description=anomaly_desc
        )
        
        return RAGExplanation(
            answer=result["answer"],
            sources=result["sources"],
            retrieval_time=result["metadata"].get("retrieval_time", 0),
            generation_time=result["metadata"].get("generation_time", 0)
        )
    
    def _build_anomaly_description(self, anomaly: AnomalyResult) -> str:
        """构建异常的文字描述"""
        channel_desc = CHANNEL_PHYSICS.get(anomaly.channel, anomaly.channel)
        
        desc_parts = [
            f"检测到{channel_desc}通道异常",
            f"异常类型: {anomaly.anomaly_type}",
            f"展示分: {anomaly.display_score:.2f}（不改变正式标签）",
            f"正式判定: {anomaly.final_pred}; 门控: {anomaly.gate_op}; 版本: {anomaly.detector_version}",
        ]
        
        # 添加关键特征（键名与 _extract_feature_summary 产出一致）
        feat = anomaly.feature_summary
        label = ANOMALY_TYPE_LABELS.get(anomaly.anomaly_type, anomaly.anomaly_type)
        desc_parts[1] = f"异常类型: {anomaly.anomaly_type}（{label}）"
        for feat_key, desc_fmt in (
            ("mean", "均值: {:.4g}"),
            ("std", "标准差: {:.4g}"),
            ("kurtosis", "峰度: {:.3f}"),
            ("n_peaks", "峰值数: {:.1f}"),
            ("diff_peaks", "一阶差分峰数: {:.1f}"),
        ):
            if feat_key in feat:
                desc_parts.append(desc_fmt.format(feat[feat_key]))

        return "; ".join(desc_parts)
    
    # ===== Step 3: 联合Pipeline =====
    
    def detect_and_explain(
        self,
        segments_df: Optional[pd.DataFrame] = None,
        max_explanations: int = 10,
        progress_callback=None
    ) -> List[CombinedResult]:
        """
        完整流程：检测异常 + 生成RAG解释

        Args:
            segments_df: 可选的segments数据（None则自动加载）
            max_explanations: 最多解释多少个异常（RAG API调用有限制）
            progress_callback: 进度回调函数 callback(current, total, message)

        Returns:
            List[CombinedResult]: 联合结果列表
        """
        print("\n" + "="*60)
        print("AnomalyRAGPipeline: 完整流程")
        print("="*60)

        # Step 1: 检测（将 detect 的 0-100 映射到整体进度的 0-10）
        if progress_callback:
            progress_callback(0, 100, "正在运行异常检测算法...")
        def _detect_progress(current, total, message):
            if progress_callback:
                mapped = int(10 * current / total)
                progress_callback(mapped, 100, message)
        anomalies = self.detect(segments_df, progress_callback=_detect_progress if progress_callback else None)

        if not anomalies:
            print("\n未检测到异常")
            return []

        # Step 2: 并发为每个异常生成解释（限制数量）
        self._init_rag()  # 提前初始化，避免线程竞争
        anomalies_to_explain = anomalies[:min(len(anomalies), max_explanations)]
        n_to_explain = len(anomalies_to_explain)

        print(f"\n将为前 {n_to_explain} 个异常并发生成RAG解释...")

        completed_count = 0

        def _explain_one(idx_anomaly):
            nonlocal completed_count
            idx, anomaly = idx_anomaly
            facts = {name: getattr(anomaly, name) for name in (
                "rule_hits", "rule_pred", "if_pred", "if_score", "gate_op", "gate_pred", "final_pred",
                "postprocess", "display_score", "detector_version", "feature_version")}
            print(f"\n[{idx+1}/{n_to_explain}] Segment {anomaly.segment}, {anomaly.channel}")
            try:
                rag_result = self.explain(anomaly)
                result = CombinedResult(
                    segment=anomaly.segment,
                    channel=anomaly.channel,
                    anomaly_score=anomaly.anomaly_score,
                    anomaly_type=anomaly.anomaly_type,
                    feature_summary=anomaly.feature_summary,
                    rag_explanation=rag_result.answer,
                    rag_sources=rag_result.sources,
                    retrieval_time=rag_result.retrieval_time,
                    generation_time=rag_result.generation_time,
                    **facts
                )
            except Exception as e:
                print(f"  [ERROR] RAG解释失败: {e}")
                result = CombinedResult(
                    segment=anomaly.segment,
                    channel=anomaly.channel,
                    anomaly_score=anomaly.anomaly_score,
                    anomaly_type=anomaly.anomaly_type,
                    feature_summary=anomaly.feature_summary,
                    rag_explanation=f"RAG解释生成失败: {str(e)}",
                    rag_sources=[],
                    retrieval_time=0,
                    generation_time=0,
                    **facts
                )
            completed_count += 1
            if progress_callback:
                pct = 10 + int(90 * completed_count / n_to_explain)
                progress_callback(pct, 100, f"RAG解释生成中... ({completed_count}/{n_to_explain})")
            return idx, result

        indexed_results = [None] * n_to_explain
        max_workers = 4
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_explain_one, (i, a)): i
                for i, a in enumerate(anomalies_to_explain)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                indexed_results[idx] = result

        results = [r for r in indexed_results if r is not None]
        print(f"\n完成！共生成 {len(results)} 条结果")
        return results
    
    def to_json(self, results: List[CombinedResult], output_path: Optional[str] = None):
        """Save a new engineering run. Never overwrite an existing result."""
        from datetime import datetime, timezone
        from uuid import uuid4
        from src.detection.frozen import VERSION
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
        path = Path(output_path) if output_path is not None else (
            project_root / "data/results/runs" / VERSION / f"{run_id}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"detector_version": VERSION, "run_id": run_id, "results": [asdict(r) for r in results]}
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        print(f"[Saved] {path}")
        return path



def detect_and_explain(config_path: Optional[str] = None) -> List[CombinedResult]:
    """快捷函数：一步完成检测+解释"""
    pipeline = AnomalyRAGPipeline(config_path)
    return pipeline.detect_and_explain()


if __name__ == "__main__":
    """测试"""
    print("=== AnomalyRAGPipeline 测试 ===\n")
    
    pipeline = AnomalyRAGPipeline()
    results = pipeline.detect_and_explain(max_explanations=3)
    
    if results:
        print(f"\n检测到 {len(results)} 条异常，已生成RAG解释")
        
        # 保存结果
        output_path = pipeline.to_json(results)
        
        # 打印第一条结果示例
        print("\n" + "="*60)
        print("示例结果:")
        print("="*60)
        r = results[0]
        print(f"Segment: {r.segment}")
        print(f"Channel: {r.channel}")
        print(f"Anomaly Type: {r.anomaly_type}")
        print(f"Anomaly Score: {r.anomaly_score:.3f}")
        print(f"\nRAG解释:\n{r.rag_explanation[:500]}...")
    else:
        print("未检测到异常")
