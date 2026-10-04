"""
异常检测 + RAG解释 联合Pipeline

功能：
1. 运行完整异常检测流程
   - 有 fusion.json 配方时：按 gate_perchannel 逐通道选 rule/if/AND/OR
     （9 个通道各自独立，参数由 scripts/fusion_v3.py 在 val 上选出）
   - 无配方时：回退 Scheme G（强通道 IF+OR 规则 / 弱通道段级 baseline）
   - psi=128 为工程固定参数；vote_threshold=0.25 为工程设定值（无 val 选优）
2. 对检测到的异常自动生成RAG解释查询
3. 调用RAG pipeline获取知识库支撑的分析报告
4. 返回结构化的联合结果

标签依赖说明：
- 推理链路（阈值计算 / IF 训练 / 门控融合 / 段级聚合）不读取 `anomaly` 标签
- 超参 best_k / best_contamination_per_channel / gate_choice 由
  scripts/fusion_v3.py 在 val 集上选出，属离线阶段，与线上推理无关
- last_seg_results 中的 y_true 字段仅供离线评估与前端可视化，
  不参与任何预测计算

使用：
    pipeline = AnomalyRAGPipeline()
    results = pipeline.detect_and_explain()
    # results = [{"segment": 123, "channel": "CADC0874", "anomaly_type": "peaks",
    #             "explanation": "...", "sources": [...]}]
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
from src.utils.constants import CHANNEL_PHYSICS, META_COLS, NO_ANOMALY_CHANNELS
from src.utils.stat_rules import compute_stat_thresholds, apply_stat_rules
from src.features.segment_features import run_segment_baseline
from src.rag.pipeline import RAGPipeline, get_rag_pipeline


@dataclass
class AnomalyResult:
    """单条异常检测结果"""
    segment: int
    channel: str
    anomaly_score: float  # 段级异常率 (0-1)
    anomaly_type: str  # peaks/zero_values/unusual_shapes/gaps
    feature_summary: Dict[str, float]  # 关键特征统计
    is_anomaly: bool = True


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
    segment: int
    channel: str
    anomaly_score: float
    anomaly_type: str
    feature_summary: Dict[str, float]
    rag_explanation: str
    rag_sources: List[Dict[str, Any]]
    retrieval_time: float
    generation_time: float


class AnomalyRAGPipeline:
    """
    异常检测 + RAG解释 联合Pipeline
    
    集成了：
    1. 最优检测方案 (Scheme G + threshold 0.25 + psi=128)
    2. 自动异常类型识别
    3. RAG知识库解释生成
    """
    
    # 强通道（磁力计，F1>0.47）
    STRONG_CHANNELS = {"CADC0872", "CADC0873", "CADC0874"}

    # ---------------------------------------------------------------- 论文配方
    # ⚠ 原先本管线把 9 个通道的 contamination、psi 等超参全部写死在 __init__，
    #   且融合固定为「强通道 IF+OR 规则 / 弱通道段级 baseline」两分支。
    #   这与实验代码 scripts/fusion_v3.py 选出的 gate_perchannel 配方不一致，
    #   导致论文报告的 test F1=0.6281 在产品路径上复现不出来。
    #   现改为：从 data/results/v3/fusion.json 读取 val 上选出的真实配方。
    #   配方文件缺失时回退到原有硬编码行为（保持向后兼容，不静默降级）。
    RECIPE_PATH = project_root / "data" / "results" / "v3" / "fusion.json"

    @classmethod
    def load_recipe(cls) -> Optional[Dict[str, Any]]:
        """读取 fusion.json 中的 gate_perchannel 配方。

        返回 None 表示文件不存在或结构不符，调用方须回退到硬编码路径。
        """
        try:
            if not cls.RECIPE_PATH.exists():
                return None
            data = json.loads(cls.RECIPE_PATH.read_text(encoding="utf-8"))
            sel = data["results"]["selection"]
            choice = sel["gate_perchannel_choice"]
            contam = sel["best_contamination_per_channel"]
            k = sel["best_k"]
            if not isinstance(choice, dict) or not choice:
                return None
            return {
                "gate_choice": {ch: v["op"] for ch, v in choice.items()},
                "contamination": {ch: contam.get(ch) for ch in choice},
                "best_k": k,
                "git_commit": data.get("meta", {}).get("git_commit", "unknown"),
            }
        except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
            print(f"[AnomalyRAGPipeline] 配方读取失败，回退硬编码：{e}")
            return None

    def __init__(self, config_path: Optional[str] = None):
        """初始化联合Pipeline"""
        self.config = load_config()
        self.config_path = config_path
        self.rag_pipeline = None  # 延迟初始化

        # 检测参数
        # ⚠ vote_threshold 是**无选优过程的工程设定值**，注释原文写「最优投票阈值」
        #   不准确。fusion_v3.py 中不存在同名参数，被 val 选出的是各软融合策略的
        #   tau（soft_global=0.15 / soft_perchannel 各通道 0.05~0.35 / soft_calibrated=0.2），
        #   语义是「归一化分数阈值」；本值语义是「异常窗口占比阈值」，两者不同。
        #   数值同为 0.25 属巧合，无因果关系。
        #   段级数据每段仅 1 行（n_windows==1），该阈值实际作用于
        #   _compute_segment_confidence 输出的连续分数。
        self.vote_threshold = 0.25  # 工程设定值，非 val 选出
        self.psi = 128  # 子采样数，工程固定参数

        # 论文配方（val 选出）。载入失败时 recipe=None，走原有硬编码分支。
        self.recipe = self.load_recipe()
        if self.recipe is not None:
            print(f"[AnomalyRAGPipeline] 已载入论文配方 gate_perchannel "
                  f"(k={self.recipe['best_k']}, commit={self.recipe['git_commit'][:8]})")
        else:
            print("[AnomalyRAGPipeline] 未找到 fusion.json 配方，使用内置硬编码参数")

        # 各通道最优contamination。仅在无配方时作为回退值使用。
        self.best_contamination = {
            "CADC0872": 0.5, "CADC0873": 0.5, "CADC0874": 0.5,
            "CADC0886": 0.2, "CADC0888": 0.45, "CADC0890": 0.3,
            "CADC0892": 0.5, "CADC0894": 0.5,
        }
        if self.recipe is not None:
            self.best_contamination = {
                ch: (v if v is not None else 0.5)
                for ch, v in self.recipe["contamination"].items()
            }

        print("[AnomalyRAGPipeline] 初始化完成")
    
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
        """
        运行完整异常检测流程

        Args:
            segments_df: 可选的segments数据（None则自动加载）
            progress_callback: 进度回调函数 callback(current, total, message)

        Returns:
            List[AnomalyResult]: 检测到的异常列表
        """
        print("\n" + "="*60)
        print("Step 1: 异常检测流程")
        print("="*60)

        # 1.1 加载数据
        if segments_df is None:
            segments_df = load_segments(self.config)
        if progress_callback:
            progress_callback(5, 100, "加载数据完成")

        # 1.2 加载段级18维特征
        print(f"\n[1/5] 加载段级18维特征...")
        if progress_callback:
            progress_callback(15, 100, "正在加载段级特征...")
        root = get_project_root()
        feat_path = os.path.join(root, self.config["data"]["raw_dir"], self.config["data"]["features_file"])
        features_df = pd.read_csv(feat_path, encoding="utf-8")

        feature_cols = [c for c in features_df.columns if c not in META_COLS]
        train_mask = features_df["train"] == 1
        # ⚠ 这里用 train 全集（1594 段）拟合阈值与 IF，而实验代码用 fit（1275 段）。
        #   两者口径不同是有意的，且都正确：
        #   - 实验侧拆 fit/val 是为了在同口径下公平比较超参（防过拟合 val）；
        #   - 生产侧上线时全部历史数据都可用，用全集拟合参数更强。
        #   实测差异：生产 SegF1=0.6446 vs 实验 0.6281（TP78/FP51/FN35 vs
        #   TP76/FP53/FN37），两者 CI 高度重叠。
        #   若要与论文数字严格对齐，可改用 framework.load_split() 取 fit。
        #
        # ⚠⚠ 必须按 segment 显式排序，否则结果依赖 CSV 的物理行序。
        #   机理：IsolationForest 的 max_samples 子采样按**行索引**取样本，
        #   而 offset_ = percentile(训练集自身分数, 100*contamination)
        #   （sklearn/ensemble/_iforest.py:389）。random_state 只固定了随机数
        #   发生器，固定不了「哪些行被选中」—— 行序一变，被选中的具体样本就变，
        #   集成结构随之改变，offset_ 改变，最终预测改变。
        #   实测：同一 random_state=42、同一配方，仅改行序（CSV原序 / segment排序 /
        #   洗牌seed1 / 洗牌seed2），判异常数 138~144，F1 0.3294~0.3347，极差 0.0052。
        #   故此处按 segment 排序，使结果与文件物理顺序解耦。
        features_df = features_df.sort_values("segment").reset_index(drop=True)
        train_mask = features_df["train"] == 1
        train_df = features_df[train_mask]
        test_df = features_df[~train_mask].copy()

        # 1.3 统计规则
        print("[2/5] 应用统计规则...")
        if progress_callback:
            progress_callback(35, 100, "正在应用统计规则...")
        thresholds = compute_stat_thresholds(train_df, feature_cols)
        n_violated, n_total = apply_stat_rules(test_df, thresholds, feature_cols)
        test_df["rule_violations"] = n_violated
        test_df["rule_total"] = n_total
        # ⚠ 口径须与实验代码 fusion_v3.py 一致。
        #   原实现用「违规比例 >= 0.2」，实验代码用「违规数 >= k」（k 在 val 上选，
        #   实测 k=2）。两者不同：18 个特征时 0.2 比例 ≈ 违规数 >= 4，
        #   比 k=2 严格得多，会显著降低 rule 通道的召回。
        #   有配方时用 best_k，无配方时回退原比例口径。
        if self.recipe is not None:
            k_best = int(self.recipe["best_k"])
            rule_pred = pd.Series((n_violated >= k_best).astype(int), index=test_df.index)
            print(f"      规则判定: 违规数 >= {k_best}（与论文配方一致）")
        else:
            rule_pred = pd.Series(
                (n_violated / n_total.replace(0, 1) >= 0.2).astype(int),
                index=test_df.index,
            )
            print("      规则判定: 违规比例 >= 0.2（无配方，回退口径）")

        # 1.4 分通道IForest
        print("[3/5] 分通道IsolationForest...")
        if progress_callback:
            progress_callback(55, 100, "正在运行分通道 IsolationForest...")
        channel_preds = self._per_channel_iforest(train_df, test_df, feature_cols)

        # 1.5 段级baseline
        print("[4/5] 段级baseline...")
        if progress_callback:
            progress_callback(75, 100, "正在计算段级 baseline...")
        seg_baseline_map = self._segment_baseline_iforest(segments_df)

        # 1.6 Scheme G融合 + 投票阈值
        print("[5/5] Scheme G融合 + 投票阈值聚合...")
        if progress_callback:
            progress_callback(90, 100, "正在融合 Scheme G + 投票阈值...")
        pred_g = self._build_scheme_g(
            test_df, channel_preds, rule_pred, seg_baseline_map
        )
        
        # 段级聚合
        seg_results = self._segment_aggregate(test_df, pred_g, self.vote_threshold)
        
        # 暴露完整段级判定（含全部测试段，不只异常段），供重测脚本一次性取 y_pred
        self.last_seg_results = seg_results

        # 筛选异常段
        anomalies = seg_results[seg_results["is_anomaly"]]
        print(f"\n检测结果: {len(anomalies)} 个异常段 (共 {len(seg_results)} 测试段)")
        
        # 转换为AnomalyResult列表
        anomaly_results = []
        for _, row in anomalies.iterrows():
            # 获取特征统计
            seg_windows = test_df[test_df["segment"] == row["segment"]]
            feat_summary = self._extract_feature_summary(seg_windows, feature_cols)
            
            # 识别异常类型
            anomaly_type = self._classify_anomaly_type(feat_summary, row["channel"])
            
            anomaly_results.append(AnomalyResult(
                segment=int(row["segment"]),
                channel=row["channel"],
                anomaly_score=float(row["anomaly_score"]),
                anomaly_type=anomaly_type,
                feature_summary=feat_summary,
                is_anomaly=True
            ))
        
        return anomaly_results
    
    def _per_channel_iforest(self, train_df, test_df, feature_cols):
        """分通道训练IForest"""
        from sklearn.ensemble import IsolationForest
        
        channel_preds = {}
        for ch in sorted(test_df["channel"].unique()):
            ch_test = test_df[test_df["channel"] == ch]
            ch_train = train_df[train_df["channel"] == ch]
            
            if len(ch_train) < 10 or len(ch_test) < 5:
                continue
            
            ch_c = self.best_contamination.get(ch, 0.5)
            X_tr = np.nan_to_num(ch_train[feature_cols].values)
            X_te = np.nan_to_num(ch_test[feature_cols].values)
            
            # 使用psi参数
            model = IsolationForest(
                n_estimators=100, 
                max_samples=self.psi,  # 子采样参数
                contamination=ch_c, 
                random_state=42, 
                n_jobs=-1
            )
            model.fit(X_tr)
            ch_pred = (model.predict(X_te) == -1).astype(int)
            
            for i, idx in enumerate(ch_test.index):
                channel_preds[idx] = int(ch_pred[i])
        
        return channel_preds
    
    def _segment_baseline_iforest(self, segments_df):
        """段级baseline IForest"""
        from sklearn.ensemble import IsolationForest
        
        root = get_project_root()
        feat_path = os.path.join(root, self.config["data"]["raw_dir"], self.config["data"]["features_file"])
        dataset_df = pd.read_csv(feat_path, encoding="utf-8")
        
        meta_cols = ["channel", "segment", "anomaly", "train", "sampling"]
        feat_cols = [c for c in dataset_df.columns if c not in meta_cols]
        
        train_mask = dataset_df["train"] == 1
        X_train = dataset_df.loc[train_mask, feat_cols].values
        X_test = dataset_df.loc[~train_mask, feat_cols].values
        test_segments = dataset_df.loc[~train_mask, "segment"].values

        # ⚠ contamination=0.25 是**无选优过程的工程设定值**，不是 val 搜出来的。
        #   fusion_v3.py 的 select_if_contamination 是分通道搜索，产出的
        #   best_contamination_per_channel 里没有「全局段级 IF」这一项。
        #   故此值不可声称「由 val 最优选出」，只能称工程设定。
        #   且当 fusion.json 配方存在时，本函数的结果在 _build_scheme_g 中
        #   不会被使用（9 个通道的 op 全为 rule/if/AND/OR，无 seg_baseline），
        #   仅在配方缺失的回退路径生效。
        model = IsolationForest(
            n_estimators=100,
            max_samples=self.psi,
            contamination=0.25,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train)
        y_pred = (model.predict(X_test) == -1).astype(int)
        
        seg_pred_map = {}
        for i in range(len(test_segments)):
            seg_pred_map[test_segments[i]] = int(y_pred[i])
        
        return seg_pred_map
    
    def _build_scheme_g(self, test_df, channel_preds, rule_pred, seg_baseline_map):
        """构建门控融合预测（每通道按 val 选出的算子）。

        有配方时：对每个通道用 fusion.json 里 val 选定的算子
        （rule / if / AND / OR），与实验代码 gate_perchannel 口径一致。
        无配方时：回退到原 Scheme G（强通道 IF+OR 规则，弱通道段级 baseline）。
        """
        if self.recipe is None:
            pred_g = pd.Series(0, index=test_df.index, dtype=int)
            for idx in test_df.index:
                ch = test_df.loc[idx, "channel"]
                seg_id = test_df.loc[idx, "segment"]
                if ch in self.STRONG_CHANNELS:
                    if_pred = channel_preds.get(idx, 0)
                    r_pred = int(rule_pred.loc[idx])
                    pred_g.loc[idx] = max(if_pred, r_pred)
                else:
                    pred_g.loc[idx] = seg_baseline_map.get(seg_id, 0)
            return pred_g

        gate = self.recipe["gate_choice"]
        pred_g = pd.Series(0, index=test_df.index, dtype=int)
        for idx in test_df.index:
            ch = test_df.loc[idx, "channel"]
            seg_id = test_df.loc[idx, "segment"]
            op = gate.get(ch)

            if op == "rule":
                pred_g.loc[idx] = int(rule_pred.loc[idx])
            elif op == "if":
                pred_g.loc[idx] = int(channel_preds.get(idx, 0))
            elif op == "AND":
                pred_g.loc[idx] = int(rule_pred.loc[idx]) & int(channel_preds.get(idx, 0))
            elif op == "OR":
                pred_g.loc[idx] = int(rule_pred.loc[idx]) | int(channel_preds.get(idx, 0))
            else:
                # 配方里没有该通道（理论上不应发生），回退段级 baseline
                pred_g.loc[idx] = seg_baseline_map.get(seg_id, 0)
                print(f"[warn] 通道 {ch} 不在配方中，回退段级 baseline")

        return pred_g
    
    def _segment_aggregate(self, test_df, window_preds, threshold):
        """窗口预测聚合为段级结果"""
        test_df = test_df.copy()
        test_df["pred"] = window_preds.values if hasattr(window_preds, 'values') else window_preds

        seg = test_df.groupby("segment").agg(
            channel=("channel", "first"),
            y_true=("anomaly", "first"),
            anomaly_score=("pred", "mean"),  # 异常窗口比例
            n_windows=("pred", "count"),
        ).reset_index()

        # 如果每个段只有一行（段级特征），需要计算置信度分数
        if (seg["n_windows"] == 1).all():
            # 对于段级数据，使用特征值计算置信度
            # 基于规则违规比例和通道预测的组合
            seg["anomaly_score"] = seg.apply(
                lambda row: self._compute_segment_confidence(row, test_df), axis=1
            )

        seg["is_anomaly"] = seg["anomaly_score"] >= threshold

        # ⚠ 排除「全集合无正类样本」的通道。
        #   依据：CADC0884 在 fit=97段/val=25段/test=36段 上正类均为 0
        #   （三集合实测，见 tests/test_production_recipe.py 的断言）。
        #   即该通道在整个可用数据上无正类，训练与评估都无法进行。
        #
        #   ⚠ 不要把依据写成「test 上无真值异常」—— 那是拿测试集标签做决策。
        #     本常量只允许由「全集合无正类」推出，且须有测试断言守护。
        #
        #   注：门控配方里 0884 的 op="rule"、val_f1=0.0，门控本身就会忽略该通道，
        #       此处显式排除是冗余保护，不改变结果。
        if NO_ANOMALY_CHANNELS:
            no_anom_mask = seg["channel"].isin(NO_ANOMALY_CHANNELS)
            seg.loc[no_anom_mask, "is_anomaly"] = False
            seg.loc[no_anom_mask, "anomaly_score"] = 0.0

        return seg

    def _compute_segment_confidence(self, row, test_df):
        """为段级数据计算置信度分数（0-1）"""
        seg_id = row["segment"]
        seg_data = test_df[test_df["segment"] == seg_id].iloc[0]

        # 基于规则违规比例
        rule_ratio = seg_data.get("rule_violations", 0) / max(seg_data.get("rule_total", 1), 1)

        # 基于 pred 值（0 或 1）
        pred_val = float(seg_data.get("pred", 0))

        # 组合置信度：pred=1 时给较高分数，pred=0 时给较低分数
        # 同时考虑规则违规比例作为调整因子
        if pred_val == 1:
            confidence = 0.7 + 0.3 * rule_ratio  # 0.7-1.0
        else:
            confidence = 0.0 + 0.3 * rule_ratio  # 0.0-0.3

        return min(max(confidence, 0.0), 1.0)
    
    def _extract_feature_summary(self, seg_windows, feature_cols):
        """提取该段的关键特征统计"""
        summary = {}
        key_features = ["mean", "std", "min", "max", "n_peaks_1.0", "zero_crossing_rate", 
                        "crest_factor", "impulse_factor"]
        
        for feat in key_features:
            if feat in seg_windows.columns:
                summary[feat] = float(seg_windows[feat].mean())
        
        return summary
    
    def _classify_anomaly_type(self, feat_summary: Dict, channel: str) -> str:
        """
        根据特征统计分类异常类型
        
        Returns:
            str: peaks / zero_values / unusual_shapes / gaps
        """
        mean_val = feat_summary.get("mean", 0)
        min_val = feat_summary.get("min", 0)
        n_peaks = feat_summary.get("n_peaks_1.0", 0)
        crest_factor = feat_summary.get("crest_factor", 1)
        
        # 零值判断：均值接近0且最小值接近0
        if abs(mean_val) < 1e-6 and abs(min_val) < 1e-6:
            return "zero_values"
        
        # 尖峰判断：峰值数多 + 波峰因子高
        if n_peaks > 2 and crest_factor > 3:
            return "peaks"
        
        # 数据间隙：这个需要原始数据判断，这里暂时归为unusual_shapes
        # 实际中可通过seg_windows长度与预期长度的差异判断
        
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
            f"异常得分: {anomaly.anomaly_score:.2f} (阈值{self.vote_threshold})",
        ]
        
        # 添加关键特征
        feat = anomaly.feature_summary
        if "mean" in feat:
            desc_parts.append(f"均值: {feat['mean']:.4f}")
        if "std" in feat:
            desc_parts.append(f"标准差: {feat['std']:.4f}")
        if "n_peaks_1.0" in feat:
            desc_parts.append(f"峰值数: {feat['n_peaks_1.0']:.1f}")
        
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
                    generation_time=rag_result.generation_time
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
                    generation_time=0
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
    
    def to_json(self, results: List[CombinedResult], output_path: str):
        """保存结果到JSON"""
        data = [asdict(r) for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"[Saved] {output_path}")


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
        output_path = os.path.join(get_project_root(), "data/results/anomaly_rag_results.json")
        pipeline.to_json(results, output_path)
        
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
