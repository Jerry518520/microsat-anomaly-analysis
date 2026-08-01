"""
Stage 2 判定器 — 与 ipynb(微小卫星遥测异常检测与RAG解释系统_完整Pipeline)
Stage 2「IF + 统计规则融合」严格一致的流式判定逻辑:

- 训练数据: data/processed/segments_18d.csv(缺失时按 ipynb 回退路径
  用 extract_segments_18d.generate_dataset 从 segments.csv 生成并缓存);
- 特征: 18 维段级统计特征(feature_cols = 全部列 - META_COLS, 顺序同源);
- 强通道(CADC0872/873/874): IsolationForest(n_estimators=100,
  contamination=0.2, random_state=42, 无 max_samples) predict()==-1
  OR 分通道 3σ/IQR 规则;
- 弱通道(其余 6 个): 纯规则(不建 IF 模型);
- 规则: 训练集上分通道计算各特征的 mean/std/q1/q3/iqr,
  单段违反(3σ 或 1.5·IQR 出界)累计 ≥2 条即判异常;
- 判定为二值 verdict。severity / anomaly_score 仅是展示层着色,
  不参与判定(见 Verdict 注释)。

首次训练后 joblib 落盘 data/models/stage2_detector.joblib, 后续直接加载。
"""
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.streaming.segment_features import FEATURE_COLS
from src.utils.constants import META_COLS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "stage2_detector.joblib")
FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "segments_18d.csv")

# 与 ipynb Stage 1/2 完全一致
STRONG_CHANNELS = ["CADC0872", "CADC0873", "CADC0874"]
WEAK_CHANNELS = ["CADC0884", "CADC0886", "CADC0888", "CADC0890", "CADC0892", "CADC0894"]
ALL_CHANNELS = STRONG_CHANNELS + WEAK_CHANNELS
CONTAMINATION = 0.2
RANDOM_STATE = 42
MIN_TRAIN = 10
CACHE_VERSION = "stage2-ipynb-aligned-2"

METHOD_DESC = "Stage 2: 强通道 IF(c=0.2) OR 3σ/IQR 规则; 弱通道纯规则; 段闭合判定(与 ipynb 一致)"


@dataclass
class Verdict:
    """单段判定结果。

    anomaly 是唯一正式判定输出(严格 = ipynb Stage 2 公式)。
    severity / anomaly_score 是展示层着色与进度条, 由判据强度派生:
    - 强通道: IF 与规则同时命中 → critical; 仅其一 → warning
    - 弱通道: 规则违例 ≥4 条 → critical; 3 条 → warning; 2 条 → caution
    它们不影响 anomaly, 也不对应任何离线指标。
    """
    channel: str
    anomaly: bool
    if_pred: Optional[bool]      # 强通道 IF predict 结果; 弱通道 None
    rule_hits: int               # 规则违例条数(0~36)
    rule_pred: bool
    score: Optional[float]       # 展示用连续分(强通道 = -decision_function; 弱通道 None)
    anomaly_score: float         # 0~1 展示进度条
    severity: str                # nominal/caution/warning/critical


@dataclass
class ChannelJudge:
    """单通道判定器: 强通道持 IF 模型, 全部通道持统计阈值"""
    channel: str
    model: Optional[IsolationForest]
    thresholds: dict             # feature -> {mean,std,q1,q3,iqr}
    score_tau95: float = 0.0     # 训练分 95 分位, 仅用于展示层归一化
    score_std: float = 1.0       # 训练分标准差, 仅用于展示层归一化

    def rule_eval(self, feats: dict[str, float]) -> int:
        """与 ipynb apply_rules 一致: 3σ 或 1.5·IQR 出界各计 1 条"""
        hits = 0
        for col, th in self.thresholds.items():
            v = feats.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if abs(v - th["mean"]) > 3.0 * th["std"]:
                hits += 1
            lo = th["q1"] - 1.5 * th["iqr"]
            hi = th["q3"] + 1.5 * th["iqr"]
            if v < lo or v > hi:
                hits += 1
        return hits

    def continuous_score(self, vec: np.ndarray) -> Optional[float]:
        """展示用连续异常分(强通道 = -decision_function); 弱通道无 IF, 返回 None"""
        if self.model is None:
            return None
        x = np.nan_to_num(vec.reshape(1, -1))
        return float(-self.model.decision_function(x)[0])

    def display_score(self, feats: dict[str, float], rule_hits: int,
                      score: Optional[float]) -> float:
        """0~1 展示进度条: 强通道按超出训练 τ95 的幅度(3σ 满分), 弱通道按违例数/6"""
        if score is not None:
            over = max(0.0, score - self.score_tau95)
            return round(min(1.0, over / (3 * self.score_std)), 4)
        return round(min(1.0, rule_hits / 6.0), 4)

    def judge(self, feats: dict[str, float]) -> Verdict:
        vec = np.array([feats[c] for c in FEATURE_COLS], dtype=float)
        rule_hits = self.rule_eval(feats)
        rule_pred = rule_hits >= 2

        if_pred: Optional[bool] = None
        score: Optional[float] = None
        if self.model is not None:
            x = np.nan_to_num(vec.reshape(1, -1))
            if_pred = bool(self.model.predict(x)[0] == -1)
            score = self.continuous_score(vec)
            anomaly = bool(if_pred or rule_pred)          # 强通道: OR 融合
        else:
            anomaly = rule_pred                            # 弱通道: 纯规则

        # 展示层 severity(不影响判定)
        if not anomaly:
            severity = "nominal"
        elif self.model is not None:
            severity = "critical" if (if_pred and rule_pred) else "warning"
        else:
            severity = "critical" if rule_hits >= 4 else ("warning" if rule_hits == 3 else "caution")

        return Verdict(
            channel=self.channel, anomaly=anomaly, if_pred=if_pred,
            rule_hits=rule_hits, rule_pred=rule_pred, score=score,
            anomaly_score=self.display_score(feats, rule_hits, score),
            severity=severity,
        )


def _load_or_generate_features() -> pd.DataFrame:
    """与 ipynb 第 1 节一致: 优先读缓存, 缺失则 generate_dataset 生成"""
    if os.path.exists(FEATURES_CSV):
        print(f"[Stage2] 加载段级特征: {FEATURES_CSV}")
        return pd.read_csv(FEATURES_CSV)
    print("[Stage2] segments_18d.csv 不存在, 从 segments.csv 生成(与 ipynb 回退路径一致)")
    from src.features.extract_segments_18d import generate_dataset
    raw = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "segments.csv"),
                      parse_dates=["timestamp"])
    df = generate_dataset(raw)
    os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False, encoding="utf-8-sig")
    return df


def _compute_stat_thresholds(train_df: pd.DataFrame, feature_cols: list) -> dict:
    """与 ipynb compute_stat_thresholds 逐行一致"""
    thresholds: dict[str, dict] = {}
    for ch in ALL_CHANNELS:
        ch_data = train_df[train_df["channel"] == ch]
        ch_th: dict[str, dict] = {}
        for col in feature_cols:
            vals = ch_data[col].dropna()
            if len(vals) < 5:
                continue
            std = float(vals.std())
            ch_th[col] = {
                "mean": float(vals.mean()),
                "std": std if std > 1e-6 else 1.0,
                "q1": float(vals.quantile(0.25)),
                "q3": float(vals.quantile(0.75)),
                "iqr": float(vals.quantile(0.75) - vals.quantile(0.25)),
            }
        thresholds[ch] = ch_th
    return thresholds


def train_detector() -> dict[str, ChannelJudge]:
    df = _load_or_generate_features()
    feature_cols = [c for c in df.columns if c not in META_COLS]
    if feature_cols != FEATURE_COLS:
        missing = [c for c in FEATURE_COLS if c not in feature_cols]
        if missing:
            raise ValueError(f"段级特征缺少列: {missing}")
        feature_cols = FEATURE_COLS  # 顺序对齐

    train_df = df[df["train"] == True].copy()  # noqa: E712 — 与 ipynb 一致
    thresholds = _compute_stat_thresholds(train_df, feature_cols)

    judges: dict[str, ChannelJudge] = {}
    for ch in ALL_CHANNELS:
        ch_train = train_df[train_df["channel"] == ch]
        model: Optional[IsolationForest] = None
        tau95, std = 0.0, 1.0
        # IF 的跳过规则(len<10)只针对 IF, 与 ipynb 一致;
        # 规则阈值不受此限(ipynb 的 apply_rules 对任何通道都算)
        if ch in STRONG_CHANNELS:
            if len(ch_train) < MIN_TRAIN:
                print(f"[Stage2] {ch}: 训练段不足({len(ch_train)}<{MIN_TRAIN}), IF 跳过(规则仍生效)")
            else:
                X = ch_train[feature_cols].values
                model = IsolationForest(n_estimators=100, contamination=CONTAMINATION,
                                        random_state=RANDOM_STATE, n_jobs=-1)
                model.fit(X)
                train_scores = -model.decision_function(X)
                tau95 = float(np.quantile(train_scores, 0.95))
                std = float(np.std(train_scores)) or 1.0
        th = thresholds.get(ch, {})
        if model is None and not th:
            print(f"[Stage2] {ch}: 无 IF 且无规则阈值, 跳过")
            continue
        judges[ch] = ChannelJudge(ch, model, th, tau95, std)
        role = "强通道 IF+规则" if model is not None else "弱通道 纯规则"
        print(f"[Stage2] {ch}: 就绪 (n={len(ch_train)}, {role}, 规则阈值 {len(th)} 特征)")
    return judges


def load_or_train() -> dict[str, ChannelJudge]:
    """优先从 data/models/ 加载缓存, 否则训练并落盘"""
    if os.path.exists(MODEL_PATH):
        try:
            payload = joblib.load(MODEL_PATH)
            if payload.get("version") == CACHE_VERSION and payload.get("feature_cols") == FEATURE_COLS:
                print(f"[Stage2] 从缓存加载 {len(payload['judges'])} 个通道判定器")
                return payload["judges"]
            print("[Stage2] 缓存版本/特征列不匹配, 重新训练")
        except Exception as e:
            print(f"[Stage2] 缓存加载失败({e}), 重新训练")

    t0 = time.time()
    judges = train_detector()
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"version": CACHE_VERSION, "feature_cols": FEATURE_COLS, "judges": judges}, MODEL_PATH)
    print(f"[Stage2] 训练完成({time.time() - t0:.1f}s), 已保存: {MODEL_PATH}")
    return judges
