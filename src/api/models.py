"""Pydantic 数据模型 — 定义所有 API 响应结构"""
from pydantic import BaseModel
from typing import Optional


class CoreMetrics(BaseModel):
    # 口径警告：anomaly_rate / anomaly_rate_str / anomaly_rows / total_rows 均为【点级】口径，
    # 分母是遥测采样点数（303493），不是段数。段级口径异常率为 20.44%（434 / 2123 段），
    # 是论文主表口径。两者分母不同、不可混用或相减。详见 docs/指标口径说明.md。
    anomaly_rate: float  # 点级：anomaly_rows / total_rows（采样点口径）
    anomaly_rate_str: str  # 点级，如 "33.0%"
    anomaly_rows: int  # 点级：异常采样点数
    total_rows: int  # 点级：总采样点数（非段数）
    fault_count: int
    total_anomalies: int
    best_f1: float  # 段级 SegF1（注意：与上面点级字段口径不同）
    throughput: int
    channel_count: int = 9


class ChannelInfo(BaseModel):
    channel: str
    label: str
    anomaly_rate: float  # 点级：该通道内异常采样点占比（分母为采样点数，非段数）
    status: str  # nominal | caution | warning | critical
    sparkline_values: list[float]
    anomaly_indices: list[int]


class AlertItem(BaseModel):
    segment: str
    channel: str
    channel_label: str
    anomaly_score: float
    severity: str  # nominal | caution | warning | critical
    summary: str
    anomaly_type: Optional[str] = None


class RagConfig(BaseModel):
    status: str  # online | offline
    doc_count: dict[str, int]
    chunk_count: Optional[int] = None
    embedding_model: str


class ExperimentStage(BaseModel):
    version: str
    strategy: str
    f1: float
    improvement: str


class ChannelF1(BaseModel):
    channel: str
    f1: float
    precision: Optional[float] = None
    recall: Optional[float] = None


class DiagnosisReport(BaseModel):
    channel: str
    channel_label: str
    anomaly_type: str
    anomaly_score: float
    urgency: str
    sections: dict[str, str]  # section_name -> content
    feature_summary: Optional[dict] = None
    sources: list[dict] = []
    raw_explanation: str = ""


class SystemStatus(BaseModel):
    segments_available: bool
    rag_available: bool
    rag_error: Optional[str] = None
    utc_time: str
