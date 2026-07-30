"""Pydantic 数据模型 — 定义所有 API 响应结构"""
from pydantic import BaseModel
from typing import Optional


class CoreMetrics(BaseModel):
    anomaly_rate: float
    anomaly_rate_str: str
    anomaly_rows: int
    total_rows: int
    fault_count: int
    total_anomalies: int
    best_f1: float
    throughput: int
    channel_count: int = 9


class ChannelInfo(BaseModel):
    channel: str
    label: str
    anomaly_rate: float
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
