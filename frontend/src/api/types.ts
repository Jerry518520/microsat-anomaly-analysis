// API 契约类型 — 严格对齐仓库 API.md / FRONTEND_CONTEXT.md

export type Severity = 'nominal' | 'caution' | 'warning' | 'critical' | 'offline';

export interface CoreMetrics {
  anomaly_rate: number;
  anomaly_rate_str: string;
  anomaly_rows: number;
  total_rows: number;
  fault_count: number;
  total_anomalies: number;
  best_f1: number;
  throughput: number;
  channel_count: number;
}

export interface ChannelInfo {
  channel: string;
  label: string;
  anomaly_rate: number;
  status: Severity;
  sparkline_values: number[];
  anomaly_indices: number[];
}

export interface AlertItem {
  segment: string;
  channel: string;
  channel_label: string;
  anomaly_score: number;
  severity: Severity;
  summary: string;
  anomaly_type: string | null;
}

export interface SystemStatus {
  segments_available: boolean;
  rag_available: boolean;
  rag_error: string | null;
}

export interface RagConfig {
  status: 'online' | 'offline';
  doc_count: { pdf: number; md: number; html: number };
  chunk_count: number | null;
  embedding_model: string;
}

export interface ExperimentStage {
  version: string;
  strategy: string;
  f1: number;
  improvement: string;
}

export interface ChannelF1 {
  channel: string;
  f1: number;
  precision: number | null;
  recall: number | null;
}

export interface SystemParams {
  algorithm: string;
  dimensions: number;
  contamination: number;
}

export interface DiagnosisSections {
  '可能原因'?: string;
  '影响评估'?: string;
  '结论'?: string;
  '建议措施'?: string;
  '来源'?: string;
}

export interface SourceItem {
  filename: string;
  page?: string;
  score?: number;
  local_path?: string;
}

export interface ExplanationDetail {
  segment: string;
  channel: string;
  channel_label: string;
  anomaly_type: string;
  anomaly_score: number;
  urgency: string;
  urgency_level: 'low' | 'medium' | 'high' | 'critical';
  sections: DiagnosisSections;
  garbled: boolean;
  feature_summary: Record<string, unknown> | string | null;
  sources: SourceItem[];
  raw_explanation?: string;
  error?: string;
}

export interface WaveformData {
  timestamps: (string | number)[];
  values: number[];
  anomaly_mask: number[];
  mean: number;
  std: number;
  upper_3sigma: number;
  lower_3sigma: number;
  channel: string;
  channel_label: string;
  error?: string;
}

export interface RagQueryResponse {
  answer: string;
  sources?: { filename: string; score: number }[];
  error?: string;
}

export interface DiagnosisTarget {
  channelId: string;
  segment: string;
  score?: number;
  channelLabel?: string;
}
