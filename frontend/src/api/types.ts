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
  /** LIVE 模式: 当前状态来自增长段预览分(非正式判定) */
  provisional?: boolean;
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

// ===== 实时流(LIVE 模式)契约 — 对齐 /api/stream =====

export interface LiveChannelState {
  model: 'ready' | 'unavailable';
  open_segment_len: number;
  last_ts: number | null;
  last_value: number | null;
  score: number | null;
  severity: Severity;
  /** true = 增长段预览分(段未闭合,非正式判定); false = 段闭合 Stage 2 正式判定 */
  provisional: boolean;
}

export interface LiveAlert extends AlertItem {
  first_ts: number;
  last_ts: number;
  raw_score: number;
  hits: number;
  closed_ts?: number;
}

export interface StreamStats {
  total_points: number;
  /** 段闭合正式判定次数(Stage 2) */
  judged_segments: number;
  /** 增长段预览判分次数(非正式) */
  preview_scores: number;
  alerts: number;
}

export interface StreamState {
  mode: 'idle' | 'replay' | 'live';
  speed: number;
  sim_ts: number | null;
  models_ready: boolean;
  /** 判定方法说明(Stage 2, 与 ipynb 一致) */
  method: string;
  channels: Record<string, LiveChannelState>;
  alerts: LiveAlert[];
  stats: StreamStats;
}

export interface StreamBatch {
  type: 'batch';
  sim_ts: number | null;
  mode: 'idle' | 'replay' | 'live';
  speed: number;
  points: Record<string, [number, number][]>;
  scores: Record<string, { ts: number; score: number; severity: Severity; provisional: boolean }>;
  alerts: { action: 'new' | 'update' | 'closed'; alert: LiveAlert }[];
  stats: StreamStats;
}
