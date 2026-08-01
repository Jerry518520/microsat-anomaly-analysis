// 轻量 fetch 封装 — 开发态走相对路径,由 Vite 代理到 :8000

import type {
  AlertItem,
  ChannelF1,
  ChannelInfo,
  CoreMetrics,
  ExperimentStage,
  ExplanationDetail,
  RagConfig,
  RagQueryResponse,
  StreamState,
  SystemParams,
  SystemStatus,
  WaveformData,
} from './types';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`API ${path} 返回 ${res.status}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API ${path} 返回 ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  health: () => get<{ status: string; version: string }>('/api/health'),

  dashboard: {
    metrics: () => get<CoreMetrics>('/api/dashboard/metrics'),
    channels: () => get<{ channels: ChannelInfo[] }>('/api/dashboard/channels'),
    alerts: () => get<{ alerts: AlertItem[]; total: number }>('/api/dashboard/alerts'),
    status: () => get<SystemStatus>('/api/dashboard/status'),
  },

  detection: {
    ragConfig: () => get<RagConfig>('/api/detection/rag-config'),
    experiments: () => get<{ experiments: ExperimentStage[] }>('/api/detection/experiments'),
    channelF1: () => get<{ channels: ChannelF1[]; available: boolean }>('/api/detection/channel-f1'),
    systemParams: () => get<SystemParams>('/api/detection/system-params'),
  },

  explanation: {
    detail: (segment: string, channel: string) =>
      get<ExplanationDetail>(`/api/explanation/detail?segment=${encodeURIComponent(segment)}&channel=${encodeURIComponent(channel)}`),
    waveform: (channel: string, segment?: string, contextSize = 200) =>
      get<WaveformData>(
        `/api/explanation/waveform?channel=${encodeURIComponent(channel)}${segment ? `&segment=${encodeURIComponent(segment)}` : ''}&context_size=${contextSize}`,
      ),
    query: async (query: string, channel: string, segment: string): Promise<RagQueryResponse> => {
      const res = await fetch('/api/explanation/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, channel, segment, context: {} }),
      });
      if (!res.ok) throw new Error(`RAG 问答返回 ${res.status}`);
      return res.json() as Promise<RagQueryResponse>;
    },
  },

  stream: {
    state: () => get<StreamState>('/api/stream/state'),
    replayStart: (speed: number) =>
      post<{ mode: string; speed: number }>('/api/stream/replay/start', { speed }),
    replayStop: () => post<{ mode: string }>('/api/stream/replay/stop', {}),
  },
};
