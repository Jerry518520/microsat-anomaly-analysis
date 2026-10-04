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

/**
 * API Key —— 后端自 2026-10-04 起对除 /api/health 外的所有接口启用认证。
 *
 * 密钥来源优先级：
 *   1. `VITE_API_KEY` 环境变量（构建时注入，推荐）
 *   2. `localStorage.api_key`（开发态临时切换后端用）
 *   3. 都没有 → 不带 header，由后端返回 401
 *
 * ⚠ 生产部署不要把 key 打进前端包：Vite 的 `VITE_*` 变量会被内联进
 *   产物，等于公开。这里保留读取是为了让本地开发与联调能跑通；
 *   正式上线应在反向代理（Nginx/网关）层注入并做鉴权，而不是靠前端。
 */
function apiKey(): string {
  const fromEnv = (import.meta as { env?: Record<string, string> }).env
    ?.VITE_API_KEY;
  if (fromEnv) return fromEnv;
  try {
    return window.localStorage.getItem('api_key') ?? '';
  } catch {
    return '';
  }
}

function authHeaders(): Record<string, string> {
  const k = apiKey();
  return k ? { 'X-API-Key': k } : {};
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: authHeaders() });
  if (res.status === 401) {
    throw new Error('未授权（401）：请在 localStorage.api_key 或 VITE_API_KEY 配置 API Key');
  }
  if (res.status === 503) {
    throw new Error('后端未配置 API_AUTH_KEYS（503），请检查后端 .env');
  }
  if (res.status === 429) {
    const ra = res.headers.get('Retry-After') ?? '若干';
    throw new Error(`请求过于频繁（429），请 ${ra} 秒后重试`);
  }
  if (!res.ok) throw new Error(`API ${path} 返回 ${res.status}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (res.status === 401) {
    throw new Error('未授权（401）：请在 localStorage.api_key 或 VITE_API_KEY 配置 API Key');
  }
  if (res.status === 429) {
    const ra = res.headers.get('Retry-After') ?? '若干';
    throw new Error(`请求过于频繁（429），请 ${ra} 秒后重试`);
  }
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
