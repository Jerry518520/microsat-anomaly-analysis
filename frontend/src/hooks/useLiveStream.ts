// LIVE 模式实时流 hook — WebSocket 连接 /api/stream/ws,维护各通道序列与实时告警
import { useEffect, useRef, useState } from 'react';
import type { LiveAlert, Severity, StreamBatch, StreamState } from '../api/types';

export interface LiveChannelSeries {
  values: number[];
  /** 与 values 对齐: 该点所属批次最近一次判分是否越限(含预览着色) */
  anomalyFlags: boolean[];
  severity: Severity;
  score: number | null;
  /** true = 增长段预览分(非正式判定); false = 段闭合 Stage 2 正式判定 */
  provisional: boolean;
  model: 'ready' | 'unavailable';
}

const MAX_POINTS = 500;

export interface LiveStream {
  connected: boolean;
  state: StreamState | null;
  simTs: number | null;
  speed: number;
  mode: 'idle' | 'replay' | 'live';
  alerts: LiveAlert[];
  series: Record<string, LiveChannelSeries>;
}

export function useLiveStream(active: boolean): LiveStream {
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<StreamState | null>(null);
  const [simTs, setSimTs] = useState<number | null>(null);
  const [mode, setMode] = useState<'idle' | 'replay' | 'live'>('idle');
  const [speed, setSpeed] = useState(60);
  const [alerts, setAlerts] = useState<LiveAlert[]>([]);
  const [series, setSeries] = useState<Record<string, LiveChannelSeries>>({});

  // 告警表用 ref 维护(按 segment 去重/更新),state 只存快照数组
  const alertMapRef = useRef<Map<string, LiveAlert>>(new Map());
  const seriesRef = useRef<Record<string, LiveChannelSeries>>({});

  useEffect(() => {
    if (!active) return;
    let ws: WebSocket | null = null;
    let closed = false;
    let retry = 0;
    let timer: ReturnType<typeof setTimeout>;

    const connect = () => {
      if (closed) return;
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      // ⚠ 浏览器的 WebSocket API **不能设置自定义请求头**（无可用的
      //   `headers` 选项），所以 API Key 只能走 URL query 传。
      //   后端 `src/api/security.py` 的 `require_api_key` 已同时检查
      //   `?api_key=`（供 WS 使用）与 `X-API-Key` 头（供 HTTP 使用）。
      //   注意：query 形式会进访问日志，部署时应在反向代理层限制
      //   /api/stream/ws 的访问日志记录，或改用一次性 token。
      const key = (() => {
        const fromEnv = (import.meta as { env?: Record<string, string> }).env
          ?.VITE_API_KEY;
        if (fromEnv) return fromEnv;
        try {
          return window.localStorage.getItem('api_key') ?? '';
        } catch {
          return '';
        }
      })();
      const qs = key ? `?api_key=${encodeURIComponent(key)}` : '';
      ws = new WebSocket(`${proto}://${location.host}/api/stream/ws${qs}`);

      ws.onopen = () => { retry = 0; setConnected(true); };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) {
          retry += 1;
          timer = setTimeout(connect, Math.min(8000, 500 * 2 ** retry));
        }
      };
      ws.onerror = () => ws?.close();

      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'snapshot') {
          const s = msg.state as StreamState;
          setState(s);
          setMode(s.mode);
          setSpeed(s.speed);
          setSimTs(s.sim_ts);
          alertMapRef.current = new Map(s.alerts.map(a => [a.segment, a]));
          setAlerts([...alertMapRef.current.values()]);
          // 用快照初始化通道状态(序列为空,等待 points 流入)
          const init: Record<string, LiveChannelSeries> = {};
          for (const [ch, c] of Object.entries(s.channels)) {
            init[ch] = {
              values: [], anomalyFlags: [],
              severity: c.severity, score: c.score,
              provisional: c.provisional, model: c.model,
            };
          }
          seriesRef.current = init;
          setSeries(init);
        } else if (msg.type === 'batch') {
          const b = msg as StreamBatch;
          setSimTs(b.sim_ts);
          setMode(b.mode);
          setSpeed(b.speed);

          const cur = seriesRef.current;
          for (const [ch, pts] of Object.entries(b.points)) {
            const s = cur[ch] ??= {
              values: [], anomalyFlags: [],
              severity: 'nominal', score: null, provisional: true, model: 'ready',
            };
            const flagged = (b.scores[ch]?.severity ?? s.severity) !== 'nominal';
            for (const [, v] of pts) {
              s.values.push(v);
              s.anomalyFlags.push(flagged);
            }
            if (s.values.length > MAX_POINTS) {
              const drop = s.values.length - MAX_POINTS;
              s.values.splice(0, drop);
              s.anomalyFlags.splice(0, drop);
            }
          }
          for (const [ch, sc] of Object.entries(b.scores)) {
            const s = cur[ch] ??= {
              values: [], anomalyFlags: [],
              severity: 'nominal', score: null, provisional: true, model: 'ready',
            };
            s.severity = sc.severity;
            s.score = sc.score;
            s.provisional = sc.provisional;
          }
          setSeries({ ...cur });

          let alertsChanged = false;
          for (const { action, alert } of b.alerts) {
            if (action === 'new') alertMapRef.current.set(alert.segment, alert);
            else if (alertMapRef.current.has(alert.segment)) alertMapRef.current.set(alert.segment, alert);
            alertsChanged = true;
          }
          if (alertsChanged) setAlerts([...alertMapRef.current.values()]);
        }
      };
    };

    connect();
    return () => {
      closed = true;
      clearTimeout(timer);
      ws?.close();
      alertMapRef.current.clear();
      seriesRef.current = {};
      setAlerts([]);
      setSeries({});
      setConnected(false);
    };
  }, [active]);

  return { connected, state, simTs, speed, mode, alerts, series };
}
