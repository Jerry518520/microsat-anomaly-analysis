import { useCallback, useEffect, useRef, useState } from 'react';
import { ChevronRight, RefreshCw, Radio, Square } from 'lucide-react';
import { api } from '../api/client';
import type { AlertItem, ChannelInfo, CoreMetrics, DiagnosisTarget, Severity } from '../api/types';
import ChannelCell from '../components/ChannelCell';
import CountUp from '../components/CountUp';
import OrbitViz from '../components/OrbitViz';
import SemiGauge from '../components/SemiGauge';
import { sevColor, sevLampClass, scorePct, SEVERITY_LABELS } from '../components/severity';
import { useLiveStream } from '../hooks/useLiveStream';

const REFRESH_INTERVAL = 15_000;
const REPLAY_SPEEDS = [60, 300, 1000, 2000];

const fmtSimClock = (ts: number | null) =>
  ts == null ? '--' : new Date(ts * 1000).toISOString().slice(0, 19).replace('T', ' ') + 'Z';

type DataMode = 'archive' | 'live';

export default function DashboardView({ onDiagnose }: { onDiagnose: (t: DiagnosisTarget) => void }) {
  const [mode, setMode] = useState<DataMode>('archive');
  const [metrics, setMetrics] = useState<CoreMetrics | null>(null);
  const [channels, setChannels] = useState<ChannelInfo[]>([]);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [sevFilter, setSevFilter] = useState<Severity | 'all'>('all');
  const [speed, setSpeed] = useState(300);
  const labelRef = useRef<Record<string, string>>({});

  const live = useLiveStream(mode === 'live');

  const fetchAll = useCallback(async (silent = false) => {
    if (!silent) setRefreshing(true);
    try {
      const [m, c, a] = await Promise.all([
        api.dashboard.metrics(),
        api.dashboard.channels(),
        api.dashboard.alerts(),
      ]);
      setMetrics(m);
      setChannels(c.channels || []);
      setAlerts(a.alerts || []);
      for (const ch of c.channels || []) labelRef.current[ch.channel] = ch.label;
      setError(null);
    } catch {
      setError('TELEMETRY LINK LOST — 无法连接后端 (localhost:8000)');
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
    const timer = setInterval(() => fetchAll(true), REFRESH_INTERVAL);
    return () => clearInterval(timer);
  }, [fetchAll]);

  const switchMode = (m: DataMode) => {
    setMode(m);
    if (m === 'archive') fetchAll(true);
  };

  // ===== LIVE 模式数据装配 =====
  const isLive = mode === 'live';
  const liveChannels: ChannelInfo[] = isLive
    ? Object.entries(live.series).map(([ch, s]) => ({
        channel: ch,
        label: labelRef.current[ch] ?? ch,
        anomaly_rate: s.anomalyFlags.length
          ? s.anomalyFlags.filter(Boolean).length / s.anomalyFlags.length
          : 0,
        status: s.model === 'unavailable' ? ('offline' as const) : s.severity,
        sparkline_values: s.values,
        anomaly_indices: s.anomalyFlags.flatMap((f, i) => (f ? [i] : [])),
        provisional: s.provisional,
      }))
    : [];
  const viewChannels = isLive ? liveChannels : channels;
  const viewAlerts: AlertItem[] = isLive ? live.alerts : alerts;

  const filtered = sevFilter === 'all' ? viewAlerts : viewAlerts.filter(a => a.severity === sevFilter);
  const rateColor = !metrics ? '#22d3ee' : metrics.anomaly_rate > 0.4 ? '#f43f5e' : metrics.anomaly_rate > 0.15 ? '#fbbf24' : '#2dd4a7';

  const liveStats = live.state?.stats ?? { total_points: 0, judged_segments: 0, preview_scores: 0, alerts: 0 };
  // LIVE 仪表 = 已判定段的异常占比(正式判定层口径, 与离线 anomaly_rate 同义)
  const liveJudgedRate = liveStats.judged_segments > 0 ? liveStats.alerts / liveStats.judged_segments : 0;
  const modelsReady = live.state
    ? Object.values(live.state.channels).filter(c => c.model === 'ready').length
    : 0;

  const archiveKpis = [
    { label: '告警队列', value: metrics?.fault_count ?? 0, unit: 'ALERTS', color: '#fb923c', sub: `${alerts.filter(a => a.severity === 'critical').length} CRITICAL`, fixed: 0 },
    { label: '最佳 SegF1', value: metrics?.best_f1 ?? 0, unit: 'STAGE-2', color: '#2dd4a7', sub: 'IF + 规则融合', fixed: 4 },
    { label: '数据吞吐', value: metrics ? metrics.throughput / 1000 : 0, unit: 'K ROWS', color: '#22d3ee', sub: '18 天在轨遥测', fixed: 0 },
    { label: '监测通道', value: metrics?.channel_count ?? 0, unit: 'CH', color: '#a78bfa', sub: '3× MAG + 6× PD', fixed: 0 },
  ] as const;
  const liveKpis = [
    { label: '实时告警', value: liveStats.alerts, unit: 'ALERTS', color: '#fb923c', sub: `${live.alerts.filter(a => a.severity === 'critical').length} CRITICAL`, fixed: 0 },
    { label: '流式点数', value: liveStats.total_points / 1000, unit: 'K PTS', color: '#22d3ee', sub: `SIM ${fmtSimClock(live.simTs)}`, fixed: 1 },
    { label: '判定段数', value: liveStats.judged_segments, unit: 'SEG', color: '#2dd4a7', sub: '段闭合 · Stage 2 正式判定', fixed: 0 },
    { label: '在线判定器', value: modelsReady, unit: '/ 9 CH', color: '#a78bfa', sub: 'IF×3 + 规则×9', fixed: 0 },
  ] as const;
  const kpis = isLive ? liveKpis : archiveKpis;

  return (
    <div className="h-full flex gap-3 min-h-0">
      {/* ========== 左轨:KPI ========== */}
      <div className="w-[218px] shrink-0 flex flex-col gap-3 min-h-0">
        <div className="deck-panel rise p-3 flex flex-col items-center">
          <div className="self-start text-[10px] f-disp font-semibold tracking-[0.22em] tx-3 uppercase mb-1">
            {isLive ? '判定段异常率' : '系统异常率'}
          </div>
          <SemiGauge value={isLive ? liveJudgedRate : metrics?.anomaly_rate ?? 0} color={isLive ? '#22d3ee' : rateColor} size={168} />
          <div className="text-[10px] font-mono tx-3 mt-1">
            {isLive
              ? `${liveStats.alerts.toLocaleString()} / ${liveStats.judged_segments.toLocaleString()} SEG · 预览 ${liveStats.preview_scores.toLocaleString()} 次`
              : metrics ? `${metrics.anomaly_rows.toLocaleString()} / ${metrics.total_rows.toLocaleString()} ROWS` : 'SYNCING…'}
          </div>
        </div>

        {kpis.map((k, i) => (
          <div key={k.label} className={`deck-panel rise rise-${i + 1} p-3`}>
            <div className="text-[10px] f-disp font-semibold tracking-[0.22em] tx-3 uppercase">{k.label}</div>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="big-readout text-[26px] leading-none" style={{ color: k.color, textShadow: `0 0 14px ${k.color}55` }}>
                <CountUp value={k.value} format={v => v.toFixed(k.fixed)} />
              </span>
              <span className="text-[9px] font-mono tx-3">{k.unit}</span>
            </div>
            <div className="text-[9px] font-mono tx-3 mt-1">{k.sub}</div>
          </div>
        ))}
      </div>

      {/* ========== 中央:轨道态势 + 通道矩阵 ========== */}
      <div className="flex-1 flex flex-col gap-3 min-h-0 min-w-0">
        <div className="deck-panel rise rise-1 flex-1 min-h-0 overflow-hidden relative">
          <div className="deck-head absolute top-0 left-0 right-0 z-10 bg-transparent border-b-0 justify-between">
            <span className="flex items-center gap-2"><span className="tick" /> 轨道态势 · ORBITAL SITUATION</span>
            {/* 数据模式开关 */}
            <span className="flex items-center gap-1 normal-case tracking-normal">
              {(['archive', 'live'] as const).map(m => (
                <button key={m} onClick={() => switchMode(m)}
                  className="text-[9px] font-mono px-2 py-0.5 rounded transition-all"
                  style={mode === m
                    ? m === 'live'
                      ? { color: '#f43f5e', background: 'rgba(244,63,94,0.12)', boxShadow: 'inset 0 0 0 1px rgba(244,63,94,0.5)' }
                      : { color: '#22d3ee', background: 'rgba(34,211,238,0.12)', boxShadow: 'inset 0 0 0 1px rgba(34,211,238,0.4)' }
                    : { color: 'var(--tx-3)' }}>
                  {m === 'live' ? '● LIVE' : 'ARCHIVE'}
                </button>
              ))}
            </span>
          </div>
          {error ? (
            <div className="h-full flex items-center justify-center">
              <span className="font-mono text-xs text-[#fbbf24]">{error}</span>
            </div>
          ) : (
            <OrbitViz alerts={viewAlerts} />
          )}
          {/* 角落 HUD */}
          <div className="absolute bottom-2 left-3 font-mono text-[9px] z-10">
            {isLive ? (
              <span className="text-[#f43f5e]">
                SIMULATED LIVE · 2022 遥测回放 ×{live.speed} · SIM {fmtSimClock(live.simTs)}
                {!live.connected && ' · WS 重连中…'}
                <span className="tx-3"> · {live.state?.method ?? 'Stage 2 段判定'}</span>
              </span>
            ) : (
              <span className="tx-3">ESA OPS-SAT · 18-DAY TELEMETRY WINDOW</span>
            )}
          </div>
          <div className="absolute bottom-2 right-3 z-10 flex items-center gap-2">
            {isLive ? (
              <>
                <select value={speed} onChange={e => setSpeed(Number(e.target.value))}
                  className="bg-[rgba(8,16,32,0.8)] text-[10px] font-mono tx-2 border border-[rgba(94,234,212,0.2)] rounded px-1 py-0.5">
                  {REPLAY_SPEEDS.map(s => <option key={s} value={s}>×{s}</option>)}
                </select>
                {live.mode === 'replay' ? (
                  <button onClick={() => api.stream.replayStop()} className="ghost-btn !py-0.5 !px-2 text-[10px] !text-[#f43f5e]">
                    <Square size={10} /> STOP
                  </button>
                ) : (
                  <button onClick={() => api.stream.replayStart(speed)} className="ghost-btn !py-0.5 !px-2 text-[10px] !text-[#f43f5e]">
                    <Radio size={10} /> {live.state?.models_ready === false ? 'TRAINING…' : 'START REPLAY'}
                  </button>
                )}
              </>
            ) : (
              <button onClick={() => fetchAll()} disabled={refreshing} className="ghost-btn !py-0.5 !px-2 text-[10px]">
                <RefreshCw size={10} className={refreshing ? 'animate-spin' : ''} /> SYNC
              </button>
            )}
          </div>
        </div>

        <div className="deck-panel rise rise-2 shrink-0">
          <div className="deck-head"><span className="tick" /> 遥测通道矩阵 · 9CH MATRIX {isLive && <span className="text-[#f43f5e] text-[9px] font-mono ml-2">REALTIME</span>}</div>
          <div className="p-2 grid grid-cols-3 gap-1.5">
            {viewChannels.map((ch, i) => (
              <ChannelCell key={ch.channel} data={ch} index={i}
                onClick={() => onDiagnose({ channelId: ch.channel, segment: '', channelLabel: ch.label })} />
            ))}
            {viewChannels.length === 0 && (
              <div className="col-span-3 py-6 text-center font-mono text-[10px] tx-3">
                {isLive ? 'WAITING FOR STREAM…(点击 START REPLAY)' : 'ACQUIRING CHANNELS…'}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ========== 右轨:告警队列 ========== */}
      <div className="w-[330px] shrink-0 deck-panel rise rise-3 flex flex-col min-h-0">
        <div className="deck-head justify-between">
          <span className="flex items-center gap-2"><span className="tick" /> 故障告警队列 {isLive && <span className="text-[#f43f5e] text-[9px] font-mono">LIVE</span>}</span>
          <span className="font-mono text-[10px] text-[#fb923c]">{filtered.length}/{viewAlerts.length}</span>
        </div>
        <div className="flex gap-1 px-2.5 py-2 border-b border-[rgba(94,234,212,0.08)]">
          {(['all', 'critical', 'warning', 'caution'] as const).map(f => (
            <button key={f} onClick={() => setSevFilter(f)}
              className="text-[9px] font-mono px-1.5 py-0.5 rounded transition-all"
              style={sevFilter === f
                ? { color: '#22d3ee', background: 'rgba(34,211,238,0.12)', boxShadow: 'inset 0 0 0 1px rgba(34,211,238,0.4)' }
                : { color: 'var(--tx-3)' }}>
              {f === 'all' ? 'ALL' : f.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="flex-1 overflow-y-auto p-2.5 space-y-2 min-h-0">
          {filtered.map((a, i) => {
            const color = sevColor(a.severity);
            const isLiveAlert = a.segment.startsWith('LIVE-');
            return (
              <div key={`${a.segment}-${a.channel}-${i}`} className="alert-card"
                style={{ ['--ac' as string]: color }}
                onClick={() => !isLiveAlert && onDiagnose({ channelId: a.channel, segment: a.segment, score: a.anomaly_score, channelLabel: a.channel_label })}>
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-1.5">
                    <span className={sevLampClass(a.severity)} />
                    <span className="font-mono text-[11px] tx-1">{isLiveAlert ? a.segment : `SEG #${a.segment}`}</span>
                    <span className="font-mono text-[9px] text-[#22d3ee]">{a.channel}</span>
                  </div>
                  <span className="font-mono text-[11px] font-semibold" style={{ color }}>{a.anomaly_score.toFixed(3)}</span>
                </div>
                <div className="text-[10px] tx-3 mb-1.5">{a.channel_label} · {SEVERITY_LABELS[(a.severity as Severity)] ?? a.severity}</div>
                <p className="text-[10px] leading-relaxed tx-2 line-clamp-2">{a.summary}</p>
                <div className="flex items-center justify-between mt-2">
                  <div className="flex-1 h-0.5 mr-2 rounded bg-[rgba(94,234,212,0.1)] overflow-hidden">
                    <div className="h-full" style={{ width: `${scorePct(a.anomaly_score)}%`, background: color, boxShadow: `0 0 5px ${color}` }} />
                  </div>
                  {isLiveAlert ? (
                    <span className="text-[9px] font-mono tx-3">RAG 诊断待生成</span>
                  ) : (
                    <span className="text-[9px] font-mono text-[#22d3ee] flex items-center">DIAGNOSE <ChevronRight size={10} /></span>
                  )}
                </div>
              </div>
            );
          })}
          {filtered.length === 0 && (
            <div className="py-10 text-center font-mono text-[10px] tx-3">NO ALERTS IN FILTER</div>
          )}
        </div>
      </div>
    </div>
  );
}
