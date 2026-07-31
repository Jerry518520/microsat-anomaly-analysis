import { useCallback, useEffect, useState } from 'react';
import { ChevronRight, RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import type { AlertItem, ChannelInfo, CoreMetrics, DiagnosisTarget, Severity } from '../api/types';
import ChannelCell from '../components/ChannelCell';
import CountUp from '../components/CountUp';
import OrbitViz from '../components/OrbitViz';
import SemiGauge from '../components/SemiGauge';
import { sevColor, sevLampClass, scorePct, SEVERITY_LABELS } from '../components/severity';

const REFRESH_INTERVAL = 15_000;

export default function DashboardView({ onDiagnose }: { onDiagnose: (t: DiagnosisTarget) => void }) {
  const [metrics, setMetrics] = useState<CoreMetrics | null>(null);
  const [channels, setChannels] = useState<ChannelInfo[]>([]);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [sevFilter, setSevFilter] = useState<Severity | 'all'>('all');

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

  const filtered = sevFilter === 'all' ? alerts : alerts.filter(a => a.severity === sevFilter);
  const rateColor = !metrics ? '#22d3ee' : metrics.anomaly_rate > 0.4 ? '#f43f5e' : metrics.anomaly_rate > 0.15 ? '#fbbf24' : '#2dd4a7';

  return (
    <div className="h-full flex gap-3 min-h-0">
      {/* ========== 左轨:KPI ========== */}
      <div className="w-[218px] shrink-0 flex flex-col gap-3 min-h-0">
        <div className="deck-panel rise p-3 flex flex-col items-center">
          <div className="self-start text-[10px] f-disp font-semibold tracking-[0.22em] tx-3 uppercase mb-1">系统异常率</div>
          <SemiGauge value={metrics?.anomaly_rate ?? 0} color={rateColor} size={168} />
          <div className="text-[10px] font-mono tx-3 mt-1">
            {metrics ? `${metrics.anomaly_rows.toLocaleString()} / ${metrics.total_rows.toLocaleString()} ROWS` : 'SYNCING…'}
          </div>
        </div>

        {([
          { label: '告警队列', value: metrics?.fault_count ?? 0, unit: 'ALERTS', color: '#fb923c', sub: `${alerts.filter(a => a.severity === 'critical').length} CRITICAL` },
          { label: '最佳 SegF1', value: metrics?.best_f1 ?? 0, unit: 'STAGE-2', color: '#2dd4a7', sub: 'IF + 规则融合', fixed: 4 },
          { label: '数据吞吐', value: metrics ? metrics.throughput / 1000 : 0, unit: 'K ROWS', color: '#22d3ee', sub: '18 天在轨遥测', fixed: 0 },
          { label: '监测通道', value: metrics?.channel_count ?? 0, unit: 'CH', color: '#a78bfa', sub: '3× MAG + 6× PD' },
        ] as const).map((k, i) => (
          <div key={k.label} className={`deck-panel rise rise-${i + 1} p-3`}>
            <div className="text-[10px] f-disp font-semibold tracking-[0.22em] tx-3 uppercase">{k.label}</div>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="big-readout text-[26px] leading-none" style={{ color: k.color, textShadow: `0 0 14px ${k.color}55` }}>
                <CountUp value={k.value} format={v => v.toFixed('fixed' in k ? (k as { fixed?: number }).fixed ?? 0 : 0)} />
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
          <div className="deck-head absolute top-0 left-0 right-0 z-10 bg-transparent border-b-0">
            <span className="tick" /> 轨道态势 · ORBITAL SITUATION
          </div>
          {error ? (
            <div className="h-full flex items-center justify-center">
              <span className="font-mono text-xs text-[#fbbf24]">{error}</span>
            </div>
          ) : (
            <OrbitViz alerts={alerts} />
          )}
          {/* 角落 HUD */}
          <div className="absolute bottom-2 left-3 font-mono text-[9px] tx-3 z-10">
            ESA OPS-SAT · 18-DAY TELEMETRY WINDOW
          </div>
          <div className="absolute bottom-2 right-3 z-10 flex items-center gap-2">
            <button onClick={() => fetchAll()} disabled={refreshing} className="ghost-btn !py-0.5 !px-2 text-[10px]">
              <RefreshCw size={10} className={refreshing ? 'animate-spin' : ''} /> SYNC
            </button>
          </div>
        </div>

        <div className="deck-panel rise rise-2 shrink-0">
          <div className="deck-head"><span className="tick" /> 遥测通道矩阵 · 9CH MATRIX</div>
          <div className="p-2 grid grid-cols-3 gap-1.5">
            {channels.map((ch, i) => (
              <ChannelCell key={ch.channel} data={ch} index={i}
                onClick={() => onDiagnose({ channelId: ch.channel, segment: '', channelLabel: ch.label })} />
            ))}
            {channels.length === 0 && (
              <div className="col-span-3 py-6 text-center font-mono text-[10px] tx-3">ACQUIRING CHANNELS…</div>
            )}
          </div>
        </div>
      </div>

      {/* ========== 右轨:告警队列 ========== */}
      <div className="w-[330px] shrink-0 deck-panel rise rise-3 flex flex-col min-h-0">
        <div className="deck-head justify-between">
          <span className="flex items-center gap-2"><span className="tick" /> 故障告警队列</span>
          <span className="font-mono text-[10px] text-[#fb923c]">{filtered.length}/{alerts.length}</span>
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
            return (
              <div key={`${a.segment}-${a.channel}-${i}`} className="alert-card"
                style={{ ['--ac' as string]: color }}
                onClick={() => onDiagnose({ channelId: a.channel, segment: a.segment, score: a.anomaly_score, channelLabel: a.channel_label })}>
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-1.5">
                    <span className={sevLampClass(a.severity)} />
                    <span className="font-mono text-[11px] tx-1">SEG #{a.segment}</span>
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
                  <span className="text-[9px] font-mono text-[#22d3ee] flex items-center">DIAGNOSE <ChevronRight size={10} /></span>
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
