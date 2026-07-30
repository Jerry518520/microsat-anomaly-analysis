import { useCallback, useEffect, useState } from 'react';
import { Activity, ChevronRight, Clock, RefreshCw, ShieldAlert, Database, Radio, BrainCircuit } from 'lucide-react';
import { api } from '../api/client';
import type { AlertItem, ChannelInfo, CoreMetrics, DiagnosisTarget } from '../api/types';
import ChannelCard from '../components/ChannelCard';
import { sevColor, sevDotClass, scorePct, SEVERITY_LABELS } from '../components/severity';
import type { Severity } from '../api/types';

const REFRESH_INTERVAL = 15_000;

/* ---------- 环形仪表 ---------- */
function RateGauge({ rate, color }: { rate: number; color: string }) {
  const r = 30;
  const c = 2 * Math.PI * r;
  const pct = Math.min(1, Math.max(0, rate));
  return (
    <svg width="76" height="76" viewBox="0 0 76 76" className="shrink-0">
      <circle cx="38" cy="38" r={r} fill="none" stroke="rgba(148,163,184,0.12)" strokeWidth="6" />
      <circle
        cx="38" cy="38" r={r} fill="none"
        stroke={color} strokeWidth="6" strokeLinecap="round"
        strokeDasharray={c}
        strokeDashoffset={c * (1 - pct)}
        transform="rotate(-90 38 38)"
        style={{ filter: `drop-shadow(0 0 6px ${color}88)`, transition: 'stroke-dashoffset 0.8s cubic-bezier(0.22,1,0.36,1)' }}
      />
      <text x="38" y="42" textAnchor="middle" fill="#E6EDF7" fontSize="15" fontWeight="700" fontFamily="Rajdhani, sans-serif">
        {(pct * 100).toFixed(0)}%
      </text>
    </svg>
  );
}

/* ---------- KPI 卡 ---------- */
function KpiCard({ icon, label, value, unit, sub, accent, delay }: {
  icon: React.ReactNode; label: string; value: string; unit?: string; sub?: string; accent: string; delay: string;
}) {
  return (
    <div className={`panel corner-framed p-4 rise ${delay}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-[11px] uppercase tracking-[0.15em] tx-3 font-display font-semibold">{label}</span>
        <span style={{ color: accent }} className="opacity-70">{icon}</span>
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className="stat-number text-[34px] leading-none" style={{ color: accent }}>{value}</span>
        {unit && <span className="text-xs tx-3">{unit}</span>}
      </div>
      {sub && <div className="text-[11px] tx-3 mt-1.5 font-mono">{sub}</div>}
    </div>
  );
}

export default function DashboardView({ onDiagnose }: { onDiagnose: (t: DiagnosisTarget) => void }) {
  const [metrics, setMetrics] = useState<CoreMetrics | null>(null);
  const [channels, setChannels] = useState<ChannelInfo[]>([]);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [sevFilter, setSevFilter] = useState<Severity | 'all'>('all');

  const fetchAll = useCallback(async (silent = false) => {
    if (!silent) setRefreshing(true);
    setError(null);
    try {
      const [m, c, a] = await Promise.all([
        api.dashboard.metrics(),
        api.dashboard.channels(),
        api.dashboard.alerts(),
      ]);
      setMetrics(m);
      setChannels(c.channels || []);
      setAlerts(a.alerts || []);
      setLastUpdated(new Date());
    } catch {
      setError('无法连接后端服务 (localhost:8000),请确认已启动 uvicorn src.api.main:app');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
    const timer = setInterval(() => fetchAll(true), REFRESH_INTERVAL);
    return () => clearInterval(timer);
  }, [fetchAll]);

  const criticalCount = alerts.filter(a => a.severity === 'critical').length;
  const filteredAlerts = sevFilter === 'all' ? alerts : alerts.filter(a => a.severity === sevFilter);
  const rateColor = !metrics ? '#38BDF8' : metrics.anomaly_rate > 0.4 ? '#EF4444' : metrics.anomaly_rate > 0.15 ? '#F59E0B' : '#22C55E';

  return (
    <div className="space-y-5">
      {/* 视图标题 */}
      <div className="flex items-center justify-between rise">
        <div>
          <h2 className="font-display text-xl font-bold tracking-wider tx-1">
            实时告警中心 <span className="tx-3 text-sm font-medium ml-1">REAL-TIME ALERT CENTER</span>
          </h2>
        </div>
        <div className="flex items-center gap-3">
          {lastUpdated && (
            <span className="text-[11px] tx-3 font-mono flex items-center gap-1">
              <Clock size={11} />
              {lastUpdated.toLocaleTimeString('zh-CN')}
            </span>
          )}
          <button onClick={() => fetchAll()} disabled={refreshing} className="btn-ghost">
            <RefreshCw size={12} className={refreshing ? 'animate-spin' : ''} />
            {refreshing ? '同步中' : '刷新'}
          </button>
        </div>
      </div>

      {/* KPI 行 */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
        <div className="panel corner-framed p-4 rise rise-1 col-span-2 md:col-span-1 flex items-center gap-4">
          <RateGauge rate={metrics?.anomaly_rate ?? 0} color={rateColor} />
          <div>
            <div className="text-[11px] uppercase tracking-[0.15em] tx-3 font-display font-semibold mb-1">系统异常率</div>
            <div className="stat-number text-2xl" style={{ color: rateColor }}>{metrics?.anomaly_rate_str ?? '—'}</div>
            <div className="text-[11px] tx-3 mt-1 font-mono">
              {metrics ? `${metrics.anomaly_rows.toLocaleString()} / ${metrics.total_rows.toLocaleString()} 行` : '加载中…'}
            </div>
          </div>
        </div>
        <KpiCard delay="rise-2" icon={<ShieldAlert size={16} />} label="告警队列" accent="#F97316"
          value={String(metrics?.fault_count ?? '—')} unit="条"
          sub={criticalCount > 0 ? `${criticalCount} 条 CRITICAL 待处置` : `共 ${metrics?.total_anomalies ?? 0} 条异常段`} />
        <KpiCard delay="rise-2" icon={<BrainCircuit size={16} />} label="最佳诊断 F1" accent="#2DD4BF"
          value={metrics ? metrics.best_f1.toFixed(4) : '—'}
          sub="Stage 2 · IF + 规则融合" />
        <KpiCard delay="rise-3" icon={<Database size={16} />} label="数据吞吐量" accent="#38BDF8"
          value={metrics ? `${(metrics.throughput / 1000).toFixed(0)}K` : '—'} unit="行"
          sub="18 天在轨遥测" />
        <KpiCard delay="rise-4" icon={<Radio size={16} />} label="监测通道" accent="#A78BFA"
          value={String(metrics?.channel_count ?? '—')} unit="路"
          sub="3× 磁力计 + 6× 光电二极管" />
      </div>

      {/* 通道矩阵 */}
      <div className="rise rise-2">
        <div className="flex items-center gap-2 mb-3">
          <Activity size={14} className="text-[#38BDF8]" />
          <span className="font-display font-semibold tracking-wider text-sm tx-2">遥测通道矩阵</span>
          <span className="text-[10px] font-mono tx-3 uppercase tracking-widest">Channel Matrix · 500pt window</span>
        </div>
        {error ? (
          <div className="panel p-6 text-center text-sm" style={{ borderColor: 'rgba(245,158,11,0.4)' }}>
            <span className="text-[#F59E0B]">{error}</span>
          </div>
        ) : channels.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {channels.map(ch => (
              <ChannelCard key={ch.channel} data={ch} onClick={() => onDiagnose({ channelId: ch.channel, segment: '', channelLabel: ch.label })} />
            ))}
          </div>
        ) : (
          <div className="panel p-10 text-center text-sm tx-3">{loading ? '正在接入遥测链路…' : '暂无通道数据'}</div>
        )}
      </div>

      {/* 告警队列 */}
      <div className="panel rise rise-3 overflow-hidden">
        <div className="panel-head justify-between">
          <span>故障告警队列 · FAULT QUEUE</span>
          <div className="flex items-center gap-1.5 normal-case tracking-normal">
            {(['all', 'critical', 'warning', 'caution'] as const).map(f => (
              <button
                key={f}
                onClick={() => setSevFilter(f)}
                className="text-[10px] font-mono px-2 py-0.5 rounded border transition-all"
                style={sevFilter === f
                  ? { color: '#38BDF8', borderColor: 'rgba(56,189,248,0.5)', background: 'rgba(56,189,248,0.1)' }
                  : { color: 'var(--tx-3)', borderColor: 'transparent' }}
              >
                {f === 'all' ? `全部 ${alerts.length}` : `${SEVERITY_LABELS[f]} ${alerts.filter(a => a.severity === f).length}`}
              </button>
            ))}
          </div>
        </div>
        {filteredAlerts.length === 0 ? (
          <div className="p-8 text-center text-sm tx-3">{loading ? '正在加载…' : '当前过滤条件下无告警'}</div>
        ) : (
          <div className="divide-y divide-[rgba(148,163,184,0.07)] max-h-[520px] overflow-y-auto">
            {filteredAlerts.map((a, i) => {
              const color = sevColor(a.severity);
              return (
                <div
                  key={`${a.segment}-${a.channel}-${i}`}
                  className="alert-row px-4 py-3 flex flex-col md:flex-row md:items-center gap-2 md:gap-4"
                  style={{ ['--row-accent' as string]: color }}
                  onClick={() => onDiagnose({ channelId: a.channel, segment: a.segment, score: a.anomaly_score, channelLabel: a.channel_label })}
                >
                  <div className="flex items-center gap-2.5 md:w-64 shrink-0">
                    <span className={sevDotClass(a.severity)} />
                    <span className="font-mono text-[13px] tx-1">SEG #{a.segment}</span>
                    <span className="font-mono text-[11px] text-[#38BDF8]">{a.channel}</span>
                    <span className="text-[11px] tx-3">{a.channel_label}</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs tx-2 truncate">{a.summary}</p>
                  </div>
                  <div className="flex items-center gap-3 shrink-0">
                    {a.anomaly_type && a.anomaly_type !== '未知' && (
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[rgba(148,163,184,0.08)] tx-3 border border-[rgba(148,163,184,0.15)]">
                        {a.anomaly_type}
                      </span>
                    )}
                    <div className="w-24">
                      <div className="flex justify-between text-[10px] font-mono mb-1">
                        <span className="tx-3">SCORE</span>
                        <span style={{ color }}>{a.anomaly_score.toFixed(3)}</span>
                      </div>
                      <div className="h-1 rounded-full bg-[rgba(148,163,184,0.12)] overflow-hidden">
                        <div className="h-full rounded-full" style={{ width: `${scorePct(a.anomaly_score)}%`, background: color, boxShadow: `0 0 6px ${color}` }} />
                      </div>
                    </div>
                    <span className="btn-ghost !py-1 !px-2 text-[11px]">
                      深度诊断 <ChevronRight size={12} />
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

    </div>
  );
}
