import { useCallback, useEffect, useState } from 'react';
import { Activity, FlaskConical, Crosshair, Satellite } from 'lucide-react';
import { api } from './api/client';
import type { AlertItem, DiagnosisTarget, SystemStatus } from './api/types';
import BootSequence from './components/BootSequence';
import { sevColor } from './components/severity';
import DashboardView from './views/DashboardView';
import DetectionView from './views/DetectionView';
import ExplanationView from './views/ExplanationView';

type Page = 'dashboard' | 'detection' | 'explanation';

function useUtcClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${now.getUTCFullYear()}-${pad(now.getUTCMonth() + 1)}-${pad(now.getUTCDate())} ${pad(now.getUTCHours())}:${pad(now.getUTCMinutes())}:${pad(now.getUTCSeconds())}`;
}

const NAV: { key: Page; label: string; en: string; icon: React.ReactNode; kbd: string }[] = [
  { key: 'dashboard', label: '告警中心', en: 'ALERTS', icon: <Activity size={13} />, kbd: '1' },
  { key: 'detection', label: '算法实验', en: 'BENCH', icon: <FlaskConical size={13} />, kbd: '2' },
  { key: 'explanation', label: '深度诊断', en: 'DIAG', icon: <Crosshair size={13} />, kbd: '3' },
];

export default function App() {
  const [page, setPage] = useState<Page>('dashboard');
  const [target, setTarget] = useState<DiagnosisTarget | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [tickerAlerts, setTickerAlerts] = useState<AlertItem[]>([]);
  const [booted, setBooted] = useState(false);
  const clock = useUtcClock();

  useEffect(() => {
    const poll = () => {
      api.dashboard.status().then(setStatus).catch(() => setStatus(null));
      api.dashboard.alerts().then(a => setTickerAlerts(a.alerts || [])).catch(() => {});
    };
    poll();
    const t = setInterval(poll, 30_000);
    return () => clearInterval(t);
  }, []);

  const goDiagnose = useCallback((t: DiagnosisTarget) => {
    setTarget(t);
    setPage('explanation');
  }, []);

  // 键盘快捷键 1/2/3
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.key === '1') setPage('dashboard');
      if (e.key === '2') setPage('detection');
      if (e.key === '3' && target) setPage('explanation');
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [target]);

  const apiOk = status !== null;

  return (
    <div className="h-full flex flex-col select-none overflow-hidden">
      <div className="void-backdrop" />
      {!booted && <BootSequence onDone={() => setBooted(true)} />}

      {/* ===== 顶部控制台 ===== */}
      <header className="h-[52px] shrink-0 flex items-center gap-4 px-4 border-b border-[rgba(94,234,212,0.14)] bg-[rgba(3,6,12,0.85)] backdrop-blur-md relative z-20">
        <div className="flex items-center gap-2.5">
          <Satellite size={17} className="text-[#22d3ee]" style={{ filter: 'drop-shadow(0 0 6px rgba(34,211,238,0.6))' }} />
          <div className="leading-none">
            <div className="f-brand text-[13px] font-extrabold tracking-[0.14em] phosphor">OPS-SAT</div>
            <div className="text-[8px] font-mono tx-3 tracking-[0.32em] mt-0.5">TELEMETRY ANOMALY DECK</div>
          </div>
        </div>

        {/* 页签 */}
        <nav className="flex items-center gap-1.5 mx-auto">
          {NAV.map(n => (
            <button key={n.key}
              onClick={() => (n.key !== 'explanation' || target) && setPage(n.key)}
              className={`console-tab ${page === n.key ? 'on' : ''}`}
              style={n.key === 'explanation' && !target ? { opacity: 0.35, cursor: 'not-allowed' } : undefined}>
              {n.icon}
              <span>{n.label}</span>
              <span className="hidden lg:inline text-[9px] opacity-60 font-mono">{n.en}</span>
              <span className="kbd">{n.kbd}</span>
            </button>
          ))}
        </nav>

        {/* 状态灯 + 时钟 */}
        <div className="flex items-center gap-3.5">
          {([
            ['TLM', status?.segments_available],
            ['RAG', status?.rag_available],
            ['API', apiOk],
          ] as const).map(([k, ok]) => (
            <div key={k} className="flex items-center gap-1.5 font-mono text-[9px]">
              <span className={`lamp ${ok ? 'lamp-nominal' : k === 'API' ? 'lamp-critical' : 'lamp-offline'}`} />
              <span className="tx-3">{k}</span>
            </div>
          ))}
          <span className="w-px h-4 bg-[rgba(94,234,212,0.2)]" />
          <span className="font-mono text-[11px] text-[#22d3ee] tracking-widest">{clock} <span className="tx-3 text-[9px]">UTC</span></span>
        </div>
      </header>

      {/* ===== 主区 ===== */}
      <main className="flex-1 min-h-0 p-3 relative z-10">
        {page === 'dashboard' && <DashboardView onDiagnose={goDiagnose} />}
        {page === 'detection' && <DetectionView />}
        {page === 'explanation' &&
          (target ? (
            <ExplanationView target={target} onBack={() => setPage('dashboard')} />
          ) : (
            <div className="h-full flex items-center justify-center font-mono text-[11px] tx-3">
              请先在告警中心选择一条告警进行深度诊断
            </div>
          ))}
      </main>

      {/* ===== 底部 ticker ===== */}
      <footer className="h-7 shrink-0 flex items-center border-t border-[rgba(94,234,212,0.14)] bg-[rgba(3,6,12,0.9)] overflow-hidden relative z-20">
        <div className="shrink-0 px-3 h-full flex items-center border-r border-[rgba(94,234,212,0.14)] font-mono text-[9px] tracking-[0.2em] text-[#22d3ee]">
          LIVE FEED
        </div>
        <div className="flex-1 overflow-hidden">
          {tickerAlerts.length > 0 ? (
            <div className="ticker-track">
              {[0, 1].map(dup => (
                <span key={dup} className="inline-flex">
                  {tickerAlerts.slice(0, 30).map((a, i) => (
                    <span key={`${dup}-${i}`} className="inline-flex items-center font-mono text-[10px] tx-3 mx-5">
                      <span className="w-1.5 h-1.5 rounded-full mr-2" style={{ background: sevColor(a.severity) }} />
                      SEG #{a.segment} · {a.channel} · {a.anomaly_score.toFixed(3)} —
                      <span className="tx-2 ml-1">{a.summary.slice(0, 40)}…</span>
                    </span>
                  ))}
                </span>
              ))}
            </div>
          ) : (
            <span className="font-mono text-[10px] tx-3 px-4">AWAITING TELEMETRY…</span>
          )}
        </div>
        <div className="shrink-0 px-3 h-full hidden md:flex items-center border-l border-[rgba(94,234,212,0.14)] font-mono text-[9px] tx-3">
          IF × RAG · F1 0.6281
        </div>
      </footer>

      <div className="crt-overlay" />
    </div>
  );
}
