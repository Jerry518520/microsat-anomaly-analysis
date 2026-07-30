import { useCallback, useEffect, useState } from 'react';
import { Activity, BarChart3, Crosshair, Satellite } from 'lucide-react';
import { api } from './api/client';
import type { DiagnosisTarget, SystemStatus } from './api/types';
import DashboardView from './views/DashboardView';
import DetectionView from './views/DetectionView';
import ExplanationView from './views/ExplanationView';

type Page = 'dashboard' | 'detection' | 'explanation';

/** UTC 时钟 */
function useUtcClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${now.getUTCFullYear()}-${pad(now.getUTCMonth() + 1)}-${pad(now.getUTCDate())} ${pad(now.getUTCHours())}:${pad(now.getUTCMinutes())}:${pad(now.getUTCSeconds())} UTC`;
}

const NAV: { key: Page; label: string; icon: React.ReactNode }[] = [
  { key: 'dashboard', label: '告警中心', icon: <Activity size={19} /> },
  { key: 'detection', label: '算法实验', icon: <BarChart3 size={19} /> },
  { key: 'explanation', label: '深度诊断', icon: <Crosshair size={19} /> },
];

export default function App() {
  const [page, setPage] = useState<Page>('dashboard');
  const [target, setTarget] = useState<DiagnosisTarget | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const clock = useUtcClock();

  const pollStatus = useCallback(() => {
    api.dashboard.status().then(setStatus).catch(() => setStatus(null));
  }, []);

  useEffect(() => {
    pollStatus();
    const t = setInterval(pollStatus, 30_000);
    return () => clearInterval(t);
  }, [pollStatus]);

  const goDiagnose = useCallback((t: DiagnosisTarget) => {
    setTarget(t);
    setPage('explanation');
  }, []);

  const backendOk = status !== null;

  return (
    <div className="h-full flex flex-col select-none">
      <div className="space-backdrop" />

      {/* ===== 顶栏 ===== */}
      <header className="h-14 shrink-0 flex items-center gap-4 px-4 border-b border-[rgba(148,163,184,0.14)] bg-[rgba(7,11,20,0.85)] backdrop-blur-md relative z-10">
        <div className="flex items-center gap-3">
          <div
            className="w-9 h-9 rounded-lg flex items-center justify-center"
            style={{ background: 'rgba(56,189,248,0.1)', border: '1px solid rgba(56,189,248,0.35)', boxShadow: '0 0 16px rgba(56,189,248,0.15)' }}
          >
            <Satellite size={18} className="text-[#38BDF8]" />
          </div>
          <div>
            <div className="font-display font-bold tracking-[0.18em] text-[15px] leading-tight">
              OPS-SAT <span className="text-[#38BDF8]">遥测异常诊断系统</span>
            </div>
            <div className="text-[9px] font-mono tx-3 tracking-[0.3em] uppercase">Telemetry Anomaly Diagnosis · v2.0</div>
          </div>
        </div>

        <div className="ml-auto flex items-center gap-4">
          <div className="hidden md:flex items-center gap-1.5 text-[11px] font-mono">
            <span className={`dot ${status?.segments_available ? 'dot-nominal' : 'dot-offline'}`} />
            <span className="tx-3">遥测链路</span>
          </div>
          <div className="hidden md:flex items-center gap-1.5 text-[11px] font-mono">
            <span className={`dot ${status?.rag_available ? 'dot-nominal' : 'dot-offline'}`} />
            <span className="tx-3">RAG 引擎</span>
          </div>
          <div className="hidden md:flex items-center gap-1.5 text-[11px] font-mono">
            <span className={`dot ${backendOk ? 'dot-nominal' : 'dot-critical'}`} />
            <span className="tx-3">{backendOk ? 'API 在线' : 'API 断开'}</span>
          </div>
          <span className="w-px h-5 bg-[rgba(148,163,184,0.2)]" />
          <span className="font-mono text-xs text-[#38BDF8] tracking-wider">{clock}</span>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* ===== 侧边导航 ===== */}
        <aside className="w-[72px] shrink-0 border-r border-[rgba(148,163,184,0.14)] bg-[rgba(7,11,20,0.6)] flex flex-col items-center py-4 gap-1 z-10">
          {NAV.map(n => (
            <button
              key={n.key}
              onClick={() => setPage(n.key)}
              className={`nav-btn ${page === n.key ? 'active' : ''}`}
              disabled={n.key === 'explanation' && !target}
              style={n.key === 'explanation' && !target ? { opacity: 0.35, cursor: 'not-allowed' } : undefined}
            >
              {n.icon}
              <span>{n.label}</span>
            </button>
          ))}
        </aside>

        {/* ===== 内容区 ===== */}
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-[1680px] mx-auto p-5 md:p-6">
            {page === 'dashboard' && <DashboardView onDiagnose={goDiagnose} />}
            {page === 'detection' && <DetectionView />}
            {page === 'explanation' &&
              (target ? (
                <ExplanationView target={target} onBack={() => setPage('dashboard')} />
              ) : (
                <div className="panel p-10 text-center text-sm tx-3">请先在告警中心选择一条告警进行深度诊断</div>
              ))}
          </div>
        </main>
      </div>

      {/* ===== 底栏 ===== */}
      <footer className="h-7 shrink-0 flex items-center justify-between px-4 text-[10px] font-mono tx-3 border-t border-[rgba(148,163,184,0.14)] bg-[rgba(7,11,20,0.85)]">
        <span>ESA OPS-SAT MISSION CONTROL · 18-DAY TELEMETRY</span>
        <span>ISOLATION FOREST × RAG · F1 0.5683</span>
      </footer>
    </div>
  );
}
