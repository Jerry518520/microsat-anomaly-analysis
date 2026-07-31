import { useEffect, useState } from 'react';

const BOOT_LINES = [
  { text: 'OPS-SAT MISSION CONTROL DECK v2.0 — BOOT SEQUENCE', cls: '' },
  { text: '> establishing telemetry downlink .......... [OK]', cls: 'boot-ok' },
  { text: '> loading isolation-forest ensemble ........ [OK]', cls: 'boot-ok' },
  { text: '> syncing segment index · 303,493 rows ..... [OK]', cls: 'boot-ok' },
  { text: '> mounting RAG knowledge base .............. [STANDBY]', cls: 'boot-warn' },
  { text: '> calibrating 9 telemetry channels ......... [OK]', cls: 'boot-ok' },
  { text: 'ALL SYSTEMS NOMINAL — WELCOME, OPERATOR', cls: 'boot-ok' },
];

/** 开机自检序列 — 首次加载展示一次 */
export default function BootSequence({ onDone }: { onDone: () => void }) {
  const [shown, setShown] = useState(0);
  const [fading, setFading] = useState(false);

  useEffect(() => {
    if (shown < BOOT_LINES.length) {
      const t = setTimeout(() => setShown(s => s + 1), shown === 0 ? 250 : 170);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => setFading(true), 450);
    const t2 = setTimeout(onDone, 950);
    return () => { clearTimeout(t); clearTimeout(t2); };
  }, [shown, onDone]);

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center transition-opacity duration-500"
      style={{ background: 'var(--void)', opacity: fading ? 0 : 1, pointerEvents: fading ? 'none' : 'auto' }}
    >
      <div className="w-[520px] max-w-[90vw]">
        <div className="border border-[rgba(94,234,212,0.25)] rounded p-6 bg-[rgba(5,10,20,0.8)]">
          {BOOT_LINES.slice(0, shown).map((l, i) => (
            <div key={i} className={`boot-line ${l.cls}`}>{l.text}</div>
          ))}
          {shown < BOOT_LINES.length && <span className="boot-line animate-pulse">▋</span>}
        </div>
        <div className="mt-3 h-0.5 bg-[rgba(94,234,212,0.12)] rounded overflow-hidden">
          <div
            className="h-full bg-[#22d3ee] transition-all duration-200"
            style={{ width: `${(shown / BOOT_LINES.length) * 100}%`, boxShadow: '0 0 10px #22d3ee' }}
          />
        </div>
      </div>
    </div>
  );
}
