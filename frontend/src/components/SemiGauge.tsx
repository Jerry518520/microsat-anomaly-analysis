/** 半圆仪表(180° 弧 + 刻度 + 读数) */
export default function SemiGauge({ value, color, size = 150 }: { value: number; color: string; size?: number }) {
  const pct = Math.min(1, Math.max(0, value));
  const r = 60;
  const cx = 75, cy = 70;
  const arc = (frac: number) => {
    // 从 180° → 0°
    const a = Math.PI - frac * Math.PI;
    return [cx + Math.cos(a) * r, cy - Math.sin(a) * r] as const;
  };
  const [sx, sy] = arc(0);
  const [ex, ey] = arc(pct);
  const large = pct > 0.5 ? 1 : 0;

  const ticks = Array.from({ length: 11 }, (_, i) => {
    const [x1, y1] = arc(i / 10);
    const inner = i / 10;
    const a = Math.PI - inner * Math.PI;
    const x2 = cx + Math.cos(a) * (r - (i % 5 === 0 ? 9 : 5));
    const y2 = cy - Math.sin(a) * (r - (i % 5 === 0 ? 9 : 5));
    return { x1, y1, x2, y2, key: i };
  });

  return (
    <svg width={size} height={size * 0.62} viewBox="0 0 150 93">
      {ticks.map(t => (
        <line key={t.key} x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2} stroke="rgba(94,234,212,0.25)" strokeWidth="1" />
      ))}
      {/* 底弧 */}
      <path d={`M ${sx} ${sy} A ${r} ${r} 0 0 1 ${arc(1)[0]} ${arc(1)[1]}`}
        fill="none" stroke="rgba(94,234,212,0.12)" strokeWidth="7" strokeLinecap="round" />
      {/* 值弧 */}
      <path d={`M ${sx} ${sy} A ${r} ${r} 0 ${large} 1 ${ex} ${ey}`}
        fill="none" stroke={color} strokeWidth="7" strokeLinecap="round"
        style={{ filter: `drop-shadow(0 0 6px ${color})`, transition: 'all 0.9s cubic-bezier(0.22,1,0.36,1)' }} />
      <text x={cx} y={cy - 8} textAnchor="middle" fill={color} fontSize="24" fontWeight="800"
        fontFamily="Orbitron, sans-serif" style={{ filter: `drop-shadow(0 0 8px ${color}66)` }}>
        {(pct * 100).toFixed(1)}%
      </text>
      <text x={cx} y={cy + 10} textAnchor="middle" fill="#47616c" fontSize="8" letterSpacing="2"
        fontFamily="IBM Plex Mono, monospace">
        ANOMALY RATE
      </text>
    </svg>
  );
}
