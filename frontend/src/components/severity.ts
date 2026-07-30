import type { Severity } from '../api/types';

export const SEVERITY_COLORS: Record<Severity, string> = {
  nominal: '#22C55E',
  caution: '#F97316',
  warning: '#F59E0B',
  critical: '#EF4444',
  offline: '#64748B',
};

export const SEVERITY_LABELS: Record<Severity, string> = {
  nominal: '正常',
  caution: '注意',
  warning: '警告',
  critical: '严重',
  offline: '离线',
};

export function sevColor(sev: string | undefined): string {
  return SEVERITY_COLORS[(sev as Severity) || 'offline'] ?? SEVERITY_COLORS.offline;
}

export function sevDotClass(sev: string | undefined): string {
  const s = (sev as Severity) || 'offline';
  return `dot dot-${s in SEVERITY_COLORS ? s : 'offline'}`;
}

/** 异常分 → 0~100 的百分比宽度 */
export function scorePct(score: number): number {
  return Math.min(100, Math.max(0, score * 100));
}
