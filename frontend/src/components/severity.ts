import type { Severity } from '../api/types';

export const SEVERITY_COLORS: Record<Severity, string> = {
  nominal: '#2DD4A7',
  caution: '#FB923C',
  warning: '#FBBF24',
  critical: '#F43F5E',
  offline: '#5B6B7F',
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

export function sevLampClass(sev: string | undefined): string {
  const s = (sev as Severity) || 'offline';
  return `lamp ${s in SEVERITY_COLORS ? `lamp-${s}` : 'lamp-offline'}`;
}

export function scorePct(score: number): number {
  return Math.min(100, Math.max(0, score * 100));
}
