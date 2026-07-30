import { useEffect, useState } from 'react';
import { ArrowLeft, Crosshair, FileText, Search, Wrench, CheckCircle2, Link2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { api } from '../api/client';
import type { DiagnosisTarget, ExplanationDetail, WaveformData } from '../api/types';
import Oscilloscope from '../components/Oscilloscope';
import RagChat from '../components/RagChat';
import { sevColor } from '../components/severity';

const URGENCY_COLORS: Record<string, string> = {
  low: '#22C55E',
  medium: '#F59E0B',
  high: '#F97316',
  critical: '#EF4444',
};

const SECTION_META = [
  { key: '可能原因' as const, icon: <Search size={13} />, color: '#F59E0B' },
  { key: '影响评估' as const, icon: <Crosshair size={13} />, color: '#F97316' },
  { key: '建议措施' as const, icon: <Wrench size={13} />, color: '#38BDF8' },
  { key: '结论' as const, icon: <CheckCircle2 size={13} />, color: '#2DD4BF' },
];

export default function ExplanationView({ target, onBack }: { target: DiagnosisTarget; onBack: () => void }) {
  const { channelId, segment } = target;
  const [wave, setWave] = useState<WaveformData | null>(null);
  const [detail, setDetail] = useState<ExplanationDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setWave(null);
    setDetail(null);

    const load = async () => {
      try {
        const waveP = api.explanation.waveform(channelId, segment || undefined).catch(() => null);
        const detailP = segment
          ? api.explanation.detail(segment, channelId).catch(() => null)
          : Promise.resolve(null);
        const [w, d] = await Promise.all([waveP, detailP]);
        if (w && !w.error) setWave(w);
        if (d && !d.error) setDetail(d);
        if ((!w || w.error) && (!d || d.error)) {
          setError('未获取到诊断数据 — 请确认后端已启动且该异常段存在');
        }
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [channelId, segment]);

  const channelLabel = detail?.channel_label || target.channelLabel || channelId;
  const score = detail?.anomaly_score ?? target.score ?? 0;
  const urgencyColor = URGENCY_COLORS[detail?.urgency_level || 'medium'];
  const sections = detail && !detail.garbled ? detail.sections : {};
  const hasSections = SECTION_META.some(s => sections[s.key]);

  return (
    <div className="space-y-5">
      {/* 顶栏 */}
      <div className="flex flex-wrap items-center gap-3 rise">
        <button onClick={onBack} className="btn-ghost">
          <ArrowLeft size={13} /> 返回
        </button>
        <h2 className="font-display text-xl font-bold tracking-wider tx-1">
          深度诊断 <span className="tx-3 text-sm font-medium ml-1">DEEP DIAGNOSIS</span>
        </h2>
        <div className="flex items-center gap-2 ml-auto">
          {segment && (
            <span className="font-mono text-xs px-2 py-1 rounded bg-[rgba(56,189,248,0.08)] border border-[rgba(56,189,248,0.25)] text-[#38BDF8]">
              SEG #{segment}
            </span>
          )}
          <span className="font-mono text-xs px-2 py-1 rounded bg-[rgba(148,163,184,0.08)] border border-[rgba(148,163,184,0.2)] tx-2">
            {channelId} · {channelLabel}
          </span>
          {detail && (
            <span
              className="text-xs font-semibold px-2 py-1 rounded border"
              style={{ color: urgencyColor, borderColor: `${urgencyColor}55`, background: `${urgencyColor}14` }}
            >
              紧急度 · {detail.urgency}
            </span>
          )}
        </div>
      </div>

      {error && (
        <div className="panel p-4 text-sm text-[#F59E0B] rise" style={{ borderColor: 'rgba(245,158,11,0.4)' }}>{error}</div>
      )}

      {/* 示波器 */}
      <div className="panel corner-framed rise rise-1 overflow-hidden">
        <div className="panel-head justify-between">
          <span><Crosshair size={13} className="inline -mt-0.5" /> 遥测示波器 · ±3σ 限界</span>
          <span className="flex gap-4 normal-case tracking-normal font-mono text-[10px] tx-3">
            <span><i className="inline-block w-2 h-2 rounded-full bg-[#38BDF8] mr-1" />遥测值</span>
            <span><i className="inline-block w-2 h-2 rounded-full bg-[#EF4444] mr-1" />异常点</span>
            <span><i className="inline-block w-3 h-0 border-t border-dashed border-[#F59E0B] mr-1 align-middle" />±3σ</span>
            <span><i className="inline-block w-3 h-0 border-t border-dotted border-[#2DD4BF] mr-1 align-middle" />均值</span>
          </span>
        </div>
        <div className="h-80 p-2 relative">
          <div className="sweep-line" />
          {wave ? (
            <Oscilloscope wave={wave} />
          ) : (
            <div className="h-full flex items-center justify-center text-sm tx-3">
              {loading ? '正在载入波形…' : '波形数据不可用'}
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        {/* AI 结构化诊断 */}
        <div className="xl:col-span-2 space-y-4">
          <div className="panel rise rise-2">
            <div className="panel-head justify-between">
              <span><FileText size={13} className="inline -mt-0.5" /> AI 结构化诊断</span>
              {detail && !detail.garbled && (
                <span className="text-[10px] font-mono text-[#2DD4BF] border border-[rgba(45,212,191,0.3)] bg-[rgba(45,212,191,0.08)] rounded px-1.5 py-0.5">
                  RAG GENERATED
                </span>
              )}
            </div>
            <div className="p-4">
              {/* 摘要行 */}
              <div className="grid grid-cols-3 gap-3 mb-4">
                {[
                  ['异常类型', detail?.anomaly_type && detail.anomaly_type !== '未知' ? detail.anomaly_type : '—'],
                  ['异常分', score ? score.toFixed(3) : '—'],
                  ['紧急度', detail?.urgency ?? '—'],
                ].map(([k, v], i) => (
                  <div key={k} className="rounded-lg border border-[rgba(148,163,184,0.1)] bg-[rgba(4,7,13,0.5)] p-3">
                    <div className="text-[10px] tx-3 uppercase tracking-wider mb-1">{k}</div>
                    <div
                      className="font-mono text-sm font-semibold"
                      style={{ color: i === 1 && score ? sevColor(score > 0.4 ? 'critical' : score > 0.15 ? 'warning' : 'caution') : 'var(--tx-1)' }}
                    >
                      {v}
                    </div>
                  </div>
                ))}
              </div>

              {detail?.garbled ? (
                <div className="text-sm text-[#F59E0B] bg-[rgba(245,158,11,0.08)] border border-[rgba(245,158,11,0.25)] rounded-lg p-4">
                  诊断文本异常(乱码),解释不可用 — 请重新生成 RAG 结果或检查知识库。
                </div>
              ) : hasSections ? (
                <div className="space-y-4">
                  {SECTION_META.map(({ key, icon, color }) =>
                    sections[key] ? (
                      <div key={key} className="rounded-lg border border-[rgba(148,163,184,0.09)] overflow-hidden">
                        <div
                          className="flex items-center gap-2 px-3 py-2 text-xs font-semibold font-display tracking-wider"
                          style={{ color, background: `${color}0D`, borderBottom: `1px solid ${color}22` }}
                        >
                          {icon} {key}
                        </div>
                        <div className="px-3 py-2 md-body">
                          <ReactMarkdown>{sections[key]!}</ReactMarkdown>
                        </div>
                      </div>
                    ) : null,
                  )}
                </div>
              ) : detail?.raw_explanation ? (
                <div className="md-body"><ReactMarkdown>{detail.raw_explanation}</ReactMarkdown></div>
              ) : (
                <div className="text-sm tx-3 py-6 text-center">
                  {loading ? '正在生成诊断…' : '暂无结构化诊断 — 从告警队列选择具体异常段查看'}
                </div>
              )}
            </div>
          </div>

          {/* RAG 问答 */}
          <div className="rise rise-3">
            <RagChat channel={channelId} segment={segment} />
          </div>
        </div>

        {/* 知识溯源 */}
        <div className="panel rise rise-3 self-start">
          <div className="panel-head"><Link2 size={13} /> 知识溯源 · SOURCES</div>
          <div className="p-3 space-y-2">
            {detail?.sources && detail.sources.length > 0 ? (
              detail.sources.map((s, i) => (
                <div
                  key={i}
                  className="p-3 rounded-lg border border-[rgba(148,163,184,0.1)] bg-[rgba(4,7,13,0.5)] hover:border-[rgba(45,212,191,0.4)] transition-colors"
                  title={s.local_path || ''}
                >
                  <div className="text-xs tx-1 font-medium break-all leading-snug">{s.filename}</div>
                  <div className="flex justify-between items-center mt-2">
                    <span className="text-[10px] font-mono tx-3">{s.page ? `P.${s.page}` : '—'}</span>
                    <div className="flex items-center gap-1.5">
                      <div className="w-14 h-1 rounded-full bg-[rgba(148,163,184,0.12)] overflow-hidden">
                        <div
                          className="h-full rounded-full bg-[#2DD4BF]"
                          style={{ width: `${Math.min(100, (s.score ?? 0) * 100)}%` }}
                        />
                      </div>
                      <span className="text-[10px] font-mono text-[#2DD4BF]">{(s.score ?? 0).toFixed(3)}</span>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-xs tx-3 text-center py-8">
                {loading ? '加载中…' : '暂无溯源文献'}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
