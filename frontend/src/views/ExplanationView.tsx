import { useEffect, useState } from 'react';
import { ArrowLeft, Crosshair, Search, Wrench, CheckCircle2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { api } from '../api/client';
import type { DiagnosisTarget, ExplanationDetail, WaveformData } from '../api/types';
import Oscilloscope from '../components/Oscilloscope';
import RagChat from '../components/RagChat';
import { sevColor } from '../components/severity';

const URGENCY_COLORS: Record<string, string> = {
  low: '#2dd4a7',
  medium: '#fbbf24',
  high: '#fb923c',
  critical: '#f43f5e',
};

const SECTION_META = [
  { key: '可能原因' as const, icon: <Search size={12} />, color: '#fbbf24' },
  { key: '影响评估' as const, icon: <Crosshair size={12} />, color: '#fb923c' },
  { key: '建议措施' as const, icon: <Wrench size={12} />, color: '#22d3ee' },
  { key: '结论' as const, icon: <CheckCircle2 size={12} />, color: '#2dd4a7' },
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

    (async () => {
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
    })();
  }, [channelId, segment]);

  const channelLabel = detail?.channel_label || target.channelLabel || channelId;
  const score = detail?.anomaly_score ?? target.score ?? 0;
  const urgencyColor = URGENCY_COLORS[detail?.urgency_level || 'medium'];
  const sections = detail && !detail.garbled ? detail.sections : {};
  const hasSections = SECTION_META.some(s => sections[s.key]);

  return (
    <div className="h-full overflow-y-auto space-y-3 pr-1">
      {/* 指挥条 */}
      <div className="flex flex-wrap items-center gap-2.5 rise">
        <button onClick={onBack} className="ghost-btn"><ArrowLeft size={12} /> 返回</button>
        <h2 className="f-disp text-lg font-bold tracking-[0.2em] tx-1">
          深度诊断 <span className="tx-3 text-xs font-medium tracking-[0.3em] ml-1">DEEP DIAGNOSIS</span>
        </h2>
        <div className="flex items-center gap-2 ml-auto">
          {segment && (
            <span className="font-mono text-[10px] px-2 py-1 rounded border border-[rgba(34,211,238,0.35)] bg-[rgba(34,211,238,0.08)] text-[#22d3ee]">
              SEG #{segment}
            </span>
          )}
          <span className="font-mono text-[10px] px-2 py-1 rounded border border-[rgba(94,234,212,0.2)] bg-[rgba(8,16,32,0.5)] tx-2">
            {channelId} · {channelLabel}
          </span>
          {detail && (
            <span className="text-[10px] font-semibold px-2 py-1 rounded border"
              style={{ color: urgencyColor, borderColor: `${urgencyColor}55`, background: `${urgencyColor}12` }}>
              紧急度 · {detail.urgency}
            </span>
          )}
        </div>
      </div>

      {error && (
        <div className="deck-panel p-3 rise font-mono text-[11px] text-[#fbbf24]" style={{ borderColor: 'rgba(251,191,36,0.35)' }}>
          {error}
        </div>
      )}

      <div className="grid grid-cols-12 gap-3">
        {/* 左:示波器 + 问答 */}
        <div className="col-span-12 xl:col-span-8 space-y-3">
          <div className="deck-panel rise rise-1 overflow-hidden">
            <div className="deck-head justify-between">
              <span className="flex items-center gap-2"><span className="tick" /> 遥测示波器 · ±3σ 限界</span>
              <span className="flex gap-3 normal-case tracking-normal font-mono text-[9px] tx-3">
                <span><i className="inline-block w-1.5 h-1.5 rounded-full bg-[#22d3ee] mr-1" />遥测值</span>
                <span><i className="inline-block w-1.5 h-1.5 rounded-full bg-[#f43f5e] mr-1" />异常点</span>
                <span><i className="inline-block w-3 h-0 border-t border-dashed border-[#fbbf24] mr-1 align-middle" />±3σ</span>
                <span><i className="inline-block w-3 h-0 border-t border-dotted border-[#2dd4a7] mr-1 align-middle" />均值</span>
              </span>
            </div>
            <div className="h-[340px] p-1">
              {wave ? (
                <Oscilloscope wave={wave} />
              ) : (
                <div className="h-full flex items-center justify-center font-mono text-[10px] tx-3">
                  {loading ? 'ACQUIRING WAVEFORM…' : 'WAVEFORM UNAVAILABLE'}
                </div>
              )}
            </div>
          </div>

          <div className="rise rise-2">
            <RagChat channel={channelId} segment={segment} />
          </div>
        </div>

        {/* 右:诊断 + 溯源 */}
        <div className="col-span-12 xl:col-span-4 space-y-3">
          {/* 摘要仪表 */}
          <div className="deck-panel rise rise-2 p-3.5 grid grid-cols-3 gap-2">
            {([
              ['异常类型', detail?.anomaly_type && detail.anomaly_type !== '未知' ? detail.anomaly_type : '—', undefined],
              ['异常分', score ? score.toFixed(3) : '—', score ? sevColor(score > 0.4 ? 'critical' : score > 0.15 ? 'warning' : 'caution') : undefined],
              ['紧急度', detail?.urgency ?? '—', detail ? urgencyColor : undefined],
            ] as const).map(([k, v, c]) => (
              <div key={k} className="rounded border border-[rgba(94,234,212,0.1)] bg-[rgba(2,4,10,0.5)] p-2.5 text-center">
                <div className="text-[9px] tx-3 uppercase tracking-widest mb-1 font-mono">{k}</div>
                <div className="font-mono text-[12px] font-semibold" style={{ color: c ?? 'var(--tx-1)' }}>{v}</div>
              </div>
            ))}
          </div>

          {/* AI 诊断 */}
          <div className="deck-panel rise rise-3">
            <div className="deck-head justify-between">
              <span className="flex items-center gap-2"><span className="tick" /> AI 结构化诊断</span>
              {detail && !detail.garbled && (
                <span className="text-[9px] font-mono text-[#2dd4a7] border border-[rgba(45,212,167,0.3)] bg-[rgba(45,212,167,0.07)] rounded px-1.5 py-0.5">
                  RAG
                </span>
              )}
            </div>
            <div className="p-3 max-h-[420px] overflow-y-auto">
              {detail?.garbled ? (
                <div className="text-[11px] text-[#fbbf24] bg-[rgba(251,191,36,0.07)] border border-[rgba(251,191,36,0.25)] rounded p-3 font-mono">
                  诊断文本异常(乱码),解释不可用
                </div>
              ) : hasSections ? (
                <div className="space-y-2.5">
                  {SECTION_META.map(({ key, icon, color }) =>
                    sections[key] ? (
                      <div key={key} className="rounded border overflow-hidden" style={{ borderColor: 'rgba(94,234,212,0.1)' }}>
                        <div className="flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-semibold f-disp tracking-[0.18em]"
                          style={{ color, background: `${color}0C`, borderBottom: `1px solid ${color}22` }}>
                          {icon} {key}
                        </div>
                        <div className="px-2.5 py-2 md-body text-[12px]">
                          <ReactMarkdown>{sections[key]!}</ReactMarkdown>
                        </div>
                      </div>
                    ) : null,
                  )}
                </div>
              ) : detail?.raw_explanation ? (
                <div className="md-body"><ReactMarkdown>{detail.raw_explanation}</ReactMarkdown></div>
              ) : (
                <div className="text-[11px] font-mono tx-3 py-8 text-center">
                  {loading ? 'GENERATING DIAGNOSIS…' : '暂无结构化诊断 — 从告警队列选择异常段'}
                </div>
              )}
            </div>
          </div>

          {/* 知识溯源 */}
          <div className="deck-panel rise rise-4">
            <div className="deck-head"><span className="tick" /> 知识溯源 · SOURCES</div>
            <div className="p-2.5 space-y-2 max-h-[260px] overflow-y-auto">
              {detail?.sources && detail.sources.length > 0 ? (
                detail.sources.map((s, i) => (
                  <div key={i}
                    className="p-2.5 rounded border border-[rgba(94,234,212,0.1)] bg-[rgba(2,4,10,0.5)] hover:border-[rgba(45,212,167,0.4)] transition-colors"
                    title={s.local_path || ''}>
                    <div className="text-[11px] tx-1 break-all leading-snug">{s.filename}</div>
                    <div className="flex justify-between items-center mt-1.5">
                      <span className="text-[9px] font-mono tx-3">{s.page ? `P.${s.page}` : '—'}</span>
                      <div className="flex items-center gap-1.5">
                        <div className="w-12 h-0.5 rounded bg-[rgba(94,234,212,0.12)] overflow-hidden">
                          <div className="h-full bg-[#2dd4a7]" style={{ width: `${Math.min(100, (s.score ?? 0) * 100)}%` }} />
                        </div>
                        <span className="text-[9px] font-mono text-[#2dd4a7]">{(s.score ?? 0).toFixed(3)}</span>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-[10px] font-mono tx-3 text-center py-6">
                  {loading ? 'LOADING…' : '暂无溯源文献'}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
