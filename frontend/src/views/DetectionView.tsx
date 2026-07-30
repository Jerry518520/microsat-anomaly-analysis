import { useEffect, useState } from 'react';
import { Cpu, GitBranch, BarChart3, BookOpen } from 'lucide-react';
import { api } from '../api/client';
import type { ChannelF1, ExperimentStage, RagConfig, SystemParams } from '../api/types';
import Plot from '../components/Plot';

const CHANNEL_LABELS: Record<string, string> = {
  CADC0872: '磁力计 X轴', CADC0873: '磁力计 Y轴', CADC0874: '磁力计 Z轴',
  CADC0884: '光电二极管 1', CADC0886: '光电二极管 2', CADC0888: '光电二极管 3',
  CADC0890: '光电二极管 4', CADC0892: '光电二极管 5', CADC0894: '光电二极管 6',
};

export default function DetectionView() {
  const [params, setParams] = useState<SystemParams | null>(null);
  const [experiments, setExperiments] = useState<ExperimentStage[]>([]);
  const [channelF1, setChannelF1] = useState<ChannelF1[]>([]);
  const [rag, setRag] = useState<RagConfig | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.detection.systemParams(),
      api.detection.experiments(),
      api.detection.channelF1(),
      api.detection.ragConfig(),
    ])
      .then(([p, e, c, r]) => {
        setParams(p);
        setExperiments(e.experiments || []);
        setChannelF1(c.channels || []);
        setRag(r);
      })
      .catch(() => setError('无法连接后端服务 (localhost:8000)'));
  }, []);

  if (error) {
    return (
      <div className="panel p-10 text-center text-sm text-[#F59E0B]" style={{ borderColor: 'rgba(245,158,11,0.4)' }}>{error}</div>
    );
  }

  const sortedF1 = [...channelF1].sort((a, b) => b.f1 - a.f1);

  return (
    <div className="space-y-5">
      <h2 className="font-display text-xl font-bold tracking-wider tx-1 rise">
        算法与实验 <span className="tx-3 text-sm font-medium ml-1">DETECTION BENCHMARK</span>
      </h2>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        {/* 系统参数 */}
        <div className="panel rise rise-1">
          <div className="panel-head"><Cpu size={13} /> 检测引擎参数</div>
          <div className="p-4 space-y-3">
            {[
              ['算法', params?.algorithm ?? '—'],
              ['特征维度', params ? `${params.dimensions} 维` : '—'],
              ['污染率 contamination', params ? String(params.contamination) : '—'],
            ].map(([k, v]) => (
              <div key={k} className="flex items-center justify-between py-2 border-b border-[rgba(148,163,184,0.07)] last:border-0">
                <span className="text-xs tx-3">{k}</span>
                <span className="font-mono text-sm tx-1">{v}</span>
              </div>
            ))}
            <div className="pt-2 text-[11px] tx-3 leading-relaxed">
              三级检测流水线:全局 Isolation Forest → 分通道独立 IForest → 规则融合裁决。
            </div>
          </div>
        </div>

        {/* 实验演进 F1 */}
        <div className="panel rise rise-2 xl:col-span-2">
          <div className="panel-head"><GitBranch size={13} /> 实验演进 · SegF1 Benchmark</div>
          <div className="p-2 h-64">
            {experiments.length > 0 && (
              <Plot
                data={[
                  {
                    x: experiments.map((_, i) => i),
                    y: experiments.map(e => e.f1),
                    type: 'scatter',
                    mode: 'lines+markers+text',
                    line: { color: '#38BDF8', width: 2.5, shape: 'spline', smoothing: 0.6 },
                    marker: { size: 10, color: '#0B1120', line: { color: '#38BDF8', width: 2.5 } },
                    text: experiments.map(e => `${e.f1.toFixed(4)}<br><span style="font-size:10px;color:#22C55E">${e.improvement}</span>`),
                    textposition: 'top center',
                    textfont: { family: 'Rajdhani, sans-serif', size: 13, color: '#E6EDF7' },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(56,189,248,0.07)',
                    customdata: experiments.map(e => [e.version, e.strategy]),
                    hovertemplate: '%{customdata[0]} · %{customdata[1]}<br>SegF1 = %{y:.4f}<extra></extra>',
                  },
                ]}
                layout={{
                  autosize: true,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  margin: { l: 44, r: 16, t: 30, b: 36 },
                  font: { family: 'IBM Plex Mono, monospace', color: '#5A6A84', size: 10 },
                  xaxis: {
                    gridcolor: 'rgba(148,163,184,0.06)',
                    linecolor: 'rgba(148,163,184,0.2)',
                    tickmode: 'array',
                    tickvals: experiments.map((_, i) => i),
                    ticktext: experiments.map(e => e.version),
                    range: [-0.45, experiments.length - 0.55],
                  },
                  yaxis: { gridcolor: 'rgba(148,163,184,0.08)', linecolor: 'rgba(148,163,184,0.2)', range: [0, 0.75], title: { text: 'SegF1', font: { size: 10 } } },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%', height: '100%' }}
              />
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        {/* 分通道 F1 */}
        <div className="panel rise rise-3 xl:col-span-2">
          <div className="panel-head"><BarChart3 size={13} /> 分通道灵敏度 · F1</div>
          <div className="p-2" style={{ height: Math.max(260, sortedF1.length * 34 + 60) }}>
            {sortedF1.length > 0 ? (
              <Plot
                data={[
                  {
                    y: sortedF1.map(c => `${CHANNEL_LABELS[c.channel] ?? c.channel}`),
                    x: sortedF1.map(c => c.f1),
                    type: 'bar',
                    orientation: 'h',
                    marker: {
                      color: sortedF1.map(c => c.f1),
                      colorscale: [[0, 'rgba(56,189,248,0.25)'], [1, '#38BDF8']],
                      line: { color: 'rgba(56,189,248,0.5)', width: 1 },
                    },
                    text: sortedF1.map(c => c.f1.toFixed(3)),
                    textposition: 'outside',
                    textfont: { family: 'IBM Plex Mono', size: 10, color: '#93A1B7' },
                    hovertemplate: '%{y}<br>F1 = %{x:.3f}<extra></extra>',
                  },
                ]}
                layout={{
                  autosize: true,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  margin: { l: 110, r: 50, t: 10, b: 30 },
                  font: { family: 'IBM Plex Sans, sans-serif', color: '#93A1B7', size: 11 },
                  xaxis: { gridcolor: 'rgba(148,163,184,0.08)', range: [0, Math.max(0.7, ...sortedF1.map(c => c.f1)) * 1.15] },
                  yaxis: { autorange: 'reversed', linecolor: 'rgba(148,163,184,0.2)' },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%', height: '100%' }}
              />
            ) : (
              <div className="h-full flex items-center justify-center text-sm tx-3">分通道 F1 数据不可用</div>
            )}
          </div>
        </div>

        {/* RAG 知识库 */}
        <div className="panel rise rise-4">
          <div className="panel-head"><BookOpen size={13} /> RAG 知识库</div>
          <div className="p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs tx-3">引擎状态</span>
              <span className="flex items-center gap-1.5 text-xs font-mono" style={{ color: rag?.status === 'online' ? '#22C55E' : '#64748B' }}>
                <span className={`dot ${rag?.status === 'online' ? 'dot-nominal' : 'dot-offline'}`} />
                {rag?.status === 'online' ? 'ONLINE' : 'OFFLINE'}
              </span>
            </div>
            <div className="flex items-center justify-between py-2 border-t border-[rgba(148,163,184,0.07)]">
              <span className="text-xs tx-3">嵌入模型</span>
              <span className="font-mono text-sm tx-1">{rag?.embedding_model ?? '—'}</span>
            </div>
            <div className="flex items-center justify-between py-2 border-t border-[rgba(148,163,184,0.07)]">
              <span className="text-xs tx-3">FAISS 分块</span>
              <span className="font-mono text-sm text-[#38BDF8]">{rag?.chunk_count?.toLocaleString() ?? '—'}</span>
            </div>
            <div className="pt-2 border-t border-[rgba(148,163,184,0.07)]">
              <div className="text-xs tx-3 mb-2">挂载文档分布</div>
              <div className="flex gap-2">
                {([['PDF', rag?.doc_count?.pdf ?? 0, '#EF4444'], ['MD', rag?.doc_count?.md ?? 0, '#38BDF8'], ['HTML', rag?.doc_count?.html ?? 0, '#2DD4BF']] as const).map(([k, v, c]) => (
                  <div key={k} className="flex-1 panel !rounded-lg p-2.5 text-center" style={{ borderTop: `2px solid ${c}66` }}>
                    <div className="stat-number text-xl" style={{ color: c }}>{v}</div>
                    <div className="text-[10px] tx-3 font-mono mt-0.5">{k}</div>
                  </div>
                ))}
              </div>
            </div>
            {rag?.status !== 'online' && (
              <div className="text-[11px] tx-3 leading-relaxed pt-1">
                知识库索引未加载 — 诊断摘要仍可读,自由问答不可用。
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
