import { useEffect, useState } from 'react';
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
      .catch(() => setError('TELEMETRY LINK LOST — 无法连接后端 (localhost:8000)'));
  }, []);

  if (error) {
    return (
      <div className="h-full flex items-center justify-center">
        <span className="font-mono text-xs text-[#fbbf24]">{error}</span>
      </div>
    );
  }

  const sortedF1 = [...channelF1].sort((a, b) => b.f1 - a.f1);

  return (
    <div className="h-full overflow-y-auto space-y-3 pr-1">
      <div className="grid grid-cols-12 gap-3">
        {/* 系统参数 */}
        <div className="col-span-12 xl:col-span-3 deck-panel rise">
          <div className="deck-head"><span className="tick" /> 检测引擎</div>
          <div className="p-3.5 space-y-1">
            {([
              ['ALGORITHM', params?.algorithm ?? '—'],
              ['DIMENSIONS', params ? `${params.dimensions}` : '—'],
              ['CONTAMINATION', params ? `${params.contamination}` : '—'],
            ]).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between py-2 border-b border-[rgba(94,234,212,0.07)] last:border-0">
                <span className="font-mono text-[10px] tx-3">{k}</span>
                <span className="font-mono text-[12px] tx-1">{v}</span>
              </div>
            ))}
            <div className="pt-2 text-[10px] tx-3 leading-relaxed font-mono">
              PIPELINE: GLOBAL-IF → PER-CHANNEL-IF → RULE-FUSION
            </div>
          </div>
        </div>

        {/* F1 演进轨迹 */}
        <div className="col-span-12 xl:col-span-9 deck-panel rise rise-1">
          <div className="deck-head justify-between">
            <span className="flex items-center gap-2"><span className="tick" /> 实验演进轨迹 · SEGF1 TRAJECTORY</span>
            <span className="font-mono text-[10px] text-[#2dd4a7] normal-case tracking-normal">+89.7% TOTAL GAIN</span>
          </div>
          <div className="h-[240px] p-1">
            {experiments.length > 0 && (
              <Plot
                data={[
                  {
                    x: experiments.map((_, i) => i),
                    y: experiments.map(e => e.f1),
                    type: 'scatter', mode: 'lines+markers+text',
                    line: { color: '#22d3ee', width: 2, shape: 'spline', smoothing: 0.5 },
                    marker: { size: 11, color: '#050a14', line: { color: '#22d3ee', width: 2 } },
                    text: experiments.map(e => `${e.f1.toFixed(4)}<br><span style="font-size:9px;color:#2dd4a7">${e.improvement}</span>`),
                    textposition: 'top center',
                    textfont: { family: 'Orbitron, sans-serif', size: 12, color: '#d9f6ef' },
                    fill: 'tozeroy', fillcolor: 'rgba(34,211,238,0.06)',
                    customdata: experiments.map(e => [e.version, e.strategy]),
                    hovertemplate: '%{customdata[0]} · %{customdata[1]}<br>SegF1 = %{y:.4f}<extra></extra>',
                  },
                ]}
                layout={{
                  autosize: true,
                  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                  margin: { l: 46, r: 20, t: 34, b: 34 },
                  font: { family: 'IBM Plex Mono, monospace', color: '#47616c', size: 10 },
                  xaxis: {
                    gridcolor: 'rgba(94,234,212,0.05)', linecolor: 'rgba(94,234,212,0.2)',
                    tickmode: 'array',
                    tickvals: experiments.map((_, i) => i),
                    ticktext: experiments.map(e => e.version),
                    range: [-0.45, experiments.length - 0.55],
                  },
                  yaxis: { gridcolor: 'rgba(94,234,212,0.07)', linecolor: 'rgba(94,234,212,0.2)', range: [0, 0.78] },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%', height: '100%' }}
              />
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-3">
        {/* 分通道 F1 */}
        <div className="col-span-12 xl:col-span-8 deck-panel rise rise-2">
          <div className="deck-head"><span className="tick" /> 分通道灵敏度 · PER-CHANNEL F1</div>
          <div className="p-1" style={{ height: Math.max(250, sortedF1.length * 32 + 50) }}>
            {sortedF1.length > 0 ? (
              <Plot
                data={[
                  {
                    y: sortedF1.map(c => CHANNEL_LABELS[c.channel] ?? c.channel),
                    x: sortedF1.map(c => c.f1),
                    type: 'bar', orientation: 'h',
                    marker: {
                      color: sortedF1.map(c => c.f1),
                      colorscale: [[0, 'rgba(34,211,238,0.15)'], [1, '#22d3ee']],
                      line: { color: 'rgba(34,211,238,0.4)', width: 1 },
                    },
                    text: sortedF1.map(c => c.f1.toFixed(3)),
                    textposition: 'outside',
                    textfont: { family: 'IBM Plex Mono', size: 10, color: '#7fa8a6' },
                    hovertemplate: '%{y}<br>F1 = %{x:.3f}<extra></extra>',
                  },
                ]}
                layout={{
                  autosize: true,
                  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                  margin: { l: 100, r: 48, t: 8, b: 28 },
                  font: { family: 'IBM Plex Sans, sans-serif', color: '#7fa8a6', size: 11 },
                  xaxis: { gridcolor: 'rgba(94,234,212,0.07)', range: [0, Math.max(0.7, ...sortedF1.map(c => c.f1)) * 1.15] },
                  yaxis: { autorange: 'reversed', linecolor: 'rgba(94,234,212,0.2)' },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%', height: '100%' }}
              />
            ) : (
              <div className="h-full flex items-center justify-center font-mono text-[10px] tx-3">CHANNEL F1 UNAVAILABLE</div>
            )}
          </div>
        </div>

        {/* RAG 知识库 */}
        <div className="col-span-12 xl:col-span-4 deck-panel rise rise-3">
          <div className="deck-head justify-between">
            <span className="flex items-center gap-2"><span className="tick" /> RAG 知识库</span>
            <span className="font-mono text-[10px]" style={{ color: rag?.status === 'online' ? '#2dd4a7' : '#5b6b7f' }}>
              {rag?.status === 'online' ? '● ONLINE' : '○ OFFLINE'}
            </span>
          </div>
          <div className="p-3.5 space-y-1">
            <div className="flex items-center justify-between py-2 border-b border-[rgba(94,234,212,0.07)]">
              <span className="font-mono text-[10px] tx-3">EMBED MODEL</span>
              <span className="font-mono text-[12px] tx-1">{rag?.embedding_model ?? '—'}</span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-[rgba(94,234,212,0.07)]">
              <span className="font-mono text-[10px] tx-3">FAISS CHUNKS</span>
              <span className="font-mono text-[12px] text-[#22d3ee]">{rag?.chunk_count?.toLocaleString() ?? '—'}</span>
            </div>
            <div className="pt-3">
              <div className="font-mono text-[10px] tx-3 mb-2">DOCUMENT MOUNT</div>
              <div className="flex gap-2">
                {([['PDF', rag?.doc_count?.pdf ?? 0, '#f43f5e'], ['MD', rag?.doc_count?.md ?? 0, '#22d3ee'], ['HTML', rag?.doc_count?.html ?? 0, '#2dd4a7']] as const).map(([k, v, c]) => (
                  <div key={k} className="flex-1 border rounded p-2.5 text-center"
                    style={{ borderColor: 'rgba(94,234,212,0.12)', borderTopColor: c, borderTopWidth: 2, background: 'rgba(8,16,32,0.5)' }}>
                    <div className="big-readout text-lg" style={{ color: c }}>{v}</div>
                    <div className="text-[9px] tx-3 font-mono mt-0.5">{k}</div>
                  </div>
                ))}
              </div>
            </div>
            {rag?.status !== 'online' && (
              <div className="text-[10px] font-mono tx-3 leading-relaxed pt-2">
                NOTE: 索引未加载 — 诊断摘要可读,自由问答不可用
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
