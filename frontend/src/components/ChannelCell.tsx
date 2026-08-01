import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist';
import type { ChannelInfo } from '../api/types';
import { sevColor } from './severity';

/** 通道矩阵单元格:极简编号 + 迷你波形 + 状态色 */
export default function ChannelCell({ data, index, onClick }: { data: ChannelInfo; index: number; onClick?: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const color = sevColor(data.status);

  useEffect(() => {
    const el = ref.current;
    if (!el || !data.sparkline_values?.length) return;
    const validIdx = data.anomaly_indices.filter(i => i < data.sparkline_values.length);
    Plotly.react(
      el,
      [
        {
          y: data.sparkline_values,
          mode: 'lines', type: 'scatter',
          line: { color, width: 1.2 },
          fill: 'tozeroy', fillcolor: `${color}12`,
          hoverinfo: 'skip', showlegend: false,
        },
        ...(validIdx.length
          ? [{
              x: validIdx, y: validIdx.map(i => data.sparkline_values[i]),
              mode: 'markers', type: 'scatter' as const,
              marker: { color: '#F43F5E', size: 4 },
              hoverinfo: 'skip', showlegend: false,
            }]
          : []),
      ],
      {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        plot_bgcolor: 'rgba(0,0,0,0)', paper_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { visible: false, fixedrange: true },
        yaxis: { visible: false, fixedrange: true },
        showlegend: false,
      },
      { displayModeBar: false, responsive: true, staticPlot: true },
    );
  }, [data, color]);

  return (
    <button
      onClick={onClick}
      className="relative text-left border border-[rgba(94,234,212,0.1)] rounded bg-[rgba(8,16,32,0.45)] p-2 hover:border-[rgba(94,234,212,0.35)] hover:bg-[rgba(34,211,238,0.05)] transition-all group cursor-pointer"
      style={{ borderTopColor: `${color}88`, borderTopWidth: 2 }}
    >
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] tx-2 group-hover:text-[#22d3ee] transition-colors">
          {String(index + 1).padStart(2, '0')} {data.channel}
        </span>
        <span className="w-1.5 h-1.5 rounded-full" style={{ background: color, boxShadow: `0 0 5px ${color}` }} />
      </div>
      <div ref={ref} className="w-full" style={{ height: 34 }} />
      <div className="flex items-center justify-between text-[9px] font-mono">
        <span className="tx-3 truncate mr-1">{data.label}</span>
        {data.provisional && (
          <span className="shrink-0 mr-1 px-1 rounded border border-[rgba(251,191,36,0.4)] text-[#fbbf24]" title="段未闭合 · 增长段预览分,非正式判定">预览</span>
        )}
        <span style={{ color }}>{(data.anomaly_rate * 100).toFixed(0)}%</span>
      </div>
    </button>
  );
}
