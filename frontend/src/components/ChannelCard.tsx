import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist';
import type { ChannelInfo } from '../api/types';
import { sevColor, sevDotClass, SEVERITY_LABELS } from './severity';
import type { Severity } from '../api/types';

/** 通道状态卡:sparkline + 异常点标记 + 状态 */
export default function ChannelCard({ data, onClick }: { data: ChannelInfo; onClick?: () => void }) {
  const plotRef = useRef<HTMLDivElement>(null);
  const color = sevColor(data.status);
  const status = (data.status in SEVERITY_LABELS ? data.status : 'offline') as Severity;

  useEffect(() => {
    const el = plotRef.current;
    if (!el || !data.sparkline_values?.length) return;

    const validIdx = data.anomaly_indices.filter(i => i < data.sparkline_values.length);

    Plotly.react(
      el,
      [
        {
          y: data.sparkline_values,
          mode: 'lines',
          type: 'scatter',
          line: { color, width: 1.4 },
          fill: 'tozeroy',
          fillcolor: `${color}14`,
          hoverinfo: 'skip',
          showlegend: false,
        },
        ...(validIdx.length
          ? [{
              x: validIdx,
              y: validIdx.map(i => data.sparkline_values[i]),
              mode: 'markers',
              type: 'scatter' as const,
              marker: { color: '#EF4444', size: 5, symbol: 'circle' },
              hoverinfo: 'skip',
              showlegend: false,
            }]
          : []),
      ],
      {
        height: 72,
        margin: { l: 2, r: 2, t: 2, b: 2 },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
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
      className="panel text-left p-3.5 w-full group cursor-pointer"
      style={{ borderTop: `2px solid ${color}55` }}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className={sevDotClass(data.status)} />
          <span className="font-mono text-[13px] font-medium tx-1">{data.channel}</span>
          <span className="text-xs tx-3 truncate">{data.label}</span>
        </div>
        <span
          className="text-[10px] font-mono px-1.5 py-0.5 rounded uppercase tracking-wider"
          style={{ color, background: `${color}18`, border: `1px solid ${color}40` }}
        >
          {SEVERITY_LABELS[status]}
        </span>
      </div>
      <div ref={plotRef} className="w-full opacity-90 group-hover:opacity-100 transition-opacity" style={{ height: 72 }} />
      <div className="flex items-center justify-between mt-2 text-[11px] font-mono">
        <span className="tx-3">异常占比</span>
        <span style={{ color }}>{(data.anomaly_rate * 100).toFixed(1)}%</span>
      </div>
    </button>
  );
}
