import { useMemo } from 'react';
import Plot from './Plot';
import type { WaveformData } from '../api/types';

/** 遥测示波器:波形 + 3σ 阈值带 + 异常点 + 滑块 */
export default function Oscilloscope({ wave }: { wave: WaveformData }) {
  const { values, timestamps, anomaly_mask, upper_3sigma, lower_3sigma, mean } = wave;

  const { ax, ay } = useMemo(() => {
    const ax: (string | number)[] = [];
    const ay: number[] = [];
    anomaly_mask.forEach((m, i) => {
      if (m === 1 && i < values.length) {
        ax.push(timestamps[i]);
        ay.push(values[i]);
      }
    });
    return { ax, ay };
  }, [values, timestamps, anomaly_mask]);

  const x0 = timestamps[0];
  const x1 = timestamps[timestamps.length - 1];

  return (
    <Plot
      data={[
        // 3σ 阈值带(上下界连线填充)
        {
          x: [x0, x1, x1, x0],
          y: [lower_3sigma, lower_3sigma, upper_3sigma, upper_3sigma],
          fill: 'toself',
          fillcolor: 'rgba(34, 211, 238, 0.05)',
          line: { width: 0 },
          hoverinfo: 'skip',
          showlegend: false,
          type: 'scatter',
        },
        // 主波形
        {
          x: timestamps,
          y: values,
          type: 'scatter',
          mode: 'lines',
          line: { color: '#22d3ee', width: 1.6 },
          name: '遥测值',
          hovertemplate: '%{y:.4f}<extra>%{x}</extra>',
        },
        // 异常点
        ...(ax.length
          ? [{
              x: ax,
              y: ay,
              type: 'scatter' as const,
              mode: 'markers',
              marker: {
                color: '#f43f5e',
                size: 9,
                symbol: 'x',
                line: { color: '#f43f5e', width: 2 },
              },
              name: '异常点',
              hovertemplate: '异常 %{y:.4f}<extra>%{x}</extra>',
            }]
          : []),
      ]}
      layout={{
        autosize: true,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 52, r: 16, t: 12, b: 42 },
        font: { family: 'IBM Plex Mono, monospace', color: '#47616c', size: 10 },
        xaxis: {
          gridcolor: 'rgba(94,234,212,0.07)',
          zerolinecolor: 'rgba(94,234,212,0.15)',
          showline: true,
          linecolor: 'rgba(94,234,212,0.2)',
          nticks: 10,
        },
        yaxis: {
          gridcolor: 'rgba(94,234,212,0.07)',
          zerolinecolor: 'rgba(94,234,212,0.15)',
          showline: true,
          linecolor: 'rgba(94,234,212,0.2)',
        },
        showlegend: false,
        hovermode: 'x unified',
        shapes: [
          {
            type: 'line', x0: 0, x1: 1, xref: 'paper',
            y0: upper_3sigma, y1: upper_3sigma, yref: 'y',
            line: { color: '#fbbf24', width: 1, dash: 'dashdot' },
          },
          {
            type: 'line', x0: 0, x1: 1, xref: 'paper',
            y0: lower_3sigma, y1: lower_3sigma, yref: 'y',
            line: { color: '#fbbf24', width: 1, dash: 'dashdot' },
          },
          {
            type: 'line', x0: 0, x1: 1, xref: 'paper',
            y0: mean, y1: mean, yref: 'y',
            line: { color: 'rgba(45, 212, 167, 0.5)', width: 1, dash: 'dot' },
          },
        ],
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%', height: '100%' }}
    />
  );
}
