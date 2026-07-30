import React, { useState, useEffect, useRef, useMemo, useCallback, startTransition } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  Activity,
  Sliders,
  ChevronRight,
  RefreshCw,
  Send,
  Terminal as TermIcon,
  Clock
} from 'lucide-react';
import Plotly from 'plotly.js-dist';

// ==========================================
// 1. 业务常量
// ==========================================

const CHANNEL_MAP: Record<string, string> = {
  "CADC0872": "磁力计 X轴", "CADC0873": "磁力计 Y轴", "CADC0874": "磁力计 Z轴",
  "CADC0884": "光电二极管 1", "CADC0886": "光电二极管 2", "CADC0888": "光电二极管 3",
  "CADC0890": "光电二极管 4", "CADC0892": "光电二极管 5", "CADC0894": "光电二极管 6",
};

const NO_ANOMALY_CHANNELS = ["CADC0884"];

type Severity = 'nominal' | 'caution' | 'warning' | 'critical' | 'offline';

const getSeverity = (ch: string, score: number): Severity => {
  if (NO_ANOMALY_CHANNELS.includes(ch)) return 'offline';
  if (score > 0.4) return 'critical';
  if (score > 0.15) return 'warning';
  if (score > 0.05) return 'caution';
  return 'nominal';
};

const SEVERITY_COLORS: Record<Severity, string> = {
  nominal: '#22C55E',
  caution: '#F97316',
  warning: '#F59E0B',
  critical: '#EF4444',
  offline: '#64748B',
};

// ==========================================
// 2. Plotly 图表包装
// ==========================================

interface PlotProps {
  data: Record<string, unknown>[];
  layout: Record<string, unknown>;
  config?: Record<string, unknown>;
  style?: React.CSSProperties;
  useResizeHandler?: boolean;
}

const Plot: React.FC<PlotProps> = ({ data, layout, config, style, useResizeHandler }) => {
  const plotRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = plotRef.current;
    if (el && data) {
      Plotly.newPlot(el, data, layout, config);
      return () => {
        Plotly.purge(el);
      };
    }
  }, [data, layout, config]);

  useEffect(() => {
    if (!useResizeHandler || !plotRef.current) return;
    const observer = new ResizeObserver(() => {
      if (plotRef.current) Plotly.Plots.resize(plotRef.current);
    });
    observer.observe(plotRef.current);
    return () => observer.disconnect();
  }, [useResizeHandler]);

  return <div ref={plotRef} style={style} />;
};

// ==========================================
// 3. 通道波形组件
// ==========================================

interface ChannelData {
  channel: string;
  label: string;
  anomaly_rate: number;
  status: string;
  sparkline_values: number[];
  anomaly_indices: number[];
}

const ChannelSparkline: React.FC<{ data: ChannelData }> = ({ data }) => {
  const plotRef = useRef<HTMLDivElement>(null);
  const color = SEVERITY_COLORS[data.status as Severity] || '#64748B';

  useEffect(() => {
    if (!plotRef.current || !data.sparkline_values.length) return;

    const anomalyY = data.anomaly_indices
      .filter(idx => idx < data.sparkline_values.length)
      .map(idx => data.sparkline_values[idx]);
    const anomalyX = data.anomaly_indices
      .filter(idx => idx < data.sparkline_values.length);

    const el = plotRef.current;

    Plotly.newPlot(el, [
      {
        y: data.sparkline_values,
        mode: 'lines',
        type: 'scatter',
        line: { color, width: 1.2 },
        fill: 'tozeroy',
        fillcolor: `${color}10`,
        hoverinfo: 'y',
        showlegend: false,
      },
      ...(anomalyX.length > 0 ? [{
        x: anomalyX,
        y: anomalyY,
        mode: 'markers',
        type: 'scatter' as const,
        marker: { color: '#EF4444', size: 5 },
        hoverinfo: 'x+y',
        showlegend: false,
      }] : [])
    ], {
      height: 90,
      margin: { l: 0, r: 0, t: 0, b: 0 },
      plot_bgcolor: 'rgba(0,0,0,0)',
      paper_bgcolor: 'rgba(0,0,0,0)',
      xaxis: { visible: false },
      yaxis: { visible: false },
      showlegend: false,
    }, { displayModeBar: false, responsive: true });

    return () => {
      Plotly.purge(el);
    };
  }, [data, color]);

  return (
    <div className="card p-3">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center space-x-2">
          <span className={`status-dot ${data.status === 'critical' ? 'status-dot-red' : data.status === 'warning' || data.status === 'caution' ? 'status-dot-yellow' : data.status === 'offline' ? 'status-dot-gray' : 'status-dot-green'}`} />
          <span className="text-sm font-medium text-[var(--text-primary)]">{data.channel}</span>
          <span className="text-xs text-[var(--text-muted)]">{data.label}</span>
        </div>
        <span className="font-mono text-xs" style={{ color }}>
          {(data.anomaly_rate * 100).toFixed(1)}%
        </span>
      </div>
      <div ref={plotRef} className="w-full" style={{ minHeight: 90 }} />
    </div>
  );
};

// ==========================================
// 4. 交互终端组件
// ==========================================

const TerminalComponent: React.FC<{ channelId: string }> = ({ channelId }) => {
  const [messages] = useState<Array<{ role: 'user' | 'assistant'; text: string }>>([
    { role: 'user', text: `针对 ${channelId} 的异常，对姿态控制有什么具体危害？` },
    { role: 'assistant', text: '根据卫星大系统ADCS手册第112页指出：磁力计的异常突刺会导致姿态解算矩阵无法正常收敛。当系统连续检测到单轴发散超3σ限界，建议立即切换至外部高精度星敏感器（Star Tracker）备份链路。' }
  ]);
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (!input.trim()) return;
    setInput('');
  };

  return (
    <div className="card">
      <div className="flex items-center space-x-2 text-xs font-medium text-[var(--text-secondary)] mb-3 pb-2 border-b border-[var(--border-subtle)]">
        <TermIcon size={14} />
        <span>智能交互终端</span>
      </div>
      <div className="h-40 overflow-y-auto text-xs space-y-3 mb-3 pr-2">
        {messages.map((m, i) => (
          <div key={i} className={`${m.role === 'user' ? 'text-[var(--accent-blue)]' : 'text-[var(--text-secondary)]'}`}>
            <span className="font-mono text-[10px] mr-1 opacity-60">{m.role === 'user' ? '>' : '●'}</span>
            <span className="leading-relaxed">{m.text}</span>
          </div>
        ))}
      </div>
      <div className="flex items-center space-x-2 border border-[var(--border-default)] bg-[var(--bg-input)] rounded px-3 py-2 focus-within:border-[var(--accent-blue)] transition-colors">
        <span className="text-xs font-mono text-[var(--text-muted)]">&gt;</span>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="输入遥测排故溯源追问指令..."
          className="flex-1 bg-transparent border-none text-xs text-[var(--text-primary)] focus:outline-none placeholder-[var(--text-muted)]"
        />
        <button onClick={handleSend} className="text-[var(--text-muted)] hover:text-[var(--accent-blue)] transition-colors">
          <Send size={14} />
        </button>
      </div>
    </div>
  );
};

// ==========================================
// 5. 示波器波形组件
// ==========================================

interface OscilloscopeProps {
  data: number[];
  timestamps: string[];
  anomalies: boolean[];
}

const OscilloscopePlotly: React.FC<OscilloscopeProps> = ({ data, timestamps, anomalies }) => {
  const { anomalyX, anomalyY } = useMemo(() => {
    const aX: string[] = [];
    const aY: number[] = [];
    anomalies.forEach((isAnomaly, idx) => {
      if (isAnomaly) { aX.push(timestamps[idx]); aY.push(data[idx]); }
    });
    return { anomalyX: aX, anomalyY: aY };
  }, [data, timestamps, anomalies]);

  const mean = data.reduce((a, b) => a + b, 0) / (data.length || 1);
  const std = Math.sqrt(data.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / (data.length || 1));

  return (
    <div className="w-full h-full">
      <Plot
        data={[
          {
            x: timestamps, y: data, type: 'scatter', mode: 'lines',
            line: { color: '#38BDF8', width: 1.5, shape: 'spline' },
            name: '遥测值', hoverinfo: 'x+y',
          },
          {
            x: anomalyX, y: anomalyY, type: 'scatter', mode: 'markers',
            marker: { color: '#EF4444', size: 10, symbol: 'x', line: { color: '#EF4444', width: 2 } },
            name: '异常突变', hoverinfo: 'x+y',
          }
        ]}
        layout={{
          autosize: true,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          margin: { l: 40, r: 20, t: 10, b: 30 },
          font: { family: 'IBM Plex Mono, monospace', color: '#94A3B8', size: 10 },
          xaxis: { gridcolor: '#1E293B', zerolinecolor: '#2D3548', showline: true, linecolor: '#2D3548' },
          yaxis: { gridcolor: '#1E293B', zerolinecolor: '#2D3548', showline: true, linecolor: '#2D3548' },
          showlegend: false,
          hovermode: 'x unified',
          shapes: [
            { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: mean + 3 * std, y1: mean + 3 * std, yref: 'y', line: { color: '#F59E0B', width: 1, dash: 'dashdot' } },
            { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: mean - 3 * std, y1: mean - 3 * std, yref: 'y', line: { color: '#F59E0B', width: 1, dash: 'dashdot' } }
          ]
        }}
        useResizeHandler={true}
        style={{ width: '100%', height: '100%' }}
        config={{ displayModeBar: false, responsive: true }}
      />
    </div>
  );
};

// ==========================================
// 6. Dashboard 主视图
// ==========================================

const REFRESH_INTERVAL = 30000;

interface MetricsData {
  anomaly_rate_str: string;
  anomaly_rows: number;
  total_rows: number;
  fault_count: number;
  total_anomalies: number;
  best_f1: number;
  throughput: number;
  channel_count: number;
}

interface AlertItem {
  segment: string;
  channel: string;
  channel_label: string;
  anomaly_score: number;
  severity: string;
  summary: string;
  anomaly_type: string;
}

interface AnomalyDataContext {
  seg?: string;
  ch?: string;
  score?: number;
  summary?: string;
}

interface SourceItem {
  filename: string;
  page?: string;
  score?: number;
  local_path?: string;
}

interface WaveformData {
  values: number[];
  timestamps: string[];
  anomaly_mask: number[];
}

interface DetailSections {
  '可能原因'?: string;
  '影响评估'?: string;
  '建议措施'?: string;
  '结论'?: string;
}

interface DetailData {
  channel_label?: string;
  urgency?: string;
  sections: DetailSections;
  sources?: SourceItem[];
  raw_explanation?: string;
}

const DashboardView: React.FC<{ onNavigate: (page: string, channelId?: string, anomalyData?: AnomalyDataContext) => void }> = ({ onNavigate }) => {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [channels, setChannels] = useState<ChannelData[]>([]);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchAll = useCallback(async (isAutoRefresh = false) => {
    startTransition(() => {
      if (!isAutoRefresh) setRefreshing(true);
      setError(null);
    });
    try {
      const [metricsRes, channelsRes, alertsRes] = await Promise.all([
        fetch('/api/dashboard/metrics'),
        fetch('/api/dashboard/channels'),
        fetch('/api/dashboard/alerts'),
      ]);
      if (metricsRes.ok) setMetrics(await metricsRes.json());
      if (channelsRes.ok) {
        const data = await channelsRes.json();
        setChannels(data.channels || []);
      } else {
        setError(`API 返回错误 (${channelsRes.status})`);
      }
      if (alertsRes.ok) {
        const data = await alertsRes.json();
        setAlerts(data.alerts || []);
      }
      setLastUpdated(new Date());
    } catch (e) {
      console.error('Dashboard fetch error:', e);
      setError('无法连接后端服务 (localhost:8000)，请确认已运行 python start_ui.py');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    const raf = requestAnimationFrame(() => fetchAll());
    const timer = setInterval(() => fetchAll(true), REFRESH_INTERVAL);
    return () => {
      cancelAnimationFrame(raf);
      clearInterval(timer);
    };
  }, [fetchAll]);

  const faultCount = metrics?.fault_count ?? alerts.length;
  const totalAnomalies = metrics?.total_anomalies ?? 0;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* 标题栏 */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-[var(--text-primary)] tracking-wide">
          实时告警中心
        </h2>
        <div className="flex items-center space-x-3">
          {lastUpdated && (
            <span className="text-xs text-[var(--text-muted)] font-mono">
              <Clock size={10} className="inline mr-1" />
              {lastUpdated.toLocaleTimeString('zh-CN')}
            </span>
          )}
          <button
            onClick={() => fetchAll()}
            disabled={refreshing}
            className="text-xs text-[var(--accent-blue)] hover:text-white bg-[var(--accent-blue)]/10 hover:bg-[var(--accent-blue)]/20 px-2.5 py-1 rounded border border-[var(--accent-blue)]/20 transition-all flex items-center space-x-1 disabled:opacity-50"
          >
            <RefreshCw size={12} className={refreshing ? 'animate-spin' : ''} />
            <span>{refreshing ? '刷新中...' : '刷新'}</span>
          </button>
        </div>
      </div>

      {/* 核心指标卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card">
          <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">系统异常率</div>
          <div className="text-3xl font-bold font-mono text-[var(--text-primary)]">
            {metrics?.anomaly_rate_str ?? '—'}
          </div>
          <div className="text-xs text-[var(--text-secondary)] mt-1">
            {metrics ? `${metrics.anomaly_rows.toLocaleString()} / ${metrics.total_rows.toLocaleString()} 行` : '加载中...'}
          </div>
        </div>
        <div className={`card ${faultCount > 0 ? 'card-alert-orange' : ''}`}>
          <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">告警队列 (&gt;0.05)</div>
          <div className="text-3xl font-bold font-mono text-[var(--text-primary)]">
            {faultCount}
            <span className="text-sm font-normal text-[var(--text-secondary)] ml-1">条</span>
          </div>
          <div className="text-xs text-[var(--text-secondary)] mt-1">
            共 {totalAnomalies} 条异常段
          </div>
        </div>
        <div className="card">
          <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">诊断 F1</div>
          <div className="text-3xl font-bold font-mono text-[var(--text-primary)]">
            {metrics?.best_f1?.toFixed(3) ?? '—'}
          </div>
          <div className="text-xs text-[var(--accent-green)] mt-1">Stage 2 Fusion</div>
        </div>
        <div className="card">
          <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">数据吞吐量</div>
          <div className="text-3xl font-bold font-mono text-[var(--text-primary)]">
            {metrics ? `${Math.floor(metrics.throughput / 1000)}K` : '—'}
            <span className="text-sm font-normal text-[var(--text-secondary)] ml-1">行</span>
          </div>
          <div className="text-xs text-[var(--text-secondary)] mt-1">{metrics?.channel_count ?? 9} 通道</div>
        </div>
      </div>

      {/* 遥测通道波形矩阵 */}
      <div>
        <div className="text-sm font-medium text-[var(--text-primary)] mb-3">
          遥测通道矩阵
          <span className="text-xs text-[var(--text-muted)] ml-2">实时监测中</span>
        </div>
        {error ? (
          <div className="card card-alert-yellow">
            <div className="text-sm text-[var(--accent-yellow)]">{error}</div>
          </div>
        ) : channels.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {channels.map(ch => (
              <ChannelSparkline key={ch.channel} data={ch} />
            ))}
          </div>
        ) : (
          <div className="card">
            <div className="text-sm text-[var(--text-muted)] text-center py-8">
              {loading ? '正在加载通道数据...' : '未找到数据文件'}
            </div>
          </div>
        )}
      </div>

      {/* 故障告警队列 */}
      <div className="card">
        <div className="text-sm font-medium text-[var(--text-primary)] mb-3 pb-2 border-b border-[var(--border-subtle)]">
          故障告警队列
        </div>
        {alerts.length === 0 ? (
          <div className="text-sm text-[var(--text-muted)] text-center py-6">
            {loading ? '正在加载...' : '当前无待处理告警'}
          </div>
        ) : (
          <div className="divide-y divide-[var(--border-subtle)]">
            {alerts.map((q, i) => {
              const severity = getSeverity(q.channel, q.anomaly_score);
              const color = SEVERITY_COLORS[severity];
              const alertClass = severity === 'critical' ? 'card-alert-red' : severity === 'warning' ? 'card-alert-orange' : '';

              return (
                <div key={i} className={`py-3 first:pt-0 last:pb-0 -mx-4 px-4 ${alertClass}`}>
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 mb-1">
                    <div className="flex items-center space-x-2 text-sm">
                      <span className={`status-dot ${severity === 'critical' ? 'status-dot-red' : severity === 'warning' || severity === 'caution' ? 'status-dot-yellow' : 'status-dot-green'}`} />
                      <span className="font-medium text-[var(--text-primary)]">段 #{q.segment}</span>
                      <span className="text-[var(--text-muted)]">·</span>
                      <span className="text-[var(--accent-blue)] font-mono text-xs">{q.channel}</span>
                      <span className="text-[var(--text-muted)]">·</span>
                      <span className="text-[var(--text-secondary)]">{q.channel_label}</span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <span className="font-mono text-sm font-bold" style={{ color }}>
                        {q.anomaly_score.toFixed(3)}
                      </span>
                      <button
                        onClick={() => onNavigate('explanation', q.channel, { seg: q.segment, ch: q.channel, score: q.anomaly_score, summary: q.summary })}
                        className="text-xs text-[var(--accent-blue)] hover:text-white bg-[var(--accent-blue)]/10 hover:bg-[var(--accent-blue)]/20 px-2 py-0.5 rounded border border-[var(--accent-blue)]/20 transition-all flex items-center"
                      >
                        深度诊断 <ChevronRight size={12} className="ml-0.5" />
                      </button>
                    </div>
                  </div>
                  <p className="text-xs text-[var(--text-secondary)] mt-1 pl-4 border-l-2 border-[var(--border-default)]">
                    {q.summary}
                  </p>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

// ==========================================
// 7. Detection 视图
// ==========================================

const DetectionView: React.FC = () => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in">
      <div className="card">
        <div className="text-sm font-medium text-[var(--text-primary)] mb-4 pb-2 border-b border-[var(--border-subtle)]">
          RAG 知识库配置
        </div>
        <div className="space-y-2 text-sm text-[var(--text-secondary)]">
          <div className="flex justify-between py-1 border-b border-[var(--border-subtle)]">
            <span>挂载文档分布</span>
            <span className="font-medium text-[var(--text-primary)]">8 PDF + 3 MD + 1 HTML</span>
          </div>
          <div className="flex justify-between py-1 border-b border-[var(--border-subtle)]">
            <span>FAISS CHUNKS</span>
            <span className="font-mono text-[var(--accent-blue)]">2,847 Blocks</span>
          </div>
          <div className="flex justify-between py-1">
            <span>嵌入向量引擎</span>
            <span className="font-medium text-[var(--text-primary)]">BGE-M3</span>
          </div>
        </div>
        <button className="w-full mt-4 bg-[var(--accent-blue)]/10 hover:bg-[var(--accent-blue)]/20 border border-[var(--accent-blue)]/30 text-[var(--accent-blue)] text-sm py-2 rounded transition-all flex items-center justify-center space-x-2">
          <RefreshCw size={12} /><span>全量重建 FAISS 索引</span>
        </button>
      </div>

      <div className="card">
        <div className="text-sm font-medium text-[var(--text-primary)] mb-4 pb-2 border-b border-[var(--border-subtle)]">
          实验演进路线 (F1)
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-[var(--border-default)] text-[var(--text-muted)] text-xs">
                <th className="py-2">版本</th>
                <th>技术策略</th>
                <th>段级 F1</th>
                <th>提升</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border-subtle)]">
              <tr><td className="py-2 text-[var(--text-muted)]">Stage 0</td><td>全局 IForest (c=0.2)</td><td className="font-mono">0.2996</td><td className="text-[var(--text-muted)]">—</td></tr>
              <tr><td className="py-2 text-[var(--accent-blue)]">Stage 1</td><td>分通道独立 IForest</td><td className="font-mono">0.5381</td><td className="text-[var(--accent-green)]">+79.6%</td></tr>
              <tr><td className="py-2 font-medium">Stage 2</td><td className="font-medium">IF + 规则融合</td><td className="font-mono font-bold">0.5683</td><td className="text-[var(--accent-green)] font-medium">+89.7%</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

// ==========================================
// 8. Explanation 视图
// ==========================================

// Fallback mock 数据（模块顶层，避免渲染期间调用 impure 函数）
const MOCK_FALLBACK = (() => {
  // 确定性伪随机（mulberry32）
  let s = 42;
  const rand = () => { s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const mockData = Array.from({ length: 100 }, (_, i) => Math.sin(i / 5) * 10 + (rand() * 2));
  const mockAnomalies = mockData.map((_, i) => i === 45 || i === 46 || i === 78);
  const mockTimes = Array.from({ length: 100 }, (_, i) => `14:${Math.floor(i / 60).toString().padStart(2, '0')}:${(i % 60).toString().padStart(2, '0')}`);
  return { mockData, mockAnomalies, mockTimes };
})();

const ExplanationView: React.FC<{ channelId?: string; anomalyData?: AnomalyDataContext; onBack: () => void }> = ({ channelId = 'CADC0874', anomalyData, onBack }) => {
  const chLabel = CHANNEL_MAP[channelId] || channelId;
  const score = anomalyData?.score || 0;

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [waveformData, setWaveformData] = useState<WaveformData | null>(null);
  const [detailData, setDetailData] = useState<DetailData | null>(null);

  // Fallback mock 数据

  // 从后端获取真实数据
  useEffect(() => {
    const fetchData = async () => {
      if (!channelId || !anomalyData?.seg) {
        setLoading(false);
        return;
      }
      try {
        const [waveRes, detailRes] = await Promise.all([
          fetch(`/api/explanation/waveform?channel=${channelId}&segment=${anomalyData.seg}`),
          fetch(`/api/explanation/detail?channel=${channelId}&segment=${anomalyData.seg}`)
        ]);
        if (waveRes.ok) {
          const waveData = await waveRes.json();
          if (!waveData.error) setWaveformData(waveData);
        }
        if (detailRes.ok) {
          const detData = await detailRes.json();
          if (!detData.error) setDetailData(detData);
        }
      } catch (e) {
        console.error('Explanation fetch error:', e);
        setError('无法连接后端服务');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [channelId, anomalyData?.seg]);

  // 使用真实数据或 fallback 到 mock
  const displayData = waveformData?.values || MOCK_FALLBACK.mockData;
  const displayTimes = waveformData?.timestamps || MOCK_FALLBACK.mockTimes;
  const displayAnomalies = waveformData?.anomaly_mask?.map((v: number) => v === 1) || MOCK_FALLBACK.mockAnomalies;

  // 诊断数据
  const rawSections = detailData?.sections || {};
  const sources = detailData?.sources || [];
  const urgency = detailData?.urgency || '高优先级';

  // 检查 sections 是否是乱码（key 不包含正常中文字符）
  const isGarbled = Object.keys(rawSections).some(k => !/[一-鿿]/.test(k));
  const sections = isGarbled ? {} : rawSections;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* 顶部导航 */}
      <div className="flex items-center space-x-4 pb-4 border-b border-[var(--border-subtle)]">
        <button onClick={onBack} className="text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)] bg-[var(--bg-card)] px-2 py-1 border border-[var(--border-default)] rounded transition-colors">
          ← 返回
        </button>
        <h2 className="text-base font-semibold text-[var(--text-primary)]">
          深度诊断 | 段 #{anomalyData?.seg || '—'} · {chLabel}
        </h2>
        {loading && <span className="text-xs text-[var(--text-muted)]">加载中...</span>}
        {error && <span className="text-xs text-[var(--accent-red)]">{error}</span>}
      </div>

      {/* 示波器 */}
      <div className="card card-alert-red">
        <div className="text-xs font-medium text-[var(--accent-red)] mb-2 uppercase tracking-wider">
          示波器波形
          {waveformData && <span className="ml-2 text-[var(--text-muted)]">(真实数据)</span>}
          {!waveformData && !loading && <span className="ml-2 text-[var(--text-muted)]">(模拟数据)</span>}
        </div>
        <div className="h-72 bg-[var(--bg-primary)] border border-[var(--border-default)] rounded relative overflow-hidden" style={{ minHeight: 288 }}>
          <OscilloscopePlotly data={displayData} timestamps={displayTimes} anomalies={displayAnomalies} />
          <div className="absolute bottom-2 right-2 flex space-x-4 text-[10px] font-mono text-[var(--text-muted)] bg-[var(--bg-card)]/90 px-2 py-1 rounded pointer-events-none">
            <div><span className="text-[var(--accent-blue)]">CH:</span> {channelId}</div>
            <div><span className="text-[var(--accent-yellow)]">LIMIT:</span> ±3σ</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* AI 诊断 */}
          <div className="card">
            <div className="text-sm font-medium text-[var(--text-primary)] mb-3 pb-2 border-b border-[var(--border-subtle)]">
              AI 结构化诊断
              {detailData && <span className="ml-2 text-xs text-[var(--accent-green)]">(RAG生成)</span>}
            </div>

            <div className="grid grid-cols-3 gap-3 text-sm bg-[var(--bg-primary)] p-3 rounded border border-[var(--border-subtle)] mb-4">
              <div>
                <div className="text-xs text-[var(--text-muted)] mb-0.5">通道定位</div>
                <div className="font-medium">{detailData?.channel_label || chLabel}</div>
              </div>
              <div>
                <div className="text-xs text-[var(--text-muted)] mb-0.5">异常分</div>
                <div className="font-mono font-bold text-[var(--accent-red)]">{score.toFixed(3)}</div>
              </div>
              <div>
                <div className="text-xs text-[var(--text-muted)] mb-0.5">紧急程度</div>
                <div className="font-medium text-[var(--accent-orange)]">{urgency}</div>
              </div>
            </div>

            <div className="space-y-4 text-sm prose prose-sm max-w-none prose-headings:text-[var(--text-primary)] prose-p:text-[var(--text-secondary)] prose-strong:text-[var(--text-primary)] prose-li:text-[var(--text-secondary)]">
              {sections['可能原因'] && (
                <div className="pb-3 border-b border-[var(--border-subtle)]">
                  <h4 className="font-medium text-[var(--text-primary)] mb-1">可能原因</h4>
                  <ReactMarkdown>{sections['可能原因']}</ReactMarkdown>
                </div>
              )}
              {sections['影响评估'] && (
                <div className="pb-3 border-b border-[var(--border-subtle)]">
                  <h4 className="font-medium text-[var(--text-primary)] mb-1">影响评估</h4>
                  <ReactMarkdown>{sections['影响评估']}</ReactMarkdown>
                </div>
              )}
              {sections['建议措施'] && (
                <div className="pb-3 border-b border-[var(--border-subtle)]">
                  <h4 className="font-medium text-[var(--text-primary)] mb-1">建议措施</h4>
                  <ReactMarkdown>{sections['建议措施']}</ReactMarkdown>
                </div>
              )}
              {sections['结论'] && (
                <div className="pb-3 border-b border-[var(--border-subtle)]">
                  <h4 className="font-medium text-[var(--text-primary)] mb-1">诊断结论</h4>
                  <ReactMarkdown>{sections['结论']}</ReactMarkdown>
                </div>
              )}
              {!sections['可能原因'] && !sections['结论'] && detailData?.raw_explanation && (
                <div>
                  <h4 className="font-medium text-[var(--text-primary)] mb-1">完整诊断</h4>
                  <ReactMarkdown>{detailData.raw_explanation}</ReactMarkdown>
                </div>
              )}
              {!sections['可能原因'] && !sections['结论'] && !detailData?.raw_explanation && (
                <div>
                  <p className="text-[var(--text-muted)]">暂无诊断数据，请确认后端服务已启动并生成 RAG 结果。</p>
                </div>
              )}
            </div>
          </div>

          <TerminalComponent channelId={channelId} />
        </div>

        {/* 知识溯源 */}
        <div className="card">
          <div className="text-sm font-medium text-[var(--text-primary)] mb-3 pb-2 border-b border-[var(--border-subtle)]">
            知识溯源
          </div>
          <div className="space-y-2 text-sm">
            {sources.length > 0 ? (
              sources.map((src: SourceItem, idx: number) => (
                <div
                  key={idx}
                  className="p-2.5 bg-[var(--bg-primary)] border-l-2 border-[var(--accent-green)] rounded-r hover:bg-[var(--bg-card-hover)] transition-colors cursor-pointer"
                  title={src.local_path || ''}
                >
                  <div className="font-medium">{src.filename}</div>
                  <div className="flex justify-between text-xs text-[var(--text-muted)] mt-1">
                    <span>{src.page || '—'}</span>
                    <span className="font-mono text-[var(--accent-green)]">{(src.score || 0).toFixed(3)}</span>
                  </div>
                </div>
              ))
            ) : (
              <>
                <div className="p-2.5 bg-[var(--bg-primary)] border-l-2 border-[var(--accent-green)] rounded-r">
                  <div className="font-medium text-[var(--text-muted)]">暂无溯源数据</div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// ==========================================
// 9. 应用主框架
// ==========================================

export default function App() {
  const [currentPage, setCurrentPage] = useState<string>('dashboard');
  const [targetContext, setTargetContext] = useState<{ channelId?: string; anomalyData?: AnomalyDataContext }>({});

  const handleNavigate = (page: string, channelId?: string, anomalyData?: AnomalyDataContext) => {
    setTargetContext({ channelId, anomalyData });
    setCurrentPage(page);
  };

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] flex flex-col antialiased select-none">
      {/* 顶部状态栏 */}
      <header className="h-12 border-b border-[var(--border-default)] bg-[var(--bg-card)] px-4 flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-3">
          <div className="w-6 h-6 rounded border border-[var(--accent-blue)] flex items-center justify-center text-[var(--accent-blue)] font-bold text-xs">Ω</div>
          <h1 className="text-sm font-semibold tracking-wide">
            OPS-SAT <span className="text-[var(--accent-blue)]">遥测诊断系统</span>
          </h1>
          <span className="w-px h-4 bg-[var(--border-default)]" />
          <div className="flex items-center space-x-1.5 text-xs text-[var(--accent-green)]">
            <span className="status-dot status-dot-green" />
            <span>RAG 引擎在线</span>
          </div>
        </div>
      </header>

      {/* 主体 */}
      <div className="flex-1 flex overflow-hidden">
        {/* 侧边导航 */}
        <aside className="w-14 border-r border-[var(--border-default)] bg-[var(--bg-card)] flex flex-col items-center py-4 space-y-2 shrink-0">
          <button
            onClick={() => handleNavigate('dashboard')}
            className={`p-2.5 rounded transition-colors ${currentPage === 'dashboard' ? 'text-[var(--accent-blue)] bg-[var(--accent-blue)]/10' : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}`}
            title="实时告警中心"
          >
            <Activity size={18} />
          </button>
          <button
            onClick={() => handleNavigate('detection')}
            className={`p-2.5 rounded transition-colors ${currentPage === 'detection' ? 'text-[var(--accent-blue)] bg-[var(--accent-blue)]/10' : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}`}
            title="算法与配置"
          >
            <Sliders size={18} />
          </button>
        </aside>

        {/* 内容区 */}
        <main className="flex-1 p-6 overflow-y-auto max-w-[1600px] mx-auto w-full">
          {currentPage === 'dashboard' && <DashboardView onNavigate={handleNavigate} />}
          {currentPage === 'detection' && <DetectionView />}
          {currentPage === 'explanation' && <ExplanationView channelId={targetContext.channelId} anomalyData={targetContext.anomalyData} onBack={() => handleNavigate('dashboard')} />}
        </main>
      </div>

      {/* 底部 */}
      <footer className="h-8 bg-[var(--bg-card)] border-t border-[var(--border-default)] px-4 flex items-center justify-between text-xs text-[var(--text-muted)] shrink-0">
        <div className="flex items-center space-x-2">
          <span className="text-[var(--accent-blue)] font-medium">OPS-SAT v2.0</span>
          <span>·</span>
          <span>ESA MISSION CONTROL</span>
        </div>
        <div className="flex items-center space-x-1">
          <Clock size={10} className="text-[var(--text-muted)] mr-1" />
          <span>UTC 2026-06-03</span>
        </div>
      </footer>
    </div>
  );
}
