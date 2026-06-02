import React, { useState, useEffect, useRef, useMemo } from 'react';
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
import createPlotlyComponent from 'react-plotly.js/factory';

// 临时方案：用原生 div 包裹 Plotly 图表
const Plot = ({ data, layout, config, style, useResizeHandler }: any) => {
  const plotRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (plotRef.current && data) {
      Plotly.newPlot(plotRef.current, data, layout, config);
      return () => {
        if (plotRef.current) {
          Plotly.purge(plotRef.current);
        }
      };
    }
  }, [data, layout, config]);

  return <div ref={plotRef} style={style} />;
};

// ==========================================
// 1. 业务常量与状态判定模型
// ==========================================

const CHANNEL_MAP: Record<string, string> = {
  "CADC0872": "磁力计 X轴", "CADC0873": "磁力计 Y轴", "CADC0874": "磁力计 Z轴",
  "CADC0884": "光电二极管 1", "CADC0886": "光电二极管 2", "CADC0888": "光电二极管 3",
  "CADC0890": "光电二极管 4", "CADC0892": "光电二极管 5", "CADC0894": "光电二极管 6",
};

const NO_ANOMALY_CHANNELS = ["CADC0884"];

type SystemStatus = 'NOMINAL' | 'WARNING' | 'CRITICAL' | 'OFFLINE';

const getStatusByScore = (ch: string, score: number): SystemStatus => {
  if (NO_ANOMALY_CHANNELS.includes(ch)) return 'OFFLINE';
  if (score > 0.15) return 'CRITICAL';
  if (score > 0.05) return 'WARNING';
  return 'NOMINAL';
};

// ==========================================
// 2. 共享基础视觉组件 (HUD)
// ==========================================

const StatusDot: React.FC<{ status: SystemStatus; animate?: boolean }> = ({ status, animate = true }) => {
  const config = {
    NOMINAL: { bg: 'bg-[#34D399]', shadow: 'shadow-[#34D399]/50', pulse: 'animate-pulse' },
    WARNING: { bg: 'bg-[#FBBF24]', shadow: 'shadow-[#FBBF24]/50', pulse: '' },
    CRITICAL: { bg: 'bg-[#FF2D55]', shadow: 'shadow-[#FF2D55]/50', pulse: 'animate-ping' },
    OFFLINE: { bg: 'bg-[#3A4558]', shadow: 'shadow-transparent', pulse: '' }
  };
  const current = config[status];

  return (
    <div className="relative flex items-center justify-center w-3 h-3">
      {status === 'CRITICAL' && <span className={`absolute inline-flex h-full w-full rounded-full opacity-75 bg-[#FF2D55] ${current.pulse}`} />}
      <span className={`relative inline-flex rounded-full h-2 w-2 ${current.bg} shadow-[0_0_6px_2px] ${current.shadow} ${status === 'NOMINAL' && animate ? 'animate-[pulse_3s_infinite_ease-in-out]' : ''}`} />
    </div>
  );
};

const HudCard: React.FC<{ children: React.ReactNode; className?: string; status?: SystemStatus }> = ({ children, className = '', status }) => {
  const isCritical = status === 'CRITICAL';
  return (
    <div className={`glass-panel hud-frame holo-container p-4 rounded-sm ${isCritical ? 'critical-glow-pulse' : ''} ${className}`}>
      <div className="hud-inner-corners" />
      {children}
    </div>
  );
};

// 交互终端组件 (页面C 使用)
const TerminalComponent: React.FC<{ channelId: string }> = ({ channelId }) => {
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'assistant'; text: string }>>([
    { role: 'user', text: `针对 ${channelId} 的异常，对姿态控制有什么具体危害？` },
    { role: 'assistant', text: '> RAG ENGINE: 根据卫星大系统ADCS手册第112页指出：磁力计的异常突刺会导致姿态解算矩阵无法正常收敛。当系统连续检测到单轴发散超3σ限界，建议立即切换至外部高精度星敏感器（Star Tracker）备份链路。' }
  ]);
  const [input, setInput] = useState('');
  const terminalEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;
    const userMsg = input;
    setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setInput('');
    setTimeout(() => {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        text: `> TELEMETRY COMMAND EXECUTED // 针对追问正在调取大模型与RAG混合解析：分析指出该卫星姿态角速度目前仍处于星地安全包络内，请保持监测。` 
      }]);
    }, 800);
  };

  return (
    <HudCard className="bg-[#05080E]/95 border border-[#00E5FF]/20">
      <div className="flex items-center space-x-2 text-xs font-rajdhani text-[#00E5FF] mb-3 font-bold border-b border-[#3A4558]/30 pb-2">
        <TermIcon size={14} /><span>INTELLIGENT INTERACTIVE TERMINAL // 航天器交互诊断终端</span>
      </div>
      <div className="h-40 overflow-y-auto font-ibm-mono text-xs space-y-3 mb-4 pr-2 scrollbar-thin scrollbar-thumb-[#3A4558]">
        {messages.map((m, i) => (
          <div key={i} className={`flex items-start space-x-2 ${m.role === 'user' ? 'text-[#00E5FF]' : 'text-[#7B8CA8]'}`}>
            <span className="text-[11px] mt-0.5 select-none">{m.role === 'user' ? '>' : '●'}</span>
            <p className="leading-relaxed bg-[#080C14]/40 p-1.5 rounded-sm border border-[#3A4558]/10 w-full whitespace-pre-wrap">{m.text}</p>
          </div>
        ))}
        <div ref={terminalEndRef} />
      </div>
      <div className="flex items-center space-x-2 border border-[#3A4558] bg-[#080C14] rounded-sm px-2 py-1 focus-within:border-[#00E5FF] transition-all">
        <span className="text-xs font-ibm-mono text-[#00E5FF] select-none">&gt;</span>
        <input 
          type="text" 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="输入遥测排故溯源追问指令..." 
          className="flex-1 bg-transparent border-none text-xs text-[#E5E9F0] focus:outline-none placeholder-[#3A4558] font-ibm-mono"
        />
        <button onClick={handleSend} className="text-[#7B8CA8] hover:text-[#00E5FF] transition-colors"><Send size={14} /></button>
      </div>
    </HudCard>
  );
};

// ==========================================
// 3. 高级可视化组件: Plotly 示波器
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
  const upperLimit = mean + 3 * std;
  const lowerLimit = mean - 3 * std;

  return (
    <div className="w-full h-full relative">
      <Plot
        data={[
          {
            x: timestamps, y: data, type: 'scatter', mode: 'lines',
            line: { color: '#00E5FF', width: 1.5, shape: 'spline' },
            name: '遥测值', hoverinfo: 'x+y',
          },
          {
            x: anomalyX, y: anomalyY, type: 'scatter', mode: 'markers',
            marker: { color: '#FF2D55', size: 10, symbol: 'x', line: { color: '#FF2D55', width: 2 } },
            name: '异常突变', hoverinfo: 'x+y',
          }
        ]}
        layout={{
          autosize: true,
          paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
          margin: { l: 40, r: 20, t: 20, b: 30 },
          font: { family: 'IBM Plex Mono', color: '#7B8CA8', size: 10 },
          xaxis: { gridcolor: 'rgba(58, 69, 88, 0.2)', zerolinecolor: 'rgba(58, 69, 88, 0.5)', showline: true, linecolor: '#3A4558' },
          yaxis: { gridcolor: 'rgba(58, 69, 88, 0.2)', zerolinecolor: 'rgba(58, 69, 88, 0.5)', showline: true, linecolor: '#3A4558' },
          showlegend: false,
          hovermode: 'x unified',
          shapes: [
            { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: upperLimit, y1: upperLimit, yref: 'y', line: { color: '#FBBF24', width: 1, dash: 'dashdot' } },
            { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: lowerLimit, y1: lowerLimit, yref: 'y', line: { color: '#FBBF24', width: 1, dash: 'dashdot' } }
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
// 4. 路由页面视图
// ==========================================

const DashboardView: React.FC<{ onNavigate: (page: string, channelId?: string, anomalyData?: any) => void }> = ({ onNavigate }) => {
  const queue = [
    { seg: '1247', ch: 'CADC0874', score: 0.234, summary: "AI: 磁力计Z轴出现间歇性尖峰异常，疑似太阳风暴干扰导致高频噪声畸变。" },
    { seg: '0892', ch: 'CADC0890', score: 0.156, summary: "AI: 光电管4出现周期性衰减，可能为物理传感器组件表面轻微老化。" },
    { seg: '0455', ch: 'CADC0886', score: 0.089, summary: "AI: 光电管2出现轻微读数漂移，仍在当前可接受的滤波动态基线范围内。" }
  ];

  return (
    <div className="space-y-6 animate-[fadeIn_0.5s_ease-out]">
      <div className="flex items-center justify-between">
        <h2 className="font-rajdhani text-xl font-bold tracking-wider text-[#00E5FF]">REAL-TIME ALARM CENTER // 实时告警中心</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <HudCard><div className="text-xs text-[#7B8CA8] font-ibm-mono uppercase">系统异常率</div><div className="text-2xl font-bold font-rajdhani text-[#E5E9F0] my-1 font-ibm-mono">1.2%</div><div className="text-xs font-ibm-mono text-[#FF6B35]">452行/45K行</div></HudCard>
        <HudCard status="WARNING"><div className="text-xs text-[#7B8CA8] font-ibm-mono uppercase">当前告警队列 (Score &gt; 0.05)</div><div className="text-2xl font-bold font-rajdhani text-[#E5E9F0] my-1 font-ibm-mono">{queue.length} ACTIVE</div><div className="text-xs font-ibm-mono text-[#FF6B35]">共12条异常段</div></HudCard>
        <HudCard><div className="text-xs text-[#7B8CA8] font-ibm-mono uppercase">核心诊断 F1</div><div className="text-2xl font-bold font-rajdhani text-[#E5E9F0] my-1 font-ibm-mono">0.568</div><div className="text-xs font-ibm-mono text-[#34D399]">Stage 2 Fusion</div></HudCard>
        <HudCard><div className="text-xs text-[#7B8CA8] font-ibm-mono uppercase">遥测吞吐量</div><div className="text-2xl font-bold font-rajdhani text-[#E5E9F0] my-1 font-ibm-mono">45K ROWS</div><div className="text-xs font-ibm-mono text-[#34D399]">9 通道并发</div></HudCard>
      </div>

      <HudCard>
        <div className="text-xs font-rajdhani text-[#00E5FF] mb-3 font-bold tracking-wider">FAULT ALARM QUEUE // 故障告警队列</div>
        <div className="divide-y divide-[#3A4558]/30">
          {queue.map((q, i) => {
            const status = getStatusByScore(q.ch, q.score);
            const borderColor = status === 'CRITICAL' ? '#FF2D55' : status === 'WARNING' ? '#FBBF24' : '#00E5FF';
            
            return (
              <div key={i} className="py-3 first:pt-0 last:pb-0">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 mb-1">
                  <div className="flex items-center space-x-2 text-xs font-ibm-mono">
                    <span className="text-[#E5E9F0] font-bold">段 #{q.seg}</span><span className="text-[#3A4558]">·</span>
                    <span className="text-[#00E5FF]">{q.ch}</span><span className="text-[#3A4558]">·</span>
                    <span className="text-[#E5E9F0] font-sans">{CHANNEL_MAP[q.ch]}</span>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1.5 text-xs font-ibm-mono" style={{ color: borderColor }}>
                      <span className={`w-1.5 h-1.5 rounded-full ${status === 'CRITICAL' ? 'animate-pulse' : ''}`} style={{ backgroundColor: borderColor }}></span>
                      <span>SCORE: {q.score.toFixed(3)}</span>
                    </div>
                    <button 
                      onClick={() => onNavigate('explanation', q.ch, q)}
                      className="text-xs font-ibm-mono text-[#00E5FF] hover:text-white flex items-center bg-[#00E5FF]/10 hover:bg-[#00E5FF]/30 px-2 py-0.5 rounded-sm border border-[#00E5FF]/20 transition-all"
                    >
                      深度诊断分析 <ChevronRight size={12} className="ml-0.5" />
                    </button>
                  </div>
                </div>
                <p className="text-xs text-[#7B8CA8] pl-2 border-l border-[#3A4558] mt-1">🧠 <b>AI摘要:</b> {q.summary}</p>
              </div>
            );
          })}
        </div>
      </HudCard>
    </div>
  );
};

const DetectionView: React.FC = () => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-[fadeIn_0.5s_ease-out]">
      <div className="space-y-6">
        <HudCard>
          <div className="text-xs font-rajdhani text-[#00E5FF] mb-4 font-bold tracking-wider">RAG KNOWLEDGE BASE // 知识库配置</div>
          <div className="space-y-2 text-xs font-ibm-mono text-[#7B8CA8]">
            <div className="flex justify-between border-b border-[#3A4558]/20 pb-1"><span>挂载文档分布</span><span className="text-[#E5E9F0] font-bold">8 PDF + 3 MD + 1 HTML</span></div>
            <div className="flex justify-between border-t border-[#3A4558]/20 pt-2 mt-2"><span>FAISS CHUNKS</span><span className="text-[#00E5FF]">2,847 Blocks</span></div>
            <div className="flex justify-between"><span>嵌入向量引擎</span><span className="text-[#E5E9F0]">BGE-M3</span></div>
          </div>
          <button className="w-full mt-4 bg-[#00E5FF]/10 hover:bg-[#00E5FF]/20 border border-[#00E5FF]/30 text-[#00E5FF] text-xs font-ibm-mono py-2 rounded-sm transition-all flex items-center justify-center space-x-2">
            <RefreshCw size={12} /><span>全量重建 FAISS 索引</span>
          </button>
        </HudCard>
      </div>

      <div className="space-y-6">
        <HudCard>
          <div className="text-xs font-rajdhani text-[#00E5FF] mb-4 font-bold tracking-wider">EXPERIMENT PIPELINE // 实验演进路线 (F1)</div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-ibm-mono">
              <thead><tr className="border-b border-[#3A4558] text-[#7B8CA8] text-[11px]"><th className="py-2">版本代号</th><th>技术策略</th><th>段级 F1</th><th>提升</th></tr></thead>
              <tbody className="divide-y divide-[#3A4558]/20 text-[#E5E9F0]">
                <tr><td className="py-2 text-[#7B8CA8]">Stage 0</td><td>全局 IForest (c=0.2)</td><td>0.2996</td><td className="text-[#3A4558]">—</td></tr>
                <tr><td className="py-2 text-[#00E5FF]">Stage 1</td><td>分通道独立 IForest</td><td>0.5381</td><td className="text-[#34D399]">+79.6%</td></tr>
                <tr><td className="py-2 text-[#FF6B35] font-bold">Stage 2</td><td className="font-bold">IF + 规则融合</td><td className="font-bold">0.5683</td><td className="text-[#34D399]">+89.7%</td></tr>
              </tbody>
            </table>
          </div>
        </HudCard>
      </div>
    </div>
  );
};

const ExplanationView: React.FC<{ channelId?: string; anomalyData?: any; onBack: () => void }> = ({ channelId = 'CADC0874', anomalyData, onBack }) => {
  const chLabel = CHANNEL_MAP[channelId] || channelId;
  const score = anomalyData?.score || 0.234;

  // 构造模拟波形数据 (给 Plotly 使用)
  const mockData = Array.from({length: 100}, (_, i) => Math.sin(i / 5) * 10 + (Math.random() * 2));
  const mockAnomalies = mockData.map((_, i) => i === 45 || i === 46 || i === 78);
  const mockTimes = Array.from({length: 100}, (_, i) => `14:${Math.floor(i/60).toString().padStart(2,'0')}:${(i%60).toString().padStart(2,'0')}`);
  
  return (
    <div className="space-y-6 animate-[fadeIn_0.5s_ease-out]">
      <div className="flex items-center justify-between border-b border-[#3A4558]/30 pb-4">
        <div className="flex items-center space-x-4">
          <button onClick={onBack} className="text-xs font-ibm-mono text-[#7B8CA8] hover:text-[#00E5FF] transition-colors bg-[#111827] px-2 py-1 border border-[#3A4558]/60 rounded-sm">
            ◀ BACK
          </button>
          <h2 className="font-rajdhani text-lg font-bold tracking-widest text-[#FF2D55] uppercase">
            DEEP DIAGNOSIS | 段 #{anomalyData?.seg || '1247'} · {chLabel}
          </h2>
        </div>
      </div>

      <HudCard status="CRITICAL">
        <div className="text-xs font-rajdhani text-[#FF2D55] mb-2 font-bold tracking-wider">
          OSCILLOSCOPE WAVEFORM SCREEN // 交互式示波器阵列
        </div>
        <div className="h-72 bg-[#04070D] border border-[#3A4558]/40 rounded-sm relative overflow-hidden" style={{ minHeight: '288px' }}>
          <OscilloscopePlotly data={mockData} timestamps={mockTimes} anomalies={mockAnomalies} />
          
          <div className="absolute bottom-2 right-2 flex space-x-4 text-[10px] font-ibm-mono text-[#7B8CA8] bg-[#080C14]/90 px-2 py-1 border border-[#3A4558]/40 pointer-events-none">
            <div><span className="text-[#00E5FF]">CH:</span> {channelId}</div>
            <div><span className="text-[#FBBF24]">LIMIT:</span> ±3σ</div>
          </div>
        </div>
      </HudCard>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <HudCard>
            <div className="text-xs font-rajdhani text-[#00E5FF] mb-3 font-bold tracking-wider">AI STRUCTURED DIAGNOSIS // AI 智能诊断</div>
            
            <div className="grid grid-cols-3 gap-2 text-xs font-ibm-mono bg-[#080C14] p-3 border border-[#3A4558]/30 rounded-sm mb-4">
              <div><div className="text-[#7B8CA8] text-[10px]">通道定位</div><div className="text-[#E5E9F0] font-bold">{chLabel}</div></div>
              <div><div className="text-[#7B8CA8] text-[10px]">异常分</div><div className="text-[#FF2D55] font-bold">{score.toFixed(3)} (CRITICAL)</div></div>
              <div><div className="text-[#7B8CA8] text-[10px]">紧急程度</div><div className="text-[#FF6B35] font-bold">高优先级</div></div>
            </div>

            <div className="space-y-4 text-xs mt-4">
              <div className="border-b border-[#3A4558]/30 pb-3">
                <h4 className="font-bold text-[#E5E9F0] mb-1 font-ibm-mono">🔴 可能原因</h4>
                <p className="text-[#7B8CA8] leading-relaxed">近24小时内太阳风暴引起的高能粒子流诱发了剧烈的轨道静电磁场扰动，卫星磁力计感应线圈在Z轴平面产生高频感应电动势错觉，致使遥测发生离群多阶尖峰脉冲。</p>
              </div>
              <div className="border-b border-[#3A4558]/30 pb-3">
                <h4 className="font-bold text-[#E5E9F0] mb-1 font-ibm-mono">⚡ 影响评估</h4>
                <p className="text-[#7B8CA8] leading-relaxed">若在此不加干预或未能进行软件基线滤波对冲，将直接恶化反动量轮姿态确定系统（ADCS）的物理收敛环路收敛速度。</p>
              </div>
              <div>
                <h4 className="font-bold text-[#E5E9F0] mb-1 font-ibm-mono">✅ 建议措施</h4>
                <ol className="list-decimal list-inside text-[#7B8CA8] space-y-1 mt-1 pl-1">
                  <li>下发瞬态指令刷新 Z轴 磁力计卡尔曼滤波器的静态协方差参数。</li>
                  <li>对比当前星敏感器（Star Tracker）备份联合标定数据判定真实漂移量。</li>
                </ol>
              </div>
            </div>
          </HudCard>

          <TerminalComponent channelId={channelId} />
        </div>

        <HudCard>
          <div className="text-xs font-rajdhani text-[#00E5FF] mb-3 font-bold tracking-wider">KNOWLEDGE SOURCES // 知识溯源</div>
          <div className="space-y-3 font-ibm-mono text-xs">
            <div className="p-2.5 bg-[#080C14] border-l-2 border-[#34D399] rounded-r-sm hover:bg-[#1A2233] transition-colors cursor-pointer">
              <div className="text-[#E5E9F0] font-bold">ESA_ops_anomaly.pdf</div>
              <div className="flex justify-between text-[11px] text-[#7B8CA8] mt-1"><span>p.47 · 🔗</span><span className="text-[#34D399]">0.891</span></div>
            </div>
            <div className="p-2.5 bg-[#080C14] border-l-2 border-[#00E5FF] rounded-r-sm hover:bg-[#1A2233] transition-colors cursor-pointer">
              <div className="text-[#E5E9F0] font-bold">adcs_handbook.pdf</div>
              <div className="flex justify-between text-[11px] text-[#7B8CA8] mt-1"><span>p.112 · 🔗</span><span className="text-[#00E5FF]">0.702</span></div>
            </div>
          </div>
        </HudCard>
      </div>
    </div>
  );
};

// ==========================================
// 5. 应用主框架
// ==========================================

export default function App() {
  const [currentPage, setCurrentPage] = useState<string>('dashboard');
  const [targetContext, setTargetContext] = useState<{ channelId?: string, anomalyData?: any }>({});

  const handleNavigate = (page: string, channelId?: string, anomalyData?: any) => {
    setTargetContext({ channelId, anomalyData });
    setCurrentPage(page);
  };

  return (
    <div className="relative min-h-screen bg-[#080C14] text-[#E5E9F0] flex flex-col font-sans antialiased selection:bg-[#00E5FF]/30 select-none overflow-hidden">
      <div className="noise-overlay" />

      {/* 顶部系统状态栏 */}
      <header className="h-[52px] border-b border-[#3A4558]/40 bg-[#0C1220]/80 backdrop-blur-md px-4 flex items-center justify-between z-10 hud-frame shrink-0">
        <div className="hud-inner-corners" />
        <div className="flex items-center space-x-3">
          <div className="w-6 h-6 rounded-xs border border-[#00E5FF] flex items-center justify-center text-[#00E5FF] font-bold text-xs">Ω</div>
          <h1 className="font-rajdhani font-bold tracking-widest text-sm text-[#E5E9F0]">
            OPS-SAT <span className="text-[#00E5FF]">TELEMETRY DIAGNOSTICS</span>
          </h1>
          <span className="h-4 w-[1px] bg-[#3A4558]" />
          <div className="flex items-center space-x-1.5 bg-[#080C14] px-2 py-0.5 border border-[#3A4558]/60 rounded-sm">
            <StatusDot status="NOMINAL" />
            <span className="text-[10px] font-ibm-mono text-[#34D399]">RAG ENGINE: ONLINE</span>
          </div>
        </div>
      </header>

      {/* 中轴区域 */}
      <div className="flex-1 flex relative overflow-hidden">
        {/* 侧边导航 */}
        <aside className="w-[68px] border-r border-[#3A4558]/30 bg-[#0C1220]/40 flex flex-col items-center py-4 z-10 shrink-0">
          <nav className="flex flex-col items-center space-y-4 w-full">
            <button onClick={() => handleNavigate('dashboard')} className={`p-3 rounded-sm transition-all ${currentPage === 'dashboard' ? 'text-[#00E5FF] bg-[#00E5FF]/10 border border-[#00E5FF]/20' : 'text-[#7B8CA8] hover:text-[#00E5FF]'}`}><Activity size={20} /></button>
            <button onClick={() => handleNavigate('detection')} className={`p-3 rounded-sm transition-all ${currentPage === 'detection' ? 'text-[#00E5FF] bg-[#00E5FF]/10 border border-[#00E5FF]/20' : 'text-[#7B8CA8] hover:text-[#00E5FF]'}`}><Sliders size={20} /></button>
          </nav>
        </aside>

        {/* 路由内容区 */}
        <main className="flex-1 p-6 overflow-y-auto max-w-[1600px] mx-auto w-full">
          {currentPage === 'dashboard' && <DashboardView onNavigate={handleNavigate} />}
          {currentPage === 'detection' && <DetectionView />}
          {currentPage === 'explanation' && <ExplanationView channelId={targetContext.channelId} anomalyData={targetContext.anomalyData} onBack={() => handleNavigate('dashboard')} />}
        </main>
      </div>
      
      {/* 底部信息栏 */}
      <footer className="h-[36px] bg-[#04070D] border-t border-[#3A4558]/30 px-4 flex items-center justify-between text-[11px] font-ibm-mono text-[#7B8CA8] z-10 hud-frame shrink-0">
        <div className="hud-inner-corners" />
        <div className="flex items-center space-x-2">
          <span className="text-[#00E5FF] font-bold">OPS-SAT v2.0</span><span>·</span><span>ESA MISSION CONTROL INTERFACE</span>
        </div>
        <div className="flex items-center space-x-1">
          <Clock size={11} className="text-[#00E5FF] mr-1" />
          <span className="text-[#E5E9F0]">UTC 2026-06-02 23:28:02</span>
        </div>
      </footer>
    </div>
  );
}