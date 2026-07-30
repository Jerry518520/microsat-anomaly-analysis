import { useEffect, useRef, useState } from 'react';
import { Send, Bot, User } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { api } from '../api/client';

interface Msg {
  role: 'user' | 'assistant';
  text: string;
  sources?: { filename: string; score: number }[];
  failed?: boolean;
}

/** RAG 自由问答终端 */
export default function RagChat({ channel, segment }: { channel: string; segment: string }) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, busy]);

  const send = async () => {
    const q = input.trim();
    if (!q || busy) return;
    setInput('');
    setMessages(m => [...m, { role: 'user', text: q }]);
    setBusy(true);
    try {
      const res = await api.explanation.query(q, channel, segment);
      if (res.error) {
        setMessages(m => [...m, { role: 'assistant', text: res.answer || 'RAG 引擎暂不可用。', failed: true }]);
      } else {
        setMessages(m => [...m, { role: 'assistant', text: res.answer, sources: res.sources }]);
      }
    } catch {
      setMessages(m => [...m, { role: 'assistant', text: 'RAG 引擎暂不可用(知识库索引未加载或后端未启动)。', failed: true }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel flex flex-col">
      <div className="panel-head"><Bot size={13} /> RAG 溯源问答</div>
      <div ref={listRef} className="flex-1 min-h-[180px] max-h-[320px] overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && !busy && (
          <div className="text-xs tx-3 leading-relaxed">
            针对当前异常段向知识库自由提问,例如:
            <div className="mt-2 space-y-1.5">
              {['这个异常对姿态控制有什么危害?', '磁力计出现跳变的常见原因有哪些?', '应该采取什么处置措施?'].map(s => (
                <button
                  key={s}
                  onClick={() => setInput(s)}
                  className="block text-left text-[11px] font-mono text-[#38BDF8] bg-[rgba(56,189,248,0.07)] border border-[rgba(56,189,248,0.2)] rounded px-2 py-1 hover:bg-[rgba(56,189,248,0.14)] transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex gap-2 ${m.role === 'user' ? 'justify-end' : ''}`}>
            {m.role === 'assistant' && <Bot size={14} className="text-[#38BDF8] mt-0.5 shrink-0" />}
            <div
              className="max-w-[85%] rounded-lg px-3 py-2 text-xs leading-relaxed"
              style={
                m.role === 'user'
                  ? { background: 'rgba(56,189,248,0.12)', color: 'var(--tx-1)', border: '1px solid rgba(56,189,248,0.25)' }
                  : m.failed
                    ? { background: 'rgba(245,158,11,0.08)', color: '#F59E0B', border: '1px solid rgba(245,158,11,0.25)' }
                    : { background: 'rgba(148,163,184,0.07)', color: 'var(--tx-2)', border: '1px solid rgba(148,163,184,0.12)' }
              }
            >
              <div className="md-body"><ReactMarkdown>{m.text}</ReactMarkdown></div>
              {m.sources && m.sources.length > 0 && (
                <div className="mt-2 pt-2 border-t border-[rgba(148,163,184,0.12)] space-y-0.5">
                  {m.sources.map((s, j) => (
                    <div key={j} className="flex justify-between text-[10px] font-mono tx-3">
                      <span className="truncate mr-2">{s.filename}</span>
                      <span className="text-[#2DD4BF]">{s.score.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            {m.role === 'user' && <User size={14} className="tx-3 mt-0.5 shrink-0" />}
          </div>
        ))}
        {busy && (
          <div className="flex gap-2 items-center text-xs tx-3">
            <Bot size={14} className="text-[#38BDF8]" />
            <span className="font-mono">检索知识库中<span className="animate-pulse">…</span></span>
          </div>
        )}
      </div>
      <div className="p-3 border-t border-[rgba(148,163,184,0.1)]">
        <div className="input-line flex items-center gap-2 px-3 py-2">
          <span className="text-xs font-mono text-[#38BDF8]">&gt;</span>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && send()}
            placeholder="输入追问指令…"
            className="flex-1 bg-transparent text-xs tx-1 focus:outline-none placeholder-[var(--tx-3)]"
          />
          <button onClick={send} disabled={busy} className="tx-3 hover:text-[#38BDF8] transition-colors disabled:opacity-40">
            <Send size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
