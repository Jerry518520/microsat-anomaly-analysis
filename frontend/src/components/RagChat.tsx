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

/** RAG 溯源问答终端 */
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
    <div className="deck-panel flex flex-col min-h-0">
      <div className="deck-head"><span className="tick" /> RAG 溯源问答</div>
      <div ref={listRef} className="flex-1 min-h-[150px] max-h-[300px] overflow-y-auto p-3 space-y-2.5">
        {messages.length === 0 && !busy && (
          <div className="text-[11px] tx-3 leading-relaxed">
            针对当前异常段向知识库自由提问:
            <div className="mt-2 space-y-1.5">
              {['这个异常对姿态控制有什么危害?', '磁力计出现跳变的常见原因有哪些?', '应该采取什么处置措施?'].map(s => (
                <button key={s} onClick={() => setInput(s)}
                  className="block text-left text-[10px] font-mono text-[#22d3ee] bg-[rgba(34,211,238,0.06)] border border-[rgba(34,211,238,0.2)] rounded px-2 py-1 hover:bg-[rgba(34,211,238,0.14)] transition-colors">
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex gap-2 ${m.role === 'user' ? 'justify-end' : ''}`}>
            {m.role === 'assistant' && <Bot size={13} className="text-[#22d3ee] mt-0.5 shrink-0" />}
            <div className="max-w-[85%] rounded px-3 py-2 text-xs leading-relaxed"
              style={
                m.role === 'user'
                  ? { background: 'rgba(34,211,238,0.1)', color: 'var(--tx-1)', border: '1px solid rgba(34,211,238,0.25)' }
                  : m.failed
                    ? { background: 'rgba(251,191,36,0.07)', color: '#fbbf24', border: '1px solid rgba(251,191,36,0.25)' }
                    : { background: 'rgba(94,234,212,0.05)', color: 'var(--tx-2)', border: '1px solid rgba(94,234,212,0.12)' }
              }>
              <div className="md-body"><ReactMarkdown>{m.text}</ReactMarkdown></div>
              {m.sources && m.sources.length > 0 && (
                <div className="mt-2 pt-2 border-t border-[rgba(94,234,212,0.12)] space-y-0.5">
                  {m.sources.map((s, j) => (
                    <div key={j} className="flex justify-between text-[9px] font-mono tx-3">
                      <span className="truncate mr-2">{s.filename}</span>
                      <span className="text-[#2dd4a7]">{s.score.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            {m.role === 'user' && <User size={13} className="tx-3 mt-0.5 shrink-0" />}
          </div>
        ))}
        {busy && (
          <div className="flex gap-2 items-center text-[11px] tx-3 font-mono">
            <Bot size={13} className="text-[#22d3ee]" />
            检索知识库中<span className="animate-pulse">…</span>
          </div>
        )}
      </div>
      <div className="p-2.5 border-t border-[rgba(94,234,212,0.1)]">
        <div className="cmd-input flex items-center gap-2 px-3 py-2">
          <span className="text-[11px] font-mono text-[#22d3ee]">&gt;</span>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && send()}
            placeholder="输入追问指令…"
            className="flex-1 bg-transparent text-xs tx-1 focus:outline-none placeholder-[var(--tx-3)]"
          />
          <button onClick={send} disabled={busy} className="tx-3 hover:text-[#22d3ee] transition-colors disabled:opacity-40">
            <Send size={13} />
          </button>
        </div>
      </div>
    </div>
  );
}
