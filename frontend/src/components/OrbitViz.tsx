import { useEffect, useRef } from 'react';
import type { AlertItem } from '../api/types';
import { sevColor } from './severity';

/** 确定性伪随机 */
function mulberry32(seed: number) {
  let s = seed;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Blip { theta: number; color: string; phase: number; }

/**
 * 轨道态势画布:星空 + 地球弧 + 轨道环 + 卫星(拖尾) + 异常事件脉冲
 * 数据驱动:blips 来自真实告警队列(severity 配色,位置由 segment id 哈希确定)
 */
export default function OrbitViz({ alerts }: { alerts: AlertItem[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const alertsRef = useRef<AlertItem[]>(alerts);
  alertsRef.current = alerts;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let w = 0, h = 0, raf = 0;
    const dpr = Math.min(2, window.devicePixelRatio || 1);

    const resize = () => {
      const rect = canvas.parentElement!.getBoundingClientRect();
      w = rect.width; h = rect.height;
      canvas.width = w * dpr; canvas.height = h * dpr;
      canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas.parentElement!);

    // 星星(固定)
    const rand = mulberry32(20260603);
    const stars = Array.from({ length: 130 }, () => ({
      x: rand(), y: rand(), r: rand() * 1.1 + 0.2, tw: rand() * Math.PI * 2, sp: 0.5 + rand(),
    }));

    const t0 = performance.now();

    const draw = (now: number) => {
      const t = (now - t0) / 1000;
      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const horizonY = h * 2.0;          // 地球圆心在画面下方
      const earthR = h * 1.55;
      const orbitR = h * 1.78;           // 轨道半径 > 地球半径,弧线穿过画面中部

      // --- 星空 ---
      for (const s of stars) {
        const a = 0.25 + 0.55 * Math.abs(Math.sin(t * s.sp + s.tw));
        ctx.globalAlpha = a;
        ctx.fillStyle = '#9fd8e8';
        ctx.beginPath();
        ctx.arc(s.x * w, s.y * h, s.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // --- 地球(大气辉光 + 球体 + 晨昏线感) ---
      const atm = ctx.createRadialGradient(cx, horizonY, earthR * 0.98, cx, horizonY, earthR * 1.12);
      atm.addColorStop(0, 'rgba(34,211,238,0.28)');
      atm.addColorStop(0.4, 'rgba(34,211,238,0.08)');
      atm.addColorStop(1, 'transparent');
      ctx.fillStyle = atm;
      ctx.beginPath(); ctx.arc(cx, horizonY, earthR * 1.12, 0, Math.PI * 2); ctx.fill();

      const body = ctx.createRadialGradient(cx - earthR * 0.3, horizonY - earthR * 0.3, earthR * 0.1, cx, horizonY, earthR);
      body.addColorStop(0, '#0d2436');
      body.addColorStop(0.7, '#061220');
      body.addColorStop(1, '#040a14');
      ctx.fillStyle = body;
      ctx.beginPath(); ctx.arc(cx, horizonY, earthR, 0, Math.PI * 2); ctx.fill();

      ctx.strokeStyle = 'rgba(34,211,238,0.35)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(cx, horizonY, earthR, Math.PI * 1.15, Math.PI * 1.85); ctx.stroke();

      // --- 轨道环 ---
      ctx.strokeStyle = 'rgba(94,234,212,0.18)';
      ctx.setLineDash([3, 6]);
      ctx.beginPath(); ctx.arc(cx, horizonY, orbitR, Math.PI, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);

      // --- 异常事件 blips(告警哈希定位在轨道弧上) ---
      const list = alertsRef.current.slice(0, 40);
      const blips: Blip[] = list.map((a) => {
        let hash = 0;
        for (let i = 0; i < a.segment.length; i++) hash = (hash * 31 + a.segment.charCodeAt(i)) | 0;
        return {
          theta: Math.PI + (Math.abs(hash) % 1000) / 1000 * Math.PI,
          color: sevColor(a.severity),
          phase: (Math.abs(hash) % 628) / 100,
        };
      });
      for (const b of blips) {
        const bx = cx + Math.cos(b.theta) * orbitR;
        const by = horizonY + Math.sin(b.theta) * orbitR;
        const pulse = (t * 0.9 + b.phase) % 1.6;
        // 扩散脉冲环
        ctx.globalAlpha = Math.max(0, 0.5 - pulse * 0.31);
        ctx.strokeStyle = b.color;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(bx, by, 3 + pulse * 14, 0, Math.PI * 2); ctx.stroke();
        // 核心点
        ctx.globalAlpha = 0.95;
        ctx.fillStyle = b.color;
        ctx.beginPath(); ctx.arc(bx, by, 2.2, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;

      // --- 卫星(顺轨道飞行 + 拖尾) ---
      const satTheta = Math.PI + ((t * 0.05) % 1) * Math.PI;
      // 拖尾
      for (let i = 0; i < 26; i++) {
        const tt = satTheta - i * 0.008;
        const tx = cx + Math.cos(tt) * orbitR;
        const ty = horizonY + Math.sin(tt) * orbitR;
        ctx.globalAlpha = 0.35 * (1 - i / 26);
        ctx.fillStyle = '#22d3ee';
        ctx.beginPath(); ctx.arc(tx, ty, 1.6 - i * 0.04, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
      const sx = cx + Math.cos(satTheta) * orbitR;
      const sy = horizonY + Math.sin(satTheta) * orbitR;
      // 卫星本体:菱形 + 十字线
      ctx.save();
      ctx.translate(sx, sy);
      ctx.rotate(satTheta + Math.PI / 2);
      ctx.shadowColor = '#22d3ee'; ctx.shadowBlur = 12;
      ctx.fillStyle = '#a5f3fc';
      ctx.beginPath();
      ctx.moveTo(0, -5); ctx.lineTo(3.5, 0); ctx.lineTo(0, 5); ctx.lineTo(-3.5, 0);
      ctx.closePath(); ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = 'rgba(34,211,238,0.6)';
      ctx.beginPath(); ctx.moveTo(-9, 0); ctx.lineTo(-5, 0); ctx.moveTo(5, 0); ctx.lineTo(9, 0); ctx.stroke();
      ctx.restore();
      // 扫描线(卫星→地面)
      ctx.globalAlpha = 0.10 + 0.05 * Math.sin(t * 3);
      ctx.strokeStyle = '#22d3ee';
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(cx, horizonY - earthR); ctx.stroke();
      ctx.globalAlpha = 1;

      // --- HUD 角标文字(避开面板标题栏) ---
      ctx.font = '9px "IBM Plex Mono", monospace';
      ctx.fillStyle = 'rgba(127,168,166,0.75)';
      ctx.fillText('ORBIT TRACK · LEO 550KM', 14, 46);
      const prog = (((t * 0.05) % 1) * 100).toFixed(1);
      ctx.fillText(`SAT-1  REV ${prog}%  ALT 550.2KM  V 7.59KM/S`, 14, 60);
      const events = `TRACKED EVENTS: ${blips.length}`;
      ctx.fillText(events, w - ctx.measureText(events).width - 14, 46);

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, []);

  return <canvas ref={canvasRef} className="block" />;
}
