import { useEffect, useRef } from 'react';
import * as satellite from 'satellite.js';
import type { AlertItem } from '../api/types';
import { sevColor } from './severity';

/**
 * 轨道态势画布 — 真实化原则:所有轨道数字均由 SGP4 从真实 TLE 推算,无任何编造参数。
 *
 * 数据来源与限制(保持诚实):
 * - TLE 为 OPS-SAT (NORAD 44878) 在轨期间的存档根数,历元 2023-09-27T12:41Z
 *   (来源: n2yo.com 页面 Wayback Machine 存档 20230927212719)。
 * - OPS-SAT 已于 2024-05-22 再入大气层(ESA / SatNOGS 记录),因此本视图是
 *   「历史轨道重放」: 模拟任务时钟 = TLE 历元 + 墙钟经过 × TIME_SCALE。
 * - 画面是轨道面示意投影: 弧上角度 = 真实纬度幅角(argument of latitude),
 *   运动速率与轨道周期真实;地球/星空/轨道环为示意,不代表真实比例与倾角。
 * - 异常事件 blip 的数量/颜色来自真实告警队列,弧上位置由 segment id 哈希
 *   确定(遥测数据本身不含位置信息,属于确定性布局而非真实发生位置)。
 */

const TLE1 = '1 44878U 19092F   23270.52867889  .00041164  00000-0  10235-2 0  9998';
const TLE2 = '2 44878  97.4708 100.9826 0008505 122.8882 237.3184 15.40146882209517';
const TLE_EPOCH_MS = Date.UTC(2023, 0, 1) + (270.52867889 - 1) * 86400_000; // 2023-09-27T12:41:18Z
const DECAY_NOTE = 'DECAYED 2024-05-22 · HISTORICAL REPLAY';

/** 时间倍率: 1 秒墙钟 = 60 秒任务时间(约 1.6 分钟重放一圈,周期本身真实) */
const TIME_SCALE = 60;
/** 由 TLE 第二行平均运动(rev/day)推出的轨道周期(分钟) */
const PERIOD_MIN = 1440 / parseFloat(TLE2.slice(52, 63));

const satrec = satellite.twoline2satrec(TLE1, TLE2);

type Vec3 = { x: number; y: number; z: number };
const cross = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
const dot = (a: Vec3, b: Vec3) => a.x * b.x + a.y * b.y + a.z * b.z;
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a.x, a.y, a.z) || 1;
  return { x: a.x / l, y: a.y / l, z: a.z / l };
};

interface SatState {
  /** 纬度幅角 u = ω + ν,由 ECI 位置/速度矢量算出 */
  u: number;
  latDeg: number;
  lonDeg: number;
  altKm: number;
  velKmS: number;
}

/** 在指定任务时刻做 SGP4 推算;失败(深空/发散)返回 null */
function propagateAt(simMs: number): SatState | null {
  const date = new Date(simMs);
  const pv = satellite.propagate(satrec, date);
  const p = pv?.position;
  const v = pv?.velocity;
  if (!p || !v || typeof p !== 'object' || typeof v !== 'object') return null;

  // 纬度幅角: 轨道面内从升交点到位置矢量的夹角
  const h = norm(cross(p, v));           // 轨道面法向
  const n = norm(cross({ x: 0, y: 0, z: 1 }, h)); // 升交点方向(倾角 97.5°,|n| 不近零)
  const q = cross(h, n);                 // 轨道面内领先升交点 90° 的方向
  const u = Math.atan2(dot(p, q), dot(p, n));

  const gmst = satellite.gstime(date);
  const geo = satellite.eciToGeodetic(p, gmst);
  return {
    u: u < 0 ? u + Math.PI * 2 : u,
    latDeg: satellite.degreesLat(geo.latitude),
    lonDeg: satellite.degreesLong(geo.longitude),
    altKm: geo.height,
    velKmS: Math.hypot(v.x, v.y, v.z),
  };
}

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

const pad2 = (n: number) => String(n).padStart(2, '0');
/** 模拟任务时钟 → ISO 风格 UTC 文本 */
function fmtSimClock(ms: number) {
  const d = new Date(ms);
  return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())} ` +
    `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}:${pad2(d.getUTCSeconds())}Z`;
}
const fmtLat = (v: number) => `LAT ${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1).padStart(4, '0')}°`;
const fmtLon = (v: number) => `LON ${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1).padStart(5, '0')}°`;

interface Blip { theta: number; color: string; phase: number; }

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

    // 星星(固定,纯氛围)
    const rand = mulberry32(20260603);
    const stars = Array.from({ length: 130 }, () => ({
      x: rand(), y: rand(), r: rand() * 1.1 + 0.2, tw: rand() * Math.PI * 2, sp: 0.5 + rand(),
    }));

    const t0 = performance.now();

    const draw = (now: number) => {
      const t = (now - t0) / 1000;
      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const horizonY = h * 2.0;          // 地球圆心在画面下方(示意)
      const earthR = h * 1.55;
      const orbitR = h * 1.78;

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

      // --- 地球(示意:大气辉光 + 球体) ---
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

      // --- 轨道环(示意) ---
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
        ctx.globalAlpha = Math.max(0, 0.5 - pulse * 0.31);
        ctx.strokeStyle = b.color;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(bx, by, 3 + pulse * 14, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = 0.95;
        ctx.fillStyle = b.color;
        ctx.beginPath(); ctx.arc(bx, by, 2.2, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;

      // --- 卫星(SGP4 实时推算 + 拖尾) ---
      const simMs = TLE_EPOCH_MS + (now - t0) * TIME_SCALE;
      const sat = propagateAt(simMs);
      const uToTheta = (u: number) => Math.PI + (u / (Math.PI * 2)) * Math.PI;
      if (sat) {
        // 拖尾: 对过去若干任务时刻分别推算(每点间隔 7 任务秒,全长约 3 任务分钟)
        for (let i = 26; i >= 1; i--) {
          const past = propagateAt(simMs - i * 7000);
          if (!past) continue;
          const tt = uToTheta(past.u);
          const tx = cx + Math.cos(tt) * orbitR;
          const ty = horizonY + Math.sin(tt) * orbitR;
          ctx.globalAlpha = 0.35 * (1 - i / 27);
          ctx.fillStyle = '#22d3ee';
          ctx.beginPath(); ctx.arc(tx, ty, Math.max(0.5, 1.6 - i * 0.04), 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = 1;

        const satTheta = uToTheta(sat.u);
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
        // 星地连线(径向,示意)
        ctx.globalAlpha = 0.10 + 0.05 * Math.sin(t * 3);
        ctx.strokeStyle = '#22d3ee';
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(cx, horizonY - earthR); ctx.stroke();
        ctx.globalAlpha = 1;
      }

      // --- HUD(全部为真实/推算值,并标注数据性质) ---
      ctx.font = '9px "IBM Plex Mono", monospace';
      ctx.fillStyle = 'rgba(127,168,166,0.75)';
      ctx.fillText('OPS-SAT · NORAD 44878 · ORBITAL-PLANE VIEW (SCHEMATIC)', 14, 46);
      if (sat) {
        ctx.fillStyle = 'rgba(159,216,232,0.9)';
        ctx.fillText(
          `${fmtLat(sat.latDeg)} ${fmtLon(sat.lonDeg)} ALT ${sat.altKm.toFixed(1)}KM VEL ${sat.velKmS.toFixed(2)}KM/S`,
          14, 60,
        );
        ctx.fillStyle = 'rgba(127,168,166,0.75)';
        ctx.fillText(
          `SIM ${fmtSimClock(simMs)} ×${TIME_SCALE} · PERIOD ${PERIOD_MIN.toFixed(1)}MIN · TLE EPOCH 2023-09-27`,
          14, 74,
        );
      } else {
        ctx.fillStyle = 'rgba(244,63,94,0.9)';
        ctx.fillText('SGP4 PROPAGATION ERROR', 14, 60);
      }
      const events = `TRACKED EVENTS: ${blips.length}`;
      ctx.fillStyle = 'rgba(127,168,166,0.75)';
      ctx.fillText(events, w - ctx.measureText(events).width - 14, 46);
      ctx.fillText(DECAY_NOTE, w - ctx.measureText(DECAY_NOTE).width - 14, 60);

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, []);

  return <canvas ref={canvasRef} className="block" />;
}
