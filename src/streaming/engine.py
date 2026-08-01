"""
StreamEngine — 实时遥测异常检测引擎(Stage 2 段判定版)

数据流: ingest(遥测帧) → 在线切段(与官方段同构) → 段闭合 → 18 维段级特征
→ Stage 2 判定(ipynb 严格一致) → 异常段生成告警 → 批量广播给 WebSocket 订阅者。

双层语义(诚实性设计):
- 正式判定层: 只在段闭合时发生, 判定单元/特征/规则/IF 参数与 ipynb
  Stage 2 完全一致, 离线 SegF1 指标原样适用。告警只由这一层产生。
- 增长段预览层: 段未闭合时对"当前段前缀"算 18 维特征并出展示分,
  每点刷新供大屏曲线使用。前缀分布随段长向训练分布收敛, 但短前缀有
  系统性偏差——预览分一律带 provisional=True, 不产生告警、不作判定。
- 预览着色阈值(训练分 τ95 + σ 分级)仅决定曲线颜色, 不是判定边界。

数据源:
- replay: 按真实时间戳回放 data/raw/segments.csv(18 天遥测), 倍速可调,
  界面须标注 SIMULATED LIVE —— OPS-SAT 已于 2024-05-22 再入, 不存在真实时流。
  段 id 来自数据本身, 切段与官方完全一致。
- ingest: 外部源 POST 遥测帧; 无段 id 时用采样间隔规则近似切段(见 segmenter)。
"""
import asyncio
import time
from collections import deque
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.streaming.segmenter import ClosedSegment, OnlineSegmenter
from src.streaming.segment_features import extract_segment_features, feature_vector
from src.streaming.stage2 import METHOD_DESC, Verdict, load_or_train
from src.utils.constants import CHANNEL_MAP

ALERT_HISTORY = 200
BROADCAST_INTERVAL_S = 0.5
MAX_REPLAY_SPEED = 2000
# 预览判分节流: 每通道至少间隔 0.5s 墙钟; 前缀不足 5 点不出预览分
MIN_PREVIEW_INTERVAL_S = 0.5
MIN_PREFIX_POINTS = 5


class StreamEngine:
    def __init__(self) -> None:
        self.judges = {}
        self.models_ready = False
        self._models_lock: Optional[asyncio.Lock] = None

        self.segmenter = OnlineSegmenter()
        self.latest: dict[str, dict] = {}           # ch -> {ts, value, score, severity, provisional}
        self.alerts: deque = deque(maxlen=ALERT_HISTORY)
        self._alert_seq = 0

        self.mode = "idle"                          # idle | replay | live
        self.speed = 60
        self.sim_ts: Optional[float] = None         # 当前仿真遥测时间(unix 秒)
        self.stats = {"total_points": 0, "judged_segments": 0,
                      "preview_scores": 0, "alerts": 0}

        self._subscribers: set[asyncio.Queue] = set()
        self._pending_points: dict[str, list] = {}
        self._pending_scores: dict[str, dict] = {}
        self._pending_alerts: list = []
        self._last_preview_wall: dict[str, float] = {}

        self._replay_task: Optional[asyncio.Task] = None
        self._broadcaster_task: Optional[asyncio.Task] = None
        self._segments_cache: Optional[pd.DataFrame] = None

    # ===== 模型(懒加载,训练放线程池避免阻塞事件循环) =====

    async def ensure_models(self) -> None:
        if self.models_ready:
            return
        if self._models_lock is None:
            self._models_lock = asyncio.Lock()
        async with self._models_lock:
            if self.models_ready:
                return
            loop = asyncio.get_running_loop()
            self.judges = await loop.run_in_executor(None, load_or_train)
            self.models_ready = True
            self._ensure_broadcaster()

    def _ensure_broadcaster(self) -> None:
        if self._broadcaster_task is None or self._broadcaster_task.done():
            self._broadcaster_task = asyncio.create_task(self._broadcast_loop())

    # ===== 数据接入 =====

    def ingest(self, ts: float, channel: str, value: float,
               segment_id: Optional[int] = None, sampling: int = 1) -> bool:
        """接入一帧遥测。返回是否接受(未知通道拒收)"""
        if channel not in CHANNEL_MAP:
            return False
        self.sim_ts = ts
        self.stats["total_points"] += 1
        self._pending_points.setdefault(channel, []).append([round(ts, 3), value])

        closed = self.segmenter.feed(ts, channel, value, segment_id, sampling)
        if closed is not None:
            self._judge_segment(closed)
        self._maybe_preview(channel, ts)
        return True

    # ===== 正式判定层: 段闭合 → 18 维特征 → Stage 2 =====

    def _judge_segment(self, seg: ClosedSegment) -> None:
        ch = seg.channel
        judge = self.judges.get(ch)
        if not self.models_ready or judge is None or len(seg.values) < 3:
            return
        feats = extract_segment_features(seg.values, seg.timestamps, seg.sampling)
        verdict = judge.judge(feats)
        self.stats["judged_segments"] += 1

        # 展示分: 强通道用 IF 连续分, 弱通道用规则违例数
        disp = verdict.score if verdict.score is not None else float(verdict.rule_hits)
        self.latest[ch] = {
            "ts": float(seg.timestamps[-1]), "value": float(seg.values[-1]),
            "score": round(disp, 4), "severity": verdict.severity, "provisional": False,
        }
        self._pending_scores[ch] = {
            "ts": float(seg.timestamps[-1]), "score": round(disp, 4),
            "severity": verdict.severity, "provisional": False,
        }
        if verdict.anomaly:
            self._raise_alert(seg, verdict, feats)

    def _raise_alert(self, seg: ClosedSegment, v: Verdict, feats: dict) -> None:
        self._alert_seq += 1
        seg_label = seg.seg_id if seg.official else seg.seg_id  # 官方段 id 或 LIVE-NNNN
        if_part = ("IF:异常" if v.if_pred else "IF:正常") if v.if_pred is not None else "IF:—(弱通道纯规则)"
        alert = {
            "segment": seg_label,
            "channel": seg.channel,
            "channel_label": CHANNEL_MAP[seg.channel],
            "first_ts": float(seg.timestamps[0]),
            "last_ts": float(seg.timestamps[-1]),
            "raw_score": round(v.score, 4) if v.score is not None else float(v.rule_hits),
            "anomaly_score": v.anomaly_score,
            "severity": v.severity,
            "hits": v.rule_hits,
            "anomaly_type": "unclassified (live)",
            "summary": (
                f"【实时·Stage 2】{CHANNEL_MAP[seg.channel]} 段{seg_label} 判定异常 · "
                f"{if_part} · 规则违例 {v.rule_hits} 条(≥2 判异常) · "
                f"len={int(feats['len'])} mean={feats['mean']:.4g} std={feats['std']:.4g} · "
                f"RAG 诊断待生成"
            ),
        }
        self.alerts.appendleft(alert)
        self.stats["alerts"] += 1
        self._pending_alerts.append({"action": "new", "alert": alert})

    # ===== 增长段预览层(非正式判定) =====

    def _maybe_preview(self, channel: str, ts: float) -> None:
        if not self.models_ready or channel not in self.judges:
            return
        now = time.monotonic()
        if now - self._last_preview_wall.get(channel, 0.0) < MIN_PREVIEW_INTERVAL_S:
            return
        prefix = self.segmenter.open_prefix(channel)
        if prefix is None or len(prefix[0]) < MIN_PREFIX_POINTS:
            return
        self._last_preview_wall[channel] = now

        values, tss, sampling = prefix
        feats = extract_segment_features(values, tss, sampling)
        judge = self.judges[channel]

        # 展示分: 强通道 = IF 连续分(τ95+σ 着色), 弱通道 = 规则违例计数着色
        vec = feature_vector(feats)
        score = judge.continuous_score(vec)
        rule_hits = judge.rule_eval(feats)
        if score is not None:
            disp = score
            if score > judge.score_tau95 + 3 * judge.score_std:
                sev = "critical"
            elif score > judge.score_tau95 + 1.5 * judge.score_std:
                sev = "warning"
            elif score > judge.score_tau95:
                sev = "caution"
            else:
                sev = "nominal"
        else:
            disp = float(rule_hits)
            sev = "critical" if rule_hits >= 4 else ("warning" if rule_hits >= 3 else (
                "caution" if rule_hits >= 2 else "nominal"))

        self.stats["preview_scores"] += 1
        # 不覆盖正式判定后的最新状态?——预览反映的是"当前进行中的段",
        # 时序上更新, 允许覆盖; provisional 标志保证前端可区分
        self.latest[channel] = {
            "ts": ts, "value": float(values[-1]),
            "score": round(disp, 4), "severity": sev, "provisional": True,
        }
        self._pending_scores[channel] = {
            "ts": ts, "score": round(disp, 4), "severity": sev, "provisional": True,
        }

    # ===== 回放驱动 =====

    def _load_segments(self) -> pd.DataFrame:
        if self._segments_cache is None:
            import os
            path = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                "data", "raw", "segments.csv")
            df = pd.read_csv(path)
            ts = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
            # 注意: pandas 非纳秒分辨率下 astype(int64) 单位不定,用 total_seconds 兜底
            df["ts"] = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
            df = df.sort_values("ts").reset_index(drop=True)
            self._segments_cache = df
        return self._segments_cache

    async def start_replay(self, speed: int = 60) -> dict:
        await self.ensure_models()
        await self.stop_replay()
        self.segmenter.drop()  # 清除旧数据源残留的开放段
        self.speed = max(1, min(int(speed), MAX_REPLAY_SPEED))
        self.mode = "replay"
        self._replay_task = asyncio.create_task(self._replay_loop(self.speed))
        return {"mode": self.mode, "speed": self.speed}

    async def stop_replay(self) -> dict:
        if self._replay_task and not self._replay_task.done():
            self._replay_task.cancel()
            try:
                await self._replay_task
            except asyncio.CancelledError:
                pass
        if self.mode == "replay":
            self.mode = "idle"
        # 中途停止: 开放段不完整, 判定会违反分布一致性, 直接丢弃
        self.segmenter.drop()
        return {"mode": self.mode}

    async def _replay_loop(self, speed: int) -> None:
        """按真实时间戳回放;墙钟节奏 = 遥测时间差 / speed"""
        try:
            df = await asyncio.get_running_loop().run_in_executor(None, self._load_segments)
            wall0 = time.monotonic()
            sim0 = float(df["ts"].iloc[0])
            for i, row in enumerate(df.itertuples(index=False)):
                due = wall0 + (float(row.ts) - sim0) / speed
                delay = due - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(min(delay, 1.0))
                    # 长睡眠可能被 cancel,醒来后继续按 due 推进即可
                self.ingest(float(row.ts), row.channel, float(row.value),
                            segment_id=int(row.segment), sampling=int(row.sampling))
                if i % 5000 == 0:
                    await asyncio.sleep(0)  # 让出事件循环
            # 回放正常结束: 末段不会再有"下一个段 id"触发切段, flush 补判
            for seg in self.segmenter.flush():
                self._judge_segment(seg)
            self.mode = "idle"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.mode = "idle"
            print(f"[StreamEngine] 回放异常终止: {e}")

    # ===== 广播 =====

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=300)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def _broadcast_loop(self) -> None:
        while True:
            await asyncio.sleep(BROADCAST_INTERVAL_S)
            if not self._subscribers:
                self._pending_points.clear()
                self._pending_scores.clear()
                self._pending_alerts.clear()
                continue
            if not (self._pending_points or self._pending_scores or self._pending_alerts):
                continue
            msg = {
                "type": "batch",
                "sim_ts": self.sim_ts,
                "mode": self.mode,
                "speed": self.speed,
                "points": self._pending_points,
                "scores": self._pending_scores,
                "alerts": self._pending_alerts,
                "stats": dict(self.stats),
            }
            self._pending_points = {}
            self._pending_scores = {}
            self._pending_alerts = []
            for q in list(self._subscribers):
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass  # 慢客户端丢帧,保证实时性

    # ===== 快照 =====

    def snapshot(self) -> dict[str, Any]:
        channels = {}
        for ch in CHANNEL_MAP:
            last = self.latest.get(ch)
            channels[ch] = {
                "model": "ready" if ch in self.judges else "unavailable",
                "open_segment_len": self.segmenter.open_len(ch),
                "last_ts": last["ts"] if last else None,
                "last_value": last["value"] if last else None,
                "score": last["score"] if last else None,
                "severity": last["severity"] if last else "nominal",
                "provisional": last["provisional"] if last else True,
            }
        return {
            "mode": self.mode,
            "speed": self.speed,
            "sim_ts": self.sim_ts,
            "models_ready": self.models_ready,
            "method": METHOD_DESC,
            "channels": channels,
            "alerts": list(self.alerts),
            "stats": dict(self.stats),
        }


# 模块级单例
engine = StreamEngine()
