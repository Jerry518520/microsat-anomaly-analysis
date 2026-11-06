"""A02 streaming detection with a verified frozen engineering package.

Closed segments share the A01 feature extractor and A02 judges with batch.
Growing-prefix scores are previews; only final_pred can raise a formal alert.
"""
from dataclasses import asdict
from src.detection.frozen import DetectorUnavailable, VERSION

import asyncio
import copy
import hashlib
import json
from uuid import uuid4
import time
from collections import deque
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.streaming.contracts import IngestRequest, SessionStartRequest, StreamProtocolError
from src.streaming.segmenter import ClosedSegment, OnlineSegmenter
from src.streaming.segment_features import extract_segment_features, feature_vector
from src.streaming.stage2 import METHOD_DESC, Verdict, load_or_train
from src.streaming.event_store import EventStore, StoreUnavailable
from src.utils.constants import CHANNEL_MAP

ALERT_HISTORY = 200
BROADCAST_INTERVAL_S = 0.5
MAX_REPLAY_SPEED = 2000
# 预览判分节流: 每通道至少间隔 0.5s 墙钟; 前缀不足 5 点不出预览分
MIN_PREVIEW_INTERVAL_S = 0.5
MIN_PREFIX_POINTS = 5


class StreamEngine:
    def __init__(self, store=None) -> None:
        self.judges = {}
        self.models_ready = False
        self.model_error = None
        self._models_lock: Optional[asyncio.Lock] = None

        self._store = store
        self.storage_error = None
        self._direct_session = {"session_id": str(uuid4()), "source_id": "direct",
                                "kind": "direct", "status": "active",
                                "started_at": time.time(), "ended_at": None}

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
        self.closed_events: deque = deque(maxlen=1000)
        self._pending_closed: list = []

        self._replay_task: Optional[asyncio.Task] = None
        self._broadcaster_task: Optional[asyncio.Task] = None
        self._segments_cache: Optional[pd.DataFrame] = None
        self.session_id = None
        self._sessions = {}
        self._control_lock = None
        self._idle_task = None

    # ===== 模型（只加载并校验冻结包）=====

    @property
    def store(self):
        if self._store is None:
            self._store = EventStore()
        return self._store

    async def ensure_models(self) -> None:
        if self.models_ready:
            self._ensure_broadcaster()
            return
        if self._models_lock is None:
            self._models_lock = asyncio.Lock()
        async with self._models_lock:
            if self.models_ready:
                return
            loop = asyncio.get_running_loop()
            try:
                loaded = await loop.run_in_executor(None, load_or_train)
            except DetectorUnavailable as exc:
                self.judges = {}
                self.models_ready = False
                self.model_error = str(exc)
                raise
            self.judges = loaded
            self.model_error = None
            self.models_ready = True
            self._ensure_broadcaster()

    def _ensure_broadcaster(self) -> None:
        if self._broadcaster_task is None or self._broadcaster_task.done():
            self._broadcaster_task = asyncio.create_task(self._broadcast_loop())

    # ===== 数据接入 =====

    def ingest(self, ts: float, channel: str, value: float,
               segment_id: Optional[int] = None, sampling: int = 1) -> bool:
        result = self.segmenter.feed(ts, channel, value, segment_id, sampling)
        if result.closed is not None:
            self._handle_closed(result.closed)
        if not result.accepted:
            return False
        self.sim_ts = ts
        self.stats["total_points"] += 1
        self._pending_points.setdefault(channel, []).append([ts, value])
        self._maybe_preview(channel, ts)
        return True

    def _handle_closed(self, seg: ClosedSegment) -> dict:
        record = self._prepare_closed(seg)
        session = self._direct_session if seg.session_id is None else None
        inserted = self.store.save_events([record], session=session)
        if inserted:
            self._publish_closed(record)
        return record

    def _prepare_closed(self, seg: ClosedSegment, *, strict=False) -> dict:
        record = {
            "event_id": seg.event_id, "session_id": seg.session_id or self._direct_session["session_id"],
            "source_id": seg.source_id, "channel": seg.channel,
            "segment": seg.seg_id.item() if isinstance(seg.seg_id, np.generic) else seg.seg_id,
            "source_segment_id": int(seg.source_segment_id) if seg.source_segment_id is not None else None,
            "ordinal": seg.ordinal, "part": seg.part,
            "boundary_kind": seg.boundary_kind, "close_reason": seg.close_reason,
            "completeness": seg.completeness, "sampling": int(seg.sampling),
            "n_points": len(seg.values), "first_ts": float(seg.timestamps[0]),
            "last_ts": float(seg.timestamps[-1]), "official": seg.official,
            "waveform": {"timestamps": [float(x) for x in seg.timestamps],
                         "values": [float(x) for x in seg.values]},
            "features": None, "alert": None,
        }
        if seg.completeness != "complete":
            record["status"] = "incomplete"
        elif len(seg.values) < 3:
            record["status"] = "not_applicable"
        else:
            try:
                feats, verdict = self._compute_verdict(seg)
            except (ValueError, DetectorUnavailable) as exc:
                if strict:
                    raise
                record["status"] = "failed"
                record["error"] = str(exc)
            else:
                record["status"] = "judged"
                record["features"] = feats
                record["judgement"] = {**asdict(verdict), "segment": record["segment"]}
                disp = verdict.score if verdict.score is not None else float(verdict.rule_hits)
                record["display_score"] = round(disp, 4)
                if verdict.final_pred:
                    record["alert"] = self._build_alert(seg, verdict, feats)
        return record

    def _publish_closed(self, record):
        if record["status"] == "judged":
            self.stats["judged_segments"] += 1
            latest = {"ts": record["last_ts"], "value": record["waveform"]["values"][-1],
                      "score": record["display_score"], "severity": record["judgement"]["severity"],
                      "provisional": False, "judgement": record["judgement"],
                      "event_id": record["event_id"], "session_id": record["session_id"]}
            self.latest[record["channel"]] = latest
            self._pending_scores[record["channel"]] = dict(latest)
        if record.get("alert") is not None:
            alert = {**record["alert"], "event_id": record["event_id"],
                     "session_id": record["session_id"], "source_id": record["source_id"]}
            self.alerts.appendleft(alert)
            self.stats["alerts"] += 1
            self._pending_alerts.append({"action": "new", "alert": alert})
        self.closed_events.append(record)
        self._pending_closed.append(record)

    def _source_lock(self):
        if self._control_lock is None:
            self._control_lock = asyncio.Lock()
        return self._control_lock

    def _activate_source(self, source_id, kind):
        sid = str(uuid4())
        session = {
            "session_id": sid, "source_id": source_id, "kind": kind,
            "status": "active", "started_at": time.time(), "ended_at": None,
            "last_wall": time.monotonic(), "batches": {}, "receipt": None,
        }
        self.store.save_session(self._public_session(session))
        self.session_id = sid
        self._sessions[sid] = session
        self.segmenter = OnlineSegmenter(session_id=sid, source_id=source_id,
                                         official=kind == "replay")
        self.latest.clear()
        self.alerts.clear()
        self.closed_events.clear()
        self._pending_points.clear()
        self._pending_scores.clear()
        self._pending_alerts.clear()
        self._pending_closed.clear()
        self._last_preview_wall.clear()
        self.sim_ts = None
        self.stats = {"total_points": 0, "judged_segments": 0,
                      "preview_scores": 0, "alerts": 0}
        self.mode = kind
        return self._sessions[sid]

    def _public_session(self, session):
        names = ("session_id", "source_id", "kind", "status", "started_at", "ended_at")
        return {name: session[name] for name in names}

    def _require_live(self, session_id, *, active=True):
        session = self._sessions.get(str(session_id))
        if session is None:
            session = self.store.get_session(str(session_id))
            if session is None:
                raise StreamProtocolError(404, "unknown_session", "session does not exist")
            if session["status"] == "active":
                raise StreamProtocolError(409, "session_interrupted", "source process ended; start a new session")
            self._sessions[str(session_id)] = session
        if session["kind"] != "live":
            raise StreamProtocolError(409, "source_conflict", "session is an official replay")
        if active and (session["status"] != "active" or self.session_id != str(session_id)):
            raise StreamProtocolError(409, "session_closed", "session is no longer active")
        return session

    async def start_live(self, source_id):
        source_id = SessionStartRequest(source_id=source_id).source_id
        await self.ensure_models()
        async with self._source_lock():
            if self.mode != "idle":
                raise StreamProtocolError(409, "source_conflict", "a source is already active")
            session = self._activate_source(source_id, "live")
            self._idle_task = asyncio.create_task(self._idle_monitor(session["session_id"]))
            return self._public_session(session)

    async def ingest_batch(self, session_id, batch_id, frames):
        request = IngestRequest.model_validate({"session_id": session_id,
                                                "batch_id": batch_id, "frames": frames})
        sid, bid = str(request.session_id), str(request.batch_id)
        payload = json.dumps([point.model_dump(mode="json") for point in request.frames],
                             sort_keys=True, separators=(",", ":"), allow_nan=False)
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        await self.ensure_models()
        async with self._source_lock():
            session = self._require_live(sid)
            previous = session["batches"].get(bid)
            if previous is not None:
                if previous[0] != fingerprint:
                    raise StreamProtocolError(409, "batch_conflict", "batch_id has different content")
                return json.loads(previous[1])
            # Validate the entire batch against a private copy before changing live state.
            planned = copy.deepcopy(self.segmenter)
            wall = time.monotonic()
            outcomes = [planned.feed(point.ts, point.channel, point.value,
                                     point.segment_id, point.sampling, wall=wall)
                        for point in request.frames]
            closed = [self._prepare_closed(item.closed) for item in outcomes if item.closed is not None]
            accepted = sum(item.accepted for item in outcomes)
            previews = {}
            for channel in {point.channel for point, outcome in zip(request.frames, outcomes) if outcome.accepted}:
                preview = self._prepare_preview(planned, channel, planned._last_ts[channel])
                if preview is not None:
                    previews[channel] = preview
            stats = dict(self.stats)
            stats["total_points"] += accepted
            stats["judged_segments"] += sum(item["status"] == "judged" for item in closed)
            stats["alerts"] += sum(item.get("alert") is not None for item in closed)
            stats["preview_scores"] += sum("error" not in item for item in previews.values())
            result = {"session_id": sid, "batch_id": bid, "accepted": accepted,
                      "rejected": len(request.frames) - accepted,
                      "closed_events": closed, "stats": stats}
            self.store.save_events(closed, receipt={"session_id": sid, "batch_id": bid,
                                   "fingerprint": fingerprint, "response": result})
            self.storage_error = None
            self.segmenter = planned
            channels = set()
            for point, outcome in zip(request.frames, outcomes):
                if outcome.accepted:
                    channels.add(point.channel)
                    self.sim_ts = point.ts
                    self.stats["total_points"] += 1
                    self._pending_points.setdefault(point.channel, []).append([point.ts, point.value])
            for record in closed:
                self._publish_closed(record)
            for channel, preview in previews.items():
                self._publish_preview(channel, preview)
            session["last_wall"] = wall
            frozen = json.dumps(result, allow_nan=False)
            session["batches"][bid] = (fingerprint, frozen)
            return json.loads(frozen)

    def _finish_live(self, session, status, reason):
        completeness = "complete" if status in {"ended", "completed"} else "incomplete"
        planned = copy.deepcopy(self.segmenter)
        closed = [self._prepare_closed(segment) for segment in planned.flush(
            reason=reason, completeness=completeness)]
        terminal = {**self._public_session(session), "status": status, "ended_at": time.time()}
        stats = dict(self.stats)
        stats["judged_segments"] += sum(item["status"] == "judged" for item in closed)
        stats["alerts"] += sum(item.get("alert") is not None for item in closed)
        result = {**terminal, "closed_events": closed, "stats": stats}
        terminal["receipt"] = json.dumps(result, allow_nan=False)
        self.store.save_events(closed, session=terminal)
        self.segmenter = planned
        session.update(terminal)
        for record in closed:
            self._publish_closed(record)
        self.mode = "idle"
        if self._idle_task is not None and self._idle_task is not asyncio.current_task():
            self._idle_task.cancel()
        return json.loads(session["receipt"])

    async def end_live(self, session_id):
        async with self._source_lock():
            session = self._require_live(session_id, active=False)
            if session["status"] == "ended":
                return json.loads(session["receipt"])
            self._require_live(session_id)
            return self._finish_live(session, "ended", "end")

    async def stop_live(self, session_id):
        async with self._source_lock():
            session = self._require_live(session_id, active=False)
            if session["status"] == "stopped":
                return json.loads(session["receipt"])
            self._require_live(session_id)
            return self._finish_live(session, "stopped", "stop")

    async def expire_external(self, wall=None):
        wall = time.monotonic() if wall is None else wall
        async with self._source_lock():
            if self.mode != "live":
                return []
            session = self._require_live(self.session_id)
            planned = copy.deepcopy(self.segmenter)
            closed = [self._prepare_closed(segment) for segment in planned.expire(wall)]
            self.store.save_events(closed)
            self.segmenter = planned
            for record in closed:
                self._publish_closed(record)
            if not self.segmenter._open and wall - session["last_wall"] >= 30:
                self._finish_live(session, "expired", "idle_timeout")
            return closed

    async def _idle_monitor(self, session_id):
        while self.mode == "live" and self.session_id == session_id:
            await asyncio.sleep(1)
            if self.session_id == session_id:
                try:
                    await self.expire_external()
                except StoreUnavailable as exc:
                    self.storage_error = str(exc)

    async def shutdown(self):
        tasks = [task for task in [self._idle_task, self._replay_task, self._broadcaster_task]
                 if task is not None and task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ===== 正式判定层: 段闭合 → 18 维特征 → Stage 2 =====

    def _compute_verdict(self, seg):
        judge = self.judges.get(seg.channel)
        if not self.models_ready or judge is None:
            raise DetectorUnavailable(f"closed segment detector unavailable: {seg.channel}")
        feats = extract_segment_features(seg.values, seg.timestamps, seg.sampling)
        return feats, judge.judge(feats)

    def _judge_segment(self, seg: ClosedSegment) -> None:
        record = self._prepare_closed(seg, strict=True)
        if record["status"] != "judged":
            raise ValueError("formal judgement requires a complete segment with at least 3 points")
        session = self._direct_session if seg.session_id is None else None
        if self.store.save_events([record], session=session):
            self._publish_closed(record)

    def _build_alert(self, seg: ClosedSegment, v: Verdict, feats: dict) -> dict:
        seg_label = seg.seg_id.item() if isinstance(seg.seg_id, np.generic) else seg.seg_id
        if_part = ("IF:异常" if v.if_pred else "IF:正常") if v.if_pred is not None else "IF:不可用（当前门控不需要）"
        alert = {
            **asdict(v),
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
                f"【实时·{v.detector_version}】{CHANNEL_MAP[seg.channel]} 段{seg_label} 判定异常 · "
                f"门控 {v.gate_op} · {if_part} · 规则违例 {v.rule_hits} 条(k={self.judges[seg.channel].k}) · "
                f"len={int(feats['len'])} mean={feats['mean']:.4g} std={feats['std']:.4g} · "
                f"RAG 诊断待生成"
            ),
        }
        return alert

    # ===== 增长段预览层(非正式判定) =====

    def _prepare_preview(self, segmenter, channel, ts):
        if not self.models_ready or channel not in self.judges:
            return None
        if time.monotonic() - self._last_preview_wall.get(channel, 0.) < MIN_PREVIEW_INTERVAL_S:
            return None
        prefix = segmenter.open_prefix(channel)
        if prefix is None or len(prefix[0]) < MIN_PREFIX_POINTS:
            return None
        values, tss, sampling = prefix
        try:
            feats = extract_segment_features(values, tss, sampling)
            judge = self.judges[channel]
            score = judge.continuous_score(feature_vector(feats))
            hits = judge.rule_eval(feats)
            if score is None:
                disp = float(hits)
                severity = "critical" if hits >= 4 else ("warning" if hits >= 3 else (
                    "caution" if hits >= 2 else "nominal"))
            else:
                disp = score
                if score > judge.score_tau95 + 3 * judge.score_std:
                    severity = "critical"
                elif score > judge.score_tau95 + 1.5 * judge.score_std:
                    severity = "warning"
                elif score > judge.score_tau95:
                    severity = "caution"
                else:
                    severity = "nominal"
            return {"ts": ts, "value": float(values[-1]), "score": round(disp, 4),
                    "severity": severity, "provisional": True}
        except (ValueError, DetectorUnavailable) as exc:
            return {"ts": ts, "value": float(values[-1]), "score": None,
                    "severity": "offline", "provisional": True, "error": str(exc)}

    def _publish_preview(self, channel, preview):
        self._last_preview_wall[channel] = time.monotonic()
        if "error" not in preview:
            self.stats["preview_scores"] += 1
        self.latest[channel] = preview
        self._pending_scores[channel] = dict(preview)

    def _maybe_preview(self, channel, ts):
        preview = self._prepare_preview(self.segmenter, channel, ts)
        if preview is not None:
            self._publish_preview(channel, preview)

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
            df = df.sort_values("ts", kind="stable").reset_index(drop=True)
            self._segments_cache = df
        return self._segments_cache

    async def start_replay(self, speed: int = 60) -> dict:
        if isinstance(speed, bool) or not isinstance(speed, int) or not 1 <= speed <= MAX_REPLAY_SPEED:
            raise ValueError("replay speed must be an integer from 1 to 2000")
        await self.ensure_models()
        async with self._source_lock():
            if self.mode != "idle":
                raise StreamProtocolError(409, "source_conflict", "a source is already active")
            session = self._activate_source("official-replay", "replay")
            self.speed = speed
            self._replay_task = asyncio.create_task(self._replay_loop(speed, session["session_id"]))
            return {"mode": self.mode, "speed": speed, **self._public_session(session)}

    async def stop_replay(self) -> dict:
        async with self._source_lock():
            if self.mode == "live":
                raise StreamProtocolError(409, "source_conflict", "external live source is active")
            if self._replay_task is not None and not self._replay_task.done():
                self._replay_task.cancel()
                await asyncio.gather(self._replay_task, return_exceptions=True)
            if self.mode == "replay":
                session = self._sessions[self.session_id]
                self._finish_live(session, "stopped", "stop")
            return {"mode": self.mode, "session_id": self.session_id}

    async def _replay_loop(self, speed: int, session_id: str) -> None:
        try:
            df = await asyncio.get_running_loop().run_in_executor(None, self._load_segments)
            if self.session_id != session_id or self.mode != "replay":
                return
            wall0 = time.monotonic()
            sim0 = float(df["ts"].iloc[0])
            for i, row in enumerate(df.itertuples(index=False)):
                due = wall0 + (float(row.ts) - sim0) / speed
                while due > time.monotonic():
                    await asyncio.sleep(min(due - time.monotonic(), 1.0))
                self.ingest(float(row.ts), row.channel, float(row.value),
                            segment_id=int(row.segment), sampling=int(row.sampling))
                if i % 5000 == 0:
                    await asyncio.sleep(0)
            session = self._sessions[session_id]
            self._finish_live(session, "completed", "replay_end")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.session_id == session_id:
                session = self._sessions[session_id]
                session["error"] = str(exc)
                try:
                    self._finish_live(session, "failed", "replay_error")
                except StoreUnavailable as storage_exc:
                    self.storage_error = str(storage_exc)
                    session["status"] = "failed"
                    session["ended_at"] = time.time()
                self.mode = "idle"

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
                self._pending_closed.clear()
                continue
            if not (self._pending_points or self._pending_scores or self._pending_alerts or self._pending_closed):
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
                "session_id": self.session_id,
                "closed_events": self._pending_closed,
            }
            self._pending_points = {}
            self._pending_scores = {}
            self._pending_alerts = []
            self._pending_closed = []
            for q in list(self._subscribers):
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    while not q.empty():
                        q.get_nowait()
                    q.put_nowait({"type": "snapshot", "reason": "slow_consumer", "state": self.snapshot()})

    # ===== 快照 =====

    def snapshot(self) -> dict[str, Any]:
        channels = {}
        for ch in CHANNEL_MAP:
            last = self.latest.get(ch)
            channels[ch] = {
                "model": "ready" if self.models_ready and ch in self.judges else "unavailable",
                "gate_op": self.judges[ch].gate_op if self.models_ready and ch in self.judges else None,
                "rule_k": self.judges[ch].k if self.models_ready and ch in self.judges else None,
                "judgement": last.get("judgement") if last else None,
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
            "session_id": self.session_id,
            "session": self._public_session(self._sessions[self.session_id]) if self.session_id else None,
            "models_ready": self.models_ready,
            "model_error": self.model_error,
            "storage_error": self.storage_error,
            "detector_version": VERSION,
            "method": METHOD_DESC,
            "channels": channels,
            "alerts": list(self.alerts),
            "stats": dict(self.stats),
        }


# 模块级单例
engine = StreamEngine()
