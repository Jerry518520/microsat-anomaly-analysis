"""Bounded per-channel segmentation with explicit boundary provenance."""
from dataclasses import dataclass, field
import time
from uuid import uuid4

import numpy as np

from src.streaming.contracts import StreamProtocolError, TelemetryFrame


@dataclass
class ClosedSegment:
    channel: str
    seg_id: str
    official: bool
    values: np.ndarray
    timestamps: np.ndarray
    sampling: int
    event_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    source_id: str = "unspecified"
    source_segment_id: int | None = None
    ordinal: int = 0
    part: int = 1
    boundary_kind: str = "unspecified"
    close_reason: str = "segment_change"
    completeness: str = "complete"


@dataclass
class FeedResult:
    accepted: bool
    closed: ClosedSegment | None = None
    reason: str | None = None


@dataclass
class _OpenSegment:
    seg_id: str
    source_segment_id: int | None
    sampling: int
    event_id: str
    ordinal: int
    part: int
    last_wall: float
    values: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)


class OnlineSegmenter:
    def __init__(self, session_id=None, source_id="external", official=False,
                 max_points=10000):
        if max_points < 3:
            raise ValueError("max_points must be at least three")
        self.session_id = session_id
        self.source_id = source_id
        self.official = official
        self.max_points = max_points
        self._open = {}
        self._last_ts = {}
        self._source_mode = {}
        self._closed_ids = {}
        self._oversize_ids = {}
        self._parts = {}
        self._sequence = 0

    def _new(self, channel, segment_id, sampling, wall):
        self._sequence += 1
        part = self._parts.get(channel, 0) + 1
        self._parts[channel] = part
        sid = str(segment_id) if segment_id is not None else f"LIVE-{self._sequence:04d}"
        current = _OpenSegment(sid, segment_id, sampling, str(uuid4()),
                               self._sequence, part, wall)
        self._open[channel] = current
        return current

    def _close(self, channel, reason, completeness="complete"):
        current = self._open.pop(channel)
        if current.source_segment_id is not None:
            self._closed_ids.setdefault(channel, set()).add(current.source_segment_id)
        boundary = "official" if self.official else "source_segment"
        if current.source_segment_id is None:
            boundary = "approximate_window" if reason == "max_points" else "approximate_gap"
        return ClosedSegment(
            channel, current.seg_id, self.official,
            np.asarray(current.values, dtype=float),
            np.asarray(current.timestamps, dtype=float), current.sampling,
            event_id=current.event_id, session_id=self.session_id,
            source_id=self.source_id, source_segment_id=current.source_segment_id,
            ordinal=current.ordinal, part=current.part, boundary_kind=boundary,
            close_reason=reason, completeness=completeness,
        )

    def feed(self, ts, channel, value, segment_id=None, sampling=1, *, wall=None):
        point = TelemetryFrame(ts=ts, channel=channel, value=value,
                               segment_id=segment_id, sampling=sampling)
        ts, value, sampling = point.ts, point.value, point.sampling
        supplied = segment_id is not None
        if self.official and not supplied:
            raise ValueError("official replay requires segment_id")
        if channel in self._last_ts and ts < self._last_ts[channel]:
            raise ValueError("timestamps must be nondecreasing within a channel")
        if channel in self._source_mode and self._source_mode[channel] != supplied:
            raise ValueError("segment_id presence must stay consistent within a session/channel")
        if segment_id in self._oversize_ids.get(channel, set()):
            return FeedResult(False, reason="source_oversize")
        if segment_id in self._closed_ids.get(channel, set()):
            raise StreamProtocolError(409, "closed_source_segment", "closed source segment cannot reopen")
        current = self._open.get(channel)
        reason = None
        if current is not None:
            if supplied and current.source_segment_id != segment_id:
                reason = "segment_change"
            elif not supplied and ts - current.timestamps[-1] > max(3 * current.sampling, 5):
                reason = "gap"
            elif current.sampling != sampling:
                raise ValueError("sampling cannot change inside an open segment")
            elif len(current.values) >= self.max_points:
                if supplied:
                    self._oversize_ids.setdefault(channel, set()).add(segment_id)
                    closed = self._close(channel, "source_oversize", "incomplete")
                    return FeedResult(False, closed, "source_oversize")
                reason = "max_points"
        closed = self._close(channel, reason) if reason else None
        wall = time.monotonic() if wall is None else wall
        if current is None or reason:
            current = self._new(channel, segment_id, sampling, wall)
        current.values.append(value)
        current.timestamps.append(ts)
        current.last_wall = wall
        self._last_ts[channel] = ts
        self._source_mode[channel] = supplied
        return FeedResult(True, closed)

    def open_prefix(self, channel):
        current = self._open.get(channel)
        if current is None:
            return None
        return (np.asarray(current.values, dtype=float),
                np.asarray(current.timestamps, dtype=float), current.sampling)

    def open_len(self, channel):
        current = self._open.get(channel)
        return len(current.values) if current else 0

    def expire(self, wall):
        channels = [channel for channel, current in self._open.items()
                    if wall - current.last_wall >= max(3 * current.sampling, 30)]
        return [self._close(channel, "idle_timeout", "incomplete") for channel in channels]

    def flush(self, channel=None, *, reason="end", completeness="complete"):
        channels = [channel] if channel in self._open else ([] if channel else list(self._open))
        return [self._close(ch, reason, completeness) for ch in channels]

    def drop(self, channel=None, *, reason="stop"):
        return self.flush(channel, reason=reason, completeness="incomplete")