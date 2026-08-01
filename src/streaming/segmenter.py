"""
在线切段器 — 把遥测点流切成与官方数据集同构的段

两种切段依据:
- replay 模式: segments.csv 每行自带官方 segment id, 直接按 id 分组,
  段边界与官方数据集完全一致(逐点保真);
- ingest 模式(外部源无段 id): 退化为采样间隔规则——相邻点时间差超过
  max(3×sampling, 5s) 即切段。这是官方 dataset_generator 切段规则的
  流式近似, 仅用于未来真实数据源, 并在文档中明确标注为近似。
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ClosedSegment:
    channel: str
    seg_id: str            # 官方段 id(replay)或 LIVE-NNNN(ingest)
    official: bool         # True = 官方段 id; False = 流式近似切段
    values: np.ndarray
    timestamps: np.ndarray
    sampling: int


@dataclass
class _OpenSegment:
    seg_id: str
    official: bool
    sampling: int
    values: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)

    def close(self, channel: str) -> ClosedSegment:
        return ClosedSegment(
            channel=channel, seg_id=self.seg_id, official=self.official,
            values=np.array(self.values, dtype=float),
            timestamps=np.array(self.timestamps, dtype=float),
            sampling=self.sampling,
        )


class OnlineSegmenter:
    """每通道维护一个开放段; 段边界到达时产出 ClosedSegment"""

    def __init__(self) -> None:
        self._open: dict[str, _OpenSegment] = {}
        self._live_seq = 0

    def feed(self, ts: float, channel: str, value: float,
             segment_id: Optional[int] = None, sampling: int = 1
             ) -> Optional[ClosedSegment]:
        """
        送入一个遥测点; 若触发切段, 先返回闭合的段。
        segment_id 非 None → 官方 id 切段(replay); 否则间隔规则(ingest)。
        """
        cur = self._open.get(channel)

        if segment_id is not None:
            sid = str(segment_id)
            if cur is None:
                self._open[channel] = _OpenSegment(sid, True, sampling)
            elif cur.seg_id != sid:
                closed = cur.close(channel)
                self._open[channel] = _OpenSegment(sid, True, sampling)
                self._append(channel, ts, value)
                return closed
        else:
            gap_limit = max(3 * sampling, 5)
            if cur is None:
                self._live_seq += 1
                self._open[channel] = _OpenSegment(f"LIVE-{self._live_seq:04d}", False, sampling)
            elif cur.timestamps and ts - cur.timestamps[-1] > gap_limit:
                closed = cur.close(channel)
                self._live_seq += 1
                self._open[channel] = _OpenSegment(f"LIVE-{self._live_seq:04d}", False, sampling)
                self._append(channel, ts, value)
                return closed

        self._append(channel, ts, value)
        return None

    def _append(self, channel: str, ts: float, value: float) -> None:
        seg = self._open[channel]
        seg.values.append(value)
        seg.timestamps.append(ts)

    def open_prefix(self, channel: str) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
        """当前开放段的前缀(values, timestamps, sampling), 供增长段预览"""
        seg = self._open.get(channel)
        if seg is None or not seg.values:
            return None
        return (np.array(seg.values, dtype=float),
                np.array(seg.timestamps, dtype=float), seg.sampling)

    def open_len(self, channel: str) -> int:
        seg = self._open.get(channel)
        return len(seg.values) if seg else 0

    def flush(self, channel: Optional[str] = None) -> list[ClosedSegment]:
        """强制闭合开放段(回放正常结束时调用; 中途停止应改用 drop)"""
        channels = [channel] if channel else list(self._open)
        closed = []
        for ch in channels:
            seg = self._open.pop(ch, None)
            if seg and seg.values:
                closed.append(seg.close(ch))
        return closed

    def drop(self, channel: Optional[str] = None) -> None:
        """丢弃开放段(回放中途停止: 段不完整, 判定会违反分布一致性)"""
        if channel:
            self._open.pop(channel, None)
        else:
            self._open.clear()
