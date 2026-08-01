"""Stream API — 实时遥测流接入/回放/推送"""
import asyncio
import os
import sys

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.streaming.engine import engine

router = APIRouter()


class ReplayRequest(BaseModel):
    speed: int = 60


class IngestFrame(BaseModel):
    ts: float                       # unix 秒
    channel: str
    value: float
    segment_id: int | None = None   # 官方段 id(可选); 缺省时按采样间隔规则近似切段
    sampling: int = 1               # 采样间隔(秒)


class IngestRequest(BaseModel):
    frames: list[IngestFrame]


@router.get("/state")
async def get_state():
    """实时流状态快照(模式/仿真时钟/各通道最新判定/活跃告警/统计)。
    首次调用会触发 Stage 2 判定器训练(数秒~首次生成特征约 1 分钟)。"""
    await engine.ensure_models()
    return engine.snapshot()


@router.post("/replay/start")
async def start_replay(req: ReplayRequest):
    """开始回放 18 天遥测(SIMULATED LIVE)。speed=倍速,1~2000"""
    return await engine.start_replay(req.speed)


@router.post("/replay/stop")
async def stop_replay():
    """停止回放"""
    return await engine.stop_replay()


@router.post("/ingest")
async def ingest_frames(req: IngestRequest):
    """
    通用实时接入: 外部数据源推送遥测帧。
    回放进行中返回 409(同一时刻只允许一个数据源)。
    """
    if engine.mode == "replay":
        return {"accepted": 0, "rejected": len(req.frames),
                "error": "replay in progress, stop it first"}
    await engine.ensure_models()
    engine.mode = "live"
    accepted = 0
    for f in req.frames:
        if engine.ingest(f.ts, f.channel, f.value, f.segment_id, f.sampling):
            accepted += 1
    return {"accepted": accepted, "rejected": len(req.frames) - accepted}


@router.websocket("/ws")
async def stream_ws(ws: WebSocket):
    """实时推送: 连接即发 snapshot,随后每 0.5s 批量推送 points/scores/alerts"""
    await ws.accept()
    await engine.ensure_models()
    await ws.send_json({"type": "snapshot", "state": engine.snapshot()})
    q = engine.subscribe()
    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=15.0)
            except asyncio.TimeoutError:
                msg = {"type": "ping"}   # 保活;连接已断时 send 会抛异常退出循环
            await ws.send_json(msg)
    except (WebSocketDisconnect, RuntimeError, OSError):
        pass
    finally:
        engine.unsubscribe(q)
