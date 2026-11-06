"""Stream API — 实时遥测流接入/回放/推送"""
import asyncio
import os
import sys
from anyio import CancelScope

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from src.detection.frozen import DetectorUnavailable
from pydantic import BaseModel, ConfigDict, Field
from src.streaming.contracts import (TelemetryFrame, IngestRequest, SessionStartRequest,
                                     SessionRequest, StreamProtocolError)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.streaming.engine import engine
from src.streaming.event_store import StoreUnavailable, EventConflict

router = APIRouter()


def _unavailable(exc):
    return HTTPException(status_code=503, detail={"error": "detector_unavailable", "message": str(exc)})


async def _ensure_models():
    try:
        await engine.ensure_models()
    except DetectorUnavailable as exc:
        raise _unavailable(exc) from exc
    except StoreUnavailable as exc:
        raise HTTPException(status_code=503, detail={"error": "storage_unavailable", "message": str(exc)}) from exc
    except EventConflict as exc:
        raise HTTPException(status_code=409, detail={"error": "event_conflict", "message": str(exc)}) from exc


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    speed: int = Field(default=60, ge=1, le=2000)


IngestFrame = TelemetryFrame


async def _stream_call(operation):
    try:
        return await operation
    except DetectorUnavailable as exc:
        raise _unavailable(exc) from exc
    except StoreUnavailable as exc:
        engine.storage_error = str(exc)
        raise HTTPException(status_code=503, detail={"error": "storage_unavailable", "message": str(exc)}) from exc
    except EventConflict as exc:
        raise HTTPException(status_code=409, detail={"error": "event_conflict", "message": str(exc)}) from exc
    except StreamProtocolError as exc:
        raise HTTPException(status_code=exc.status_code,
                            detail={"error": exc.code, "message": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422,
                            detail={"error": "invalid_stream_input", "message": str(exc)}) from exc


@router.post("/live/start")
async def start_live(req: SessionStartRequest):
    return await _stream_call(engine.start_live(req.source_id))


@router.post("/live/end")
async def end_live(req: SessionRequest):
    return await _stream_call(engine.end_live(str(req.session_id)))


@router.post("/live/stop")
async def stop_live(req: SessionRequest):
    return await _stream_call(engine.stop_live(str(req.session_id)))


@router.get("/state")
async def get_state():
    """实时流状态快照(模式/仿真时钟/各通道最新判定/活跃告警/统计)。
    首次调用只加载和校验冻结包；缺失或不兼容返回503。"""
    await _ensure_models()
    return engine.snapshot()


@router.post("/replay/start")
async def start_replay(req: ReplayRequest):
    """开始回放 18 天遥测(SIMULATED LIVE)。speed=倍速,1~2000"""
    return await _stream_call(engine.start_replay(req.speed))


@router.post("/replay/stop")
async def stop_replay():
    """停止回放"""
    return await _stream_call(engine.stop_replay())


@router.post("/ingest")
async def ingest_frames(req: IngestRequest):
    """Accept one validated, idempotent batch for an active external session."""
    return await _stream_call(engine.ingest_batch(
        str(req.session_id), str(req.batch_id), [frame.model_dump() for frame in req.frames]))


@router.websocket("/ws")
async def stream_ws(ws: WebSocket):
    """实时推送: 连接即发 snapshot,随后每 0.5s 批量推送 points/scores/alerts"""
    await ws.accept()
    try:
        await engine.ensure_models()
    except DetectorUnavailable as exc:
        await ws.send_json({"type": "error", "error": "detector_unavailable", "message": str(exc), "state": engine.snapshot()})
        await ws.close(code=1013)
        return
    q = engine.subscribe()
    disconnected = asyncio.create_task(_wait_disconnect(ws))
    next_message = asyncio.create_task(q.get())
    try:
        await ws.send_json({"type": "snapshot", "state": engine.snapshot()})
        while True:
            done, _ = await asyncio.wait([next_message, disconnected], timeout=15.,
                                         return_when=asyncio.FIRST_COMPLETED)
            if disconnected in done:
                break
            msg = {"type": "ping"}
            if next_message in done:
                msg = next_message.result()
                next_message = asyncio.create_task(q.get())
            await asyncio.wait_for(ws.send_json(msg), timeout=5.)
    except (WebSocketDisconnect, RuntimeError, OSError, asyncio.TimeoutError):
        pass
    finally:
        engine.unsubscribe(q)
        disconnected.cancel()
        next_message.cancel()
        with CancelScope(shield=True):
            await asyncio.gather(disconnected, next_message, return_exceptions=True)


async def _wait_disconnect(ws):
    while True:
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            return


# ---------------------------------------------------------------- 告警查询（持久化）

@router.get("/alerts")
def query_alerts(
    since: float | None = None,
    until: float | None = None,
    channel: str | None = None,
    severity: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """查询历史告警（SQLite）。

    与 `/state` 内的 `alerts` 不同：那个只有内存中最近 200 条且**重启即丢**，
    这里读的是持久化库，可按时间范围/通道/严重度检索。
    """
    limit, offset = max(1, min(int(limit), 1000)), max(0, int(offset))
    try:
        items, total = engine.store.query_alerts(since=since, until=until, channel=channel,
                                               severity=severity, limit=limit, offset=offset)
    except StoreUnavailable as exc:
        raise HTTPException(status_code=503, detail={"error": "storage_unavailable"}) from exc
    return {"total": total, "limit": limit, "offset": offset, "items": items}


@router.get("/alerts/stats")
def alerts_stats():
    """告警汇总：总数、按严重度分布、按通道分布。"""
    try:
        return engine.store.stats()
    except StoreUnavailable as exc:
        raise HTTPException(status_code=503, detail={"error": "storage_unavailable"}) from exc


@router.get("/events/{event_id}")
def get_event(event_id: str):
    try:
        event = engine.store.get_event(event_id)
    except StoreUnavailable as exc:
        raise HTTPException(status_code=503, detail={"error": "storage_unavailable"}) from exc
    if event is None:
        raise HTTPException(status_code=404, detail={"error": "unknown_event"})
    return event
