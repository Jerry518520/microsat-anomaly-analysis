"""HTTP and actual WebSocket dependency/transport regressions."""
import asyncio

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse


@pytest.fixture
def transport(monkeypatch):
    from src.api.routes import stream
    from src.api import security
    from src.streaming.engine import StreamEngine
    from src.streaming.event_store import EventStore
    engine = StreamEngine(store=EventStore(":memory:"))
    engine.models_ready = True
    monkeypatch.setattr(stream, "engine", engine)
    monkeypatch.delenv("API_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("API_RATE_LIMIT_DISABLED", raising=False)
    monkeypatch.setenv("API_AUTH_KEYS", "transport-test")
    security._limiter.reset()
    app = FastAPI()
    app.include_router(stream.router, prefix="/api/stream",
                       dependencies=[Depends(security.require_api_key), Depends(security.rate_limit)])
    return app, engine


@pytest.mark.parametrize("key,configured,status", [("", True, 401), ("wrong", True, 401), ("", False, 503)])
def test_http_and_ws_reject_consistently(transport, monkeypatch, key, configured, status):
    app, _ = transport
    if not configured:
        monkeypatch.delenv("API_AUTH_KEYS")
    with TestClient(app) as client:
        assert client.get("/api/stream/state", headers={"X-API-Key": key}).status_code == status
        with pytest.raises(WebSocketDenialResponse) as rejected:
            with client.websocket_connect(f"/api/stream/ws?api_key={key}"):
                pass
        assert rejected.value.status_code == status


@pytest.mark.parametrize("disabled", [False, True])
def test_ws_snapshot_batch_reconnect_and_cleanup(transport, monkeypatch, disabled):
    app, engine = transport
    if disabled:
        monkeypatch.setenv("API_AUTH_DISABLED", "true")
    key = "" if disabled else "transport-test"
    headers = {"X-API-Key": key}
    with TestClient(app) as client:
        with client.websocket_connect(f"/api/stream/ws?api_key={key}") as ws:
            assert ws.receive_json()["type"] == "snapshot"
            sid = client.post("/api/stream/live/start", headers=headers, json={"source_id": "wire"}).json()["session_id"]
            from uuid import uuid4
            response = client.post("/api/stream/ingest", headers=headers, json={"session_id": sid,
                "batch_id": str(uuid4()), "frames": [{"ts": 1., "value": 2., "channel": "CADC0884"}]})
            assert response.status_code == 200
            batch = ws.receive_json()
            assert batch["type"] == "batch" and batch["stats"]["total_points"] == 1
            assert batch["session_id"] == sid
        assert not engine._subscribers
        with client.websocket_connect(f"/api/stream/ws?api_key={key}") as ws:
            assert ws.receive_json()["state"]["stats"]["total_points"] == 1
        client.post("/api/stream/live/stop", headers=headers, json={"session_id": sid})


def test_ws_and_http_rate_limit(transport, monkeypatch):
    from src.api import security
    app, _ = transport
    monkeypatch.setattr(security, "DEFAULT_LIMITS", [("/api/stream", 60., 1)])
    with TestClient(app) as client:
        headers = {"X-API-Key": "transport-test"}
        assert client.get("/api/stream/state", headers=headers).status_code == 200
        assert client.get("/api/stream/state", headers=headers).status_code == 429
        with client.websocket_connect("/api/stream/ws?api_key=transport-test") as ws:
            assert ws.receive_json()["type"] == "snapshot"
        with pytest.raises(WebSocketDenialResponse) as rejected:
            with client.websocket_connect("/api/stream/ws?api_key=transport-test"):
                pass
        assert rejected.value.status_code == 429


def test_full_subscriber_queue_requests_snapshot_resync(monkeypatch):
    from src.streaming import engine as runtime
    from src.streaming.event_store import EventStore
    engine = runtime.StreamEngine(store=EventStore(":memory:"))
    monkeypatch.setattr(runtime, "BROADCAST_INTERVAL_S", .005)
    async def scenario():
        fast, slow = engine.subscribe(), engine.subscribe()
        slow._maxsize = 1
        slow.put_nowait({"type": "stale"})
        engine._pending_points = {"CADC0884": [[1., 2.]]}
        engine._ensure_broadcaster()
        message = await asyncio.wait_for(fast.get(), timeout=1)
        assert message["type"] == "batch"
        resync = slow.get_nowait()
        assert resync["type"] == "snapshot" and resync["reason"] == "slow_consumer"
        await engine.shutdown()
    asyncio.run(scenario())
