"""Acceptance checks for external sessions and bounded segmentation."""
import asyncio
from uuid import uuid4

import pytest
from pydantic import ValidationError

CHANNEL = "CADC0884"


def frame(ts=0.0, value=1.0, **extra):
    return {"ts": ts, "channel": CHANNEL, "value": value, "sampling": 5, **extra}


@pytest.mark.parametrize("field,value", [
    ("ts", True), ("ts", "0"), ("ts", float("inf")),
    ("value", False), ("value", "1"), ("value", float("nan")),
    ("sampling", 0), ("sampling", -1), ("sampling", True),
    ("sampling", 1.5), ("channel", "UNKNOWN"), ("segment_id", True),
])
def test_frame_rejects_invalid_input(field, value):
    from src.streaming.contracts import TelemetryFrame
    data = frame()
    data[field] = value
    with pytest.raises(ValidationError):
        TelemetryFrame.model_validate(data)


def test_external_source_id_is_not_an_official_boundary():
    from src.streaming.segmenter import OnlineSegmenter
    s = OnlineSegmenter(session_id=str(uuid4()), source_id="external")
    s.feed(0., CHANNEL, 1., 10, 5)
    s.feed(5., CHANNEL, 2., 10, 5)
    result = s.feed(10., CHANNEL, 3., 11, 5)
    assert result.accepted
    assert result.closed.source_segment_id == 10
    assert not result.closed.official
    assert result.closed.boundary_kind == "source_segment"
    assert result.closed.close_reason == "segment_change"


def test_same_time_samples_are_preserved_and_reversed_time_rejected():
    from src.streaming.segmenter import OnlineSegmenter
    s = OnlineSegmenter()
    s.feed(5., CHANNEL, 1., None, 5)
    s.feed(5., CHANNEL, 1., None, 5)
    with pytest.raises(ValueError):
        s.feed(4., CHANNEL, 9., None, 5)
    assert s.open_len(CHANNEL) == 2
    closed = s.flush()[0]
    assert closed.values.tolist() == [1., 1.]


def test_sampling_change_and_reopening_source_segment_are_rejected():
    from src.streaming.segmenter import OnlineSegmenter
    s = OnlineSegmenter()
    s.feed(0., CHANNEL, 1., 1, 5)
    with pytest.raises(ValueError):
        s.feed(5., CHANNEL, 1., 1, 1)
    s.feed(5., CHANNEL, 2., 2, 5)
    with pytest.raises(ValueError):
        s.feed(10., CHANNEL, 3., 1, 5)


def test_segment_size_policy_distinguishes_approximate_and_source_boundaries():
    from src.streaming.segmenter import OnlineSegmenter
    approximate = OnlineSegmenter(max_points=3)
    for i in range(3):
        approximate.feed(float(i), CHANNEL, float(i), None, 1)
    result = approximate.feed(3., CHANNEL, 3., None, 1)
    assert result.accepted
    assert result.closed.close_reason == "max_points"
    assert result.closed.completeness == "complete"
    assert result.closed.boundary_kind == "approximate_window"
    supplied = OnlineSegmenter(max_points=3)
    for i in range(3):
        supplied.feed(float(i), CHANNEL, float(i), 7, 1)
    result = supplied.feed(3., CHANNEL, 3., 7, 1)
    assert not result.accepted
    assert result.closed.close_reason == "source_oversize"
    assert result.closed.completeness == "incomplete"
    assert not supplied.feed(4., CHANNEL, 4., 7, 1).accepted


def test_idle_expiry_and_end_close_only_once():
    from src.streaming.segmenter import OnlineSegmenter
    s = OnlineSegmenter()
    s.feed(0., CHANNEL, 1., 7, 5, wall=10.)
    assert s.expire(39.) == []
    closed = s.expire(40.)
    assert len(closed) == 1
    assert closed[0].close_reason == "idle_timeout"
    assert closed[0].completeness == "incomplete"
    assert s.flush() == []
    assert s.expire(80.) == []


def test_event_identity_changes_across_replay_sessions():
    from src.streaming.segmenter import OnlineSegmenter
    ids = []
    for _ in range(2):
        s = OnlineSegmenter(session_id=str(uuid4()), official=True)
        s.feed(0., CHANNEL, 1., 42, 5)
        ids.append(s.flush()[0].event_id)
    assert ids[0] != ids[1]

@pytest.fixture
def runtime_engine(monkeypatch):
    from src.streaming import engine as runtime
    from src.streaming.event_store import EventStore
    from src.detection.frozen import EngineeringJudge
    instance = runtime.StreamEngine(store=EventStore(":memory:"))
    instance.models_ready = True
    monkeypatch.setattr(instance, "_ensure_broadcaster", lambda: None)
    monkeypatch.setattr(instance, "_prepare_preview", lambda *args: None)
    judged = []
    real_compute = instance._compute_verdict
    thresholds = {"mean": {"mean": 0., "std": 1., "q1": 0., "q3": 0., "iqr": 0.}}
    instance.judges = {CHANNEL: EngineeringJudge(CHANNEL, None, thresholds, k=1, gate_op="rule")}
    def compute(segment):
        judged.append(segment)
        return real_compute(segment)
    monkeypatch.setattr(instance, "_compute_verdict", compute)
    return instance, judged


def test_session_end_retry_and_distinct_sessions(runtime_engine):
    engine, judged = runtime_engine
    async def scenario():
        first = await engine.start_live("sensor")
        sid = first["session_id"]
        points = [frame(float(i * 5), segment_id=7) for i in range(3)]
        result = await engine.ingest_batch(sid, str(uuid4()), points)
        assert result["accepted"] == 3
        ended = await engine.end_live(sid)
        assert ended == await engine.end_live(sid)
        assert len(judged) == 1
        second = await engine.start_live("sensor")
        assert second["session_id"] != sid
        with pytest.raises(ValueError):
            await engine.ingest_batch(sid, str(uuid4()), points)
        assert engine.stats["total_points"] == 0
        await engine.ingest_batch(second["session_id"], str(uuid4()), points)
        await engine.end_live(second["session_id"])
        assert len(judged) == 2
        assert judged[0].event_id != judged[1].event_id
        await engine.shutdown()
    asyncio.run(scenario())


def test_batch_validation_is_atomic_and_retry_does_not_deduplicate_observations(runtime_engine):
    engine, _ = runtime_engine
    async def scenario():
        sid = (await engine.start_live("sensor"))["session_id"]
        bid = str(uuid4())
        points = [frame(5.), frame(5.)]
        first = await engine.ingest_batch(sid, bid, points)
        assert first == await engine.ingest_batch(sid, bid, points)
        assert engine.stats["total_points"] == 2
        with pytest.raises(ValueError):
            await engine.ingest_batch(sid, bid, [frame(5., value=2.)])
        assert engine.stats["total_points"] == 2
        with pytest.raises(ValueError):
            await engine.ingest_batch(sid, str(uuid4()), [frame(10.), frame(4.)])
        assert engine.stats["total_points"] == engine.segmenter.open_len(CHANNEL) == 2
        await engine.ingest_batch(sid, str(uuid4()), points)
        assert engine.stats["total_points"] == 4
        assert engine.segmenter.open_len(CHANNEL) == 4
        await engine.stop_live(sid)
        await engine.shutdown()
    asyncio.run(scenario())


def test_timeout_and_stop_never_formalize_incomplete_or_short_data(runtime_engine):
    import time
    engine, judged = runtime_engine
    async def scenario():
        sid = (await engine.start_live("sensor"))["session_id"]
        await engine.ingest_batch(sid, str(uuid4()), [frame(0.), frame(5.), frame(10.)])
        await engine.expire_external(time.monotonic() + 31)
        assert engine.snapshot()["session"]["status"] == "expired"
        assert not judged
        assert engine.closed_events[-1]["completeness"] == "incomplete"
        with pytest.raises(ValueError):
            await engine.ingest_batch(sid, str(uuid4()), [frame(15.)])
        sid = (await engine.start_live("sensor"))["session_id"]
        await engine.ingest_batch(sid, str(uuid4()), [frame(0.), frame(5.)])
        await engine.end_live(sid)
        assert engine.closed_events[-1]["status"] == "not_applicable"
        assert not judged
        await engine.shutdown()
    asyncio.run(scenario())


def test_concurrent_source_starts_have_one_winner(runtime_engine):
    from src.streaming.contracts import StreamProtocolError
    engine, _ = runtime_engine
    async def scenario():
        result = await asyncio.gather(engine.start_live("one"), engine.start_live("two"), return_exceptions=True)
        assert sum(isinstance(item, dict) for item in result) == 1
        errors = [item for item in result if isinstance(item, StreamProtocolError)]
        assert len(errors) == 1 and errors[0].status_code == 409
        with pytest.raises(StreamProtocolError):
            await engine.start_replay()
        await engine.stop_live(engine.session_id)
        await engine.shutdown()
    asyncio.run(scenario())

def test_live_api_protocol_errors_and_end_flow(runtime_engine, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes import stream
    engine, judged = runtime_engine
    monkeypatch.setattr(stream, "engine", engine)
    app = FastAPI()
    app.include_router(stream.router)
    with TestClient(app) as client:
        assert client.post("/ingest", json={"frames": [frame()]}).status_code == 422
        unknown = {"session_id": str(uuid4()), "batch_id": str(uuid4()), "frames": [frame()]}
        assert client.post("/ingest", json=unknown).status_code == 404
        session = client.post("/live/start", json={"source_id": "test"})
        assert session.status_code == 200
        sid = session.json()["session_id"]
        assert client.post("/live/start", json={"source_id": "second"}).status_code == 409
        assert client.post("/replay/start", json={"speed": 60}).status_code == 409
        payload = {"session_id": sid, "batch_id": str(uuid4()),
                   "frames": [frame(0.), frame(5.), frame(10.)]}
        accepted = client.post("/ingest", json=payload)
        assert accepted.status_code == 200
        assert client.post("/ingest", json=payload).json() == accepted.json()
        ending = client.post("/live/end", json={"session_id": sid})
        assert ending.status_code == 200
        assert ending.json()["closed_events"][0]["status"] == "judged"
        assert client.post("/live/end", json={"session_id": sid}).json() == ending.json()
        assert client.post("/ingest", json=payload).status_code == 409
        assert len(judged) == 1


@pytest.mark.parametrize("bad", [
    {"sampling": -1}, {"ts": "1"}, {"value": True},
    {"channel": "unknown"}, {"extra": "unsupported"},
])
def test_live_api_rejects_invalid_frames_before_counters(runtime_engine, monkeypatch, bad):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes import stream
    engine, _ = runtime_engine
    monkeypatch.setattr(stream, "engine", engine)
    app = FastAPI()
    app.include_router(stream.router)
    with TestClient(app) as client:
        sid = client.post("/live/start", json={"source_id": "test"}).json()["session_id"]
        invalid = frame(5.)
        invalid.update(bad)
        response = client.post("/ingest", json={"session_id": sid, "batch_id": str(uuid4()),
                                               "frames": [frame(), invalid]})
        assert response.status_code == 422
        assert engine.stats["total_points"] == 0
        client.post("/live/stop", json={"session_id": sid})

def test_replay_eof_flushes_but_manual_stop_keeps_partial_unjudged(runtime_engine, monkeypatch):
    import pandas as pd
    engine, judged = runtime_engine
    def rows(times):
        return pd.DataFrame({"ts": times, "channel": CHANNEL, "value": 1.,
                             "segment": 42, "sampling": 1})
    async def scenario():
        monkeypatch.setattr(engine, "_load_segments", lambda: rows([0., 0., 0.]))
        await engine.start_replay(2000)
        await engine._replay_task
        assert len(judged) == 1
        assert engine.closed_events[-1]["close_reason"] == "replay_end"
        judged.clear()
        monkeypatch.setattr(engine, "_load_segments", lambda: rows([0., 0., 0., 60.]))
        ready = asyncio.Event()
        original = engine.ingest
        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            if engine.segmenter.open_len(CHANNEL) >= 3:
                ready.set()
            return result
        monkeypatch.setattr(engine, "ingest", observe)
        await engine.start_replay(1)
        await asyncio.wait_for(ready.wait(), timeout=3)
        await engine.stop_replay()
        assert not judged
        assert engine.closed_events[-1]["close_reason"] == "stop"
        assert engine.closed_events[-1]["completeness"] == "incomplete"
        await engine.shutdown()
    asyncio.run(scenario())


def test_feature_failure_is_recorded_without_a_normal_label(runtime_engine, monkeypatch):
    engine, _ = runtime_engine
    def failed(segment):
        raise ValueError("feature overflow")
    monkeypatch.setattr(engine, "_compute_verdict", failed)
    async def scenario():
        sid = (await engine.start_live("sensor"))["session_id"]
        await engine.ingest_batch(sid, str(uuid4()), [frame(0.), frame(5.), frame(10.)])
        receipt = await engine.end_live(sid)
        record = receipt["closed_events"][0]
        assert record["status"] == "failed"
        assert "feature overflow" in record["error"]
        assert "judgement" not in record
        assert engine.stats["judged_segments"] == 0
        await engine.shutdown()
    asyncio.run(scenario())

def test_idle_monitor_expires_without_another_frame(runtime_engine):
    engine, judged = runtime_engine
    async def scenario():
        sid = (await engine.start_live("sensor"))["session_id"]
        await engine.ingest_batch(sid, str(uuid4()), [frame(0.), frame(5.), frame(10.)])
        engine._sessions[sid]["last_wall"] -= 31
        engine.segmenter._open[CHANNEL].last_wall -= 31
        async def wait_for_expiry():
            while engine.mode == "live":
                await asyncio.sleep(.02)
        await asyncio.wait_for(wait_for_expiry(), timeout=3)
        assert engine._sessions[sid]["status"] == "expired"
        assert engine.closed_events[-1]["close_reason"] == "idle_timeout"
        assert not judged
        await engine.shutdown()
    asyncio.run(scenario())


def test_batch_limit_is_enforced_before_acceptance(runtime_engine, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes import stream
    engine, _ = runtime_engine
    monkeypatch.setattr(stream, "engine", engine)
    app = FastAPI()
    app.include_router(stream.router)
    with TestClient(app) as client:
        sid = client.post("/live/start", json={"source_id": "test"}).json()["session_id"]
        result = client.post("/ingest", json={"session_id": sid, "batch_id": str(uuid4()),
                                             "frames": [frame()] * 1001})
        assert result.status_code == 422
        assert engine.stats["total_points"] == 0
        client.post("/live/stop", json={"session_id": sid})
