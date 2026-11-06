"""Durable identity, migration and atomic event persistence."""
import copy
import json
import sqlite3
from uuid import uuid4

import pytest


def make_session():
    return {"session_id": str(uuid4()), "source_id": "test-source", "kind": "live",
            "status": "active", "started_at": 1., "ended_at": None}


def make_event(session, *, anomaly=True):
    return {"event_id": str(uuid4()), "session_id": session["session_id"],
            "source_id": session["source_id"], "channel": "CADC0884", "segment": "7",
            "source_segment_id": 7, "status": "judged", "completeness": "complete",
            "first_ts": 1., "last_ts": 3., "n_points": 3,
            "waveform": {"timestamps": [1., 2., 3.], "values": [1., 2., 1.]},
            "features": {"mean": 4 / 3},
            "judgement": {"final_pred": anomaly, "severity": "warning", "anomaly_score": .8},
            "alert": {"segment": "7", "channel": "CADC0884", "severity": "warning"} if anomaly else None}


def test_event_idempotency_conflict_and_restart(tmp_path):
    from src.streaming.event_store import EventStore, EventConflict
    path = tmp_path / "events.db"
    store = EventStore(path)
    session = make_session()
    store.save_session(session)
    event = make_event(session)
    assert store.save_events([event]) == 1
    assert store.save_events([copy.deepcopy(event)]) == 0
    changed = copy.deepcopy(event)
    changed["waveform"]["values"][0] = 99.
    with pytest.raises(EventConflict):
        store.save_events([changed])
    store.close()
    restarted = EventStore(path)
    assert restarted.get_event(event["event_id"]) == event
    alerts, total = restarted.query_alerts()
    assert total == len(alerts) == 1
    assert alerts[0]["event_id"] == event["event_id"]
    restarted.close()


def test_alert_failure_rolls_back_event_and_receipt(tmp_path):
    from src.streaming.event_store import EventStore, StoreUnavailable
    store = EventStore(tmp_path / "events.db")
    session = make_session()
    store.save_session(session)
    event = make_event(session)
    store.connection.execute("CREATE TRIGGER fail_alert BEFORE INSERT ON stream_alerts BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
    with pytest.raises(StoreUnavailable):
        store.save_events([event], receipt={"session_id": session["session_id"],
            "batch_id": str(uuid4()), "fingerprint": "abc", "response": {"accepted": 3}})
    assert store.get_event(event["event_id"]) is None
    assert store.connection.execute("SELECT count(*) FROM stream_receipts").fetchone()[0] == 0
    store.close()


def test_nonfinite_payload_rejected_before_any_write(tmp_path):
    from src.streaming.event_store import EventStore
    store = EventStore(tmp_path / "events.db")
    session = make_session()
    store.save_session(session)
    event = make_event(session)
    event["features"]["mean"] = float("nan")
    with pytest.raises(ValueError):
        store.save_events([event])
    assert store.get_event(event["event_id"]) is None
    store.close()


def test_two_instances_preserve_distinct_events_with_same_display_segment(tmp_path):
    from src.streaming.event_store import EventStore
    path = tmp_path / "events.db"
    first, second = EventStore(path), EventStore(path)
    a, b = make_session(), make_session()
    first.save_session(a)
    second.save_session(b)
    one, two = make_event(a), make_event(b)
    two["waveform"]["values"] = [10., 20., 10.]
    first.save_events([one])
    second.save_events([two])
    assert first.get_event(one["event_id"]) == one
    assert first.get_event(two["event_id"]) == two
    assert second.query_alerts()[1] == 2
    first.close()
    second.close()


def test_legacy_migration_preserves_payload_and_stable_mapping(tmp_path):
    from src.streaming.event_store import EventStore
    path = tmp_path / "legacy.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE alerts (id INTEGER PRIMARY KEY, ts REAL, segment TEXT, channel TEXT, severity TEXT, payload TEXT)")
    db.execute("CREATE TABLE segment_verdicts (id INTEGER PRIMARY KEY, segment TEXT, channel TEXT, features TEXT)")
    db.execute("INSERT INTO alerts VALUES (1, 10, 'LIVE-0001', 'CADC0884', 'warning', 'broken-json')")
    db.execute("INSERT INTO segment_verdicts VALUES (1, 'LIVE-0001', 'CADC0884', '{}')")
    db.commit()
    before = db.execute("SELECT * FROM alerts").fetchall()
    db.close()
    store = EventStore(path)
    items, total = store.query_alerts()
    assert total == 1
    assert items[0]["status"] == "legacy_unverified"
    assert items[0]["raw_payload"] == "broken-json"
    identity = items[0]["event_id"]
    assert store.get_event(identity)["waveform"] is None
    assert store.connection.execute("SELECT * FROM alerts").fetchall() == before
    assert store.migration_backup is not None and store.migration_backup.exists()
    store.close()
    restarted = EventStore(path)
    assert restarted.query_alerts()[0][0]["event_id"] == identity
    restarted.close()


def test_engine_storage_failure_preserves_open_segment_and_retry(tmp_path, monkeypatch):
    import asyncio
    from src.streaming.engine import StreamEngine
    from src.streaming.event_store import EventStore, StoreUnavailable
    from src.detection.frozen import EngineeringJudge
    store = EventStore(tmp_path / "runtime.db")
    engine = StreamEngine(store=store)
    engine.models_ready = True
    engine.judges = {"CADC0884": EngineeringJudge("CADC0884", None,
        {"mean": {"mean": 0., "std": 1., "q1": 0., "q3": 0., "iqr": 0.}}, k=1, gate_op="rule")}
    async def scenario():
        sid = (await engine.start_live("sensor"))["session_id"]
        bid = str(uuid4())
        frames = [{"ts": float(i * 5), "channel": "CADC0884", "value": 100.,
                   "segment_id": 7, "sampling": 5} for i in range(3)]
        await engine.ingest_batch(sid, bid, frames)
        store.connection.execute("CREATE TRIGGER fail_alert BEFORE INSERT ON stream_alerts BEGIN SELECT RAISE(ABORT, 'injected'); END")
        with pytest.raises(StoreUnavailable):
            await engine.end_live(sid)
        assert engine.mode == "live"
        assert engine.segmenter.open_len("CADC0884") == 3
        assert engine.stats["judged_segments"] == 0
        assert len(engine.alerts) == 0
        store.connection.execute("DROP TRIGGER fail_alert")
        ended = await engine.end_live(sid)
        assert ended == await engine.end_live(sid)
        event = ended["closed_events"][0]
        assert len(event["features"]) == 18
        assert store.get_event(event["event_id"]) == event
        assert store.query_alerts()[1] == engine.stats["alerts"] == 1
        restarted = StreamEngine(store=EventStore(tmp_path / "runtime.db"))
        assert ended == await restarted.end_live(sid)
        assert restarted.store.get_event(event["event_id"])["waveform"]["values"] == [100.] * 3
        await engine.shutdown()
        await restarted.shutdown()
    asyncio.run(scenario())


def test_query_failure_does_not_report_empty_history(tmp_path):
    from src.streaming.event_store import EventStore, StoreUnavailable
    store = EventStore(tmp_path / "closed.db")
    store.close()
    with pytest.raises(StoreUnavailable):
        store.query_alerts()


def test_event_api_identity_paging_and_database_failure(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes import stream
    from src.streaming.engine import StreamEngine
    from src.streaming.event_store import EventStore
    store = EventStore(tmp_path / "api.db")
    engine = StreamEngine(store=store)
    engine.models_ready = True
    monkeypatch.setattr(stream, "engine", engine)
    session = make_session()
    event = make_event(session)
    store.save_session(session)
    store.save_events([event])
    app = FastAPI()
    app.include_router(stream.router)
    with TestClient(app) as client:
        assert client.get(f"/events/{event['event_id']}").json() == event
        assert client.get(f"/events/{uuid4()}").status_code == 404
        page = client.get("/alerts?limit=5000&offset=-1").json()
        assert page["limit"] == 1000 and page["offset"] == 0 and page["total"] == 1
        store.close()
        assert client.get("/alerts").status_code == 503
        assert client.get(f"/events/{event['event_id']}").status_code == 503
        assert client.post("/live/start", json={"source_id": "test"}).status_code == 503


def test_batch_receipt_failure_leaves_no_accepted_points(tmp_path):
    import asyncio
    from src.streaming.engine import StreamEngine
    from src.streaming.event_store import EventStore, StoreUnavailable
    store = EventStore(tmp_path / "batch.db")
    engine = StreamEngine(store=store)
    engine.models_ready = True
    async def scenario():
        sid = (await engine.start_live("test"))["session_id"]
        bid = str(uuid4())
        frames = [{"ts": 1., "value": 2., "channel": "CADC0884"}]
        store.connection.execute("CREATE TRIGGER fail_receipt BEFORE INSERT ON stream_receipts BEGIN SELECT RAISE(ABORT, 'injected'); END")
        with pytest.raises(StoreUnavailable):
            await engine.ingest_batch(sid, bid, frames)
        assert engine.stats["total_points"] == engine.segmenter.open_len("CADC0884") == 0
        store.connection.execute("DROP TRIGGER fail_receipt")
        result = await engine.ingest_batch(sid, bid, frames)
        assert result == await engine.ingest_batch(sid, bid, frames)
        assert engine.stats["total_points"] == 1
        await engine.shutdown()
    asyncio.run(scenario())


def test_two_real_engines_and_exact_event_retry(tmp_path):
    from src.streaming.engine import StreamEngine
    from src.streaming.event_store import EventStore
    from src.streaming.segmenter import ClosedSegment
    from src.detection.frozen import EngineeringJudge
    identities = []
    for value in (100., 200.):
        engine = StreamEngine(store=EventStore(tmp_path / "engines.db"))
        engine.models_ready = True
        engine.judges = {"CADC0884": EngineeringJudge("CADC0884", None,
            {"mean": {"mean": 0., "std": 1., "q1": 0., "q3": 0., "iqr": 0.}}, k=1, gate_op="rule")}
        segment = ClosedSegment("CADC0884", "LIVE-0001", False, [value] * 3, [0., 5., 10.], 5)
        engine._judge_segment(segment)
        engine._judge_segment(segment)
        assert engine.stats["judged_segments"] == engine.stats["alerts"] == 1
        identities.append(segment.event_id)
    assert identities[0] != identities[1]
    assert engine.store.stats()["verdicts"] == engine.store.stats()["alerts"] == 2
    assert engine.store.get_event(identities[0])["waveform"]["values"] == [100.] * 3
    assert engine.store.get_event(identities[1])["waveform"]["values"] == [200.] * 3
