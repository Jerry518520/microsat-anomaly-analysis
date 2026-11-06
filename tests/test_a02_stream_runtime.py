"""Runtime failures are explicit and cannot train or report ready."""
import asyncio
import importlib

import pytest


@pytest.fixture
def runtime(monkeypatch):
    persistence = importlib.import_module("src.streaming.persistence")
    monkeypatch.setattr(persistence, "init_db", lambda: None)
    monkeypatch.setattr(persistence, "save_verdict", lambda **kwargs: None)
    monkeypatch.setattr(persistence, "save_alert", lambda *a, **k: None)
    return importlib.import_module("src.streaming.engine")


def test_engine_preserves_model_failure_state(runtime, monkeypatch):
    from src.detection.frozen import DetectorUnavailable
    def failed():
        raise DetectorUnavailable("missing frozen model")
    monkeypatch.setattr(runtime, "load_or_train", failed)
    engine = runtime.StreamEngine()
    with pytest.raises(DetectorUnavailable):
        asyncio.run(engine.ensure_models())
    assert not engine.models_ready
    assert "missing frozen model" in engine.snapshot()["model_error"]
    assert all(c["model"] == "unavailable" for c in engine.snapshot()["channels"].values())


@pytest.mark.parametrize("path,payload", [("/state", None), ("/replay/start", {"speed": 60}),
                                         ("/ingest", {"session_id": "00000000-0000-4000-8000-000000000001",
                                          "batch_id": "00000000-0000-4000-8000-000000000002",
                                          "frames": [{"ts": 0., "channel": "CADC0884", "value": 1., "sampling": 5}]})])
def test_routes_return_503_on_package_failure(runtime, monkeypatch, path, payload):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.detection.frozen import DetectorUnavailable
    routes = importlib.import_module("src.api.routes.stream")
    async def failed():
        raise DetectorUnavailable("frozen package invalid")
    monkeypatch.setattr(routes.engine, "ensure_models", failed)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        response = client.get(path) if payload is None else client.post(path, json=payload)
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "detector_unavailable"


def test_closed_segment_requires_ready_detector(runtime):
    from src.streaming.segmenter import ClosedSegment
    from src.detection.frozen import DetectorUnavailable
    engine = runtime.StreamEngine()
    # The detector must not silently drop a closed segment when unavailable.
    seg = ClosedSegment(channel="CADC0884", seg_id="new", official=False, values=[1., 2., 3.], timestamps=[0., 5., 10.], sampling=5)
    with pytest.raises(DetectorUnavailable):
        engine._judge_segment(seg)


def test_runtime_closed_segment_emits_full_formal_facts(runtime, monkeypatch):
    from src.streaming.segmenter import ClosedSegment
    from src.detection.frozen import EngineeringJudge, FEATURE_COLS
    from src.streaming.event_store import EventStore
    engine = runtime.StreamEngine(store=EventStore(":memory:"))
    th = {"mean": {"mean": 0., "std": 1., "q1": 0., "q3": 0., "iqr": 0.}}
    engine.judges = {"CADC0884": EngineeringJudge("CADC0884", None, th, k=1, gate_op="rule")}
    engine.models_ready = True
    seg = ClosedSegment(channel="CADC0884", seg_id="unseen", official=False, values=[100., 200., 100.], timestamps=[0., 5., 10.], sampling=5)
    engine._judge_segment(seg)
    facts = engine.latest["CADC0884"]["judgement"]
    assert facts["gate_pred"] == facts["final_pred"] is True
    assert facts["gate_op"] == "rule" and facts["postprocess"] == "identity"
    assert engine.store.get_event(seg.event_id)["judgement"] == facts
    assert engine.alerts[0]["final_pred"] is True
    assert "k=1" in engine.alerts[0]["summary"]
    short = ClosedSegment(channel="CADC0884", seg_id="short", official=False, values=[1., 2.], timestamps=[0., 5.], sampling=5)
    with pytest.raises(ValueError):
        engine._judge_segment(short)
