"""An explanation must use the stored event and retain its exact evidence."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest


@pytest.fixture
def event_store(tmp_path):
    from src.streaming.engine import StreamEngine
    from src.streaming.event_store import EventStore
    from src.streaming.segmenter import ClosedSegment
    from src.detection.frozen import EngineeringJudge
    store = EventStore(tmp_path / "events.db")
    engine = StreamEngine(store=store)
    engine.models_ready = True
    engine.judges = {"CADC0884": EngineeringJudge("CADC0884", None,
        {"mean": {"mean": 0., "std": 1., "q1": 0., "q3": 0., "iqr": 0.}}, k=1, gate_op="rule")}
    segment = ClosedSegment("CADC0884", "7", False, [100., 200., 100.], [0., 5., 10.], 5)
    engine._judge_segment(segment)
    return store, segment.event_id


def fake_result(status="ready"):
    return {"status": status, "answer": "supported answer", "context": "actual context",
            "sources": [{"content": "actual context", "metadata": {"chunk_id": "one", "page_index": 0}}],
            "prompts": {"system": "system", "user": "actual context"},
            "metadata": {"index_bundle_id": "index-one", "llm_model": "fake"}}


def test_same_event_inputs_persistence_cache_and_restart(event_store):
    from src.rag.event_explanations import EventExplanationService
    from src.streaming.event_store import EventStore
    store, identity = event_store
    pipeline = Mock()
    pipeline.query.return_value = fake_result()
    service = EventExplanationService(store, lambda: pipeline)
    assert service.get(identity)["status"] == "not_requested"
    result = service.generate(identity)
    assert result["status"] == "ready"
    event = store.get_event(identity)
    assert result["input"]["event_id"] == identity
    assert result["input"]["features"] == event["features"]
    assert result["input"]["waveform"] == event["waveform"]
    assert result["result"]["prompts"] == fake_result()["prompts"]
    assert pipeline.query.call_args.kwargs["additional_info"]["event_id"] == identity
    assert service.generate(identity) == result
    assert pipeline.query.call_count == 1
    restarted = EventExplanationService(EventStore(store.path), lambda: pytest.fail("cached explanation called LLM"))
    assert restarted.get(identity) == result


def test_failure_can_retry_and_history_retains_both_attempts(event_store):
    from src.rag.event_explanations import EventExplanationService
    store, identity = event_store
    pipeline = Mock()
    pipeline.query.side_effect = [fake_result("generation_failed"), fake_result()]
    service = EventExplanationService(store, lambda: pipeline)
    failed = service.generate(identity)
    assert failed["status"] == "failed"
    assert failed["result"]["status"] == "generation_failed"
    assert service.generate(identity) == failed
    ready = service.generate(identity, retry=True)
    assert ready["status"] == "ready" and ready["attempt_id"] != failed["attempt_id"]
    assert len(store.explanation_history(identity)) == 2


def test_concurrent_generation_has_one_llm_call(event_store):
    from src.rag.event_explanations import EventExplanationService
    store, identity = event_store
    started, finish = Event(), Event()
    pipeline = Mock()
    def delayed(**kwargs):
        started.set()
        assert finish.wait(5)
        return fake_result()
    pipeline.query.side_effect = delayed
    service = EventExplanationService(store, lambda: pipeline)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.generate, identity)
        assert started.wait(5)
        second = service.generate(identity)
        assert second["status"] == "generating"
        finish.set()
        assert first.result()["status"] == "ready"
    assert pipeline.query.call_count == 1


def test_expired_generation_is_retryable(event_store):
    from src.rag.event_explanations import EventExplanationService
    store, identity = event_store
    service = EventExplanationService(store, lambda: Mock(query=Mock(return_value=fake_result())))
    store.claim_explanation(identity, {"event_id": identity}, lease_seconds=-1)
    assert service.get(identity)["status"] == "failed"
    assert service.generate(identity, retry=True)["status"] == "ready"


def test_unknown_event_does_not_use_official_segment_fallback(event_store):
    from uuid import uuid4
    from src.rag.event_explanations import EventExplanationService
    from src.streaming.contracts import StreamProtocolError
    store, _ = event_store
    service = EventExplanationService(store, lambda: pytest.fail("unknown event used LLM"))
    with pytest.raises(StreamProtocolError) as error:
        service.generate(str(uuid4()))
    assert error.value.status_code == 404


def test_event_explanation_api_and_failure_status(event_store, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes import explanation, stream
    from src.streaming.engine import StreamEngine
    store, identity = event_store
    monkeypatch.setattr(stream, "engine", StreamEngine(store=store))
    pipeline = Mock(query=Mock(return_value=fake_result("generation_failed")))
    monkeypatch.setattr(explanation, "_event_pipeline", lambda: pipeline)
    app = FastAPI()
    app.include_router(explanation.router)
    with TestClient(app) as client:
        url = f"/events/{identity}"
        assert client.get(url).json()["status"] == "not_requested"
        assert client.post(url, json={"retry": "yes"}).status_code == 422
        response = client.post(url, json={})
        assert response.status_code == 503 and response.json()["status"] == "failed"
        pipeline.query.return_value = fake_result()
        ready = client.post(url, json={"retry": True})
        assert ready.status_code == 200 and ready.json()["input"]["event_id"] == identity
        assert client.get(url).json() == ready.json()


@pytest.mark.parametrize("segment", ["LIVE-0001", "2"])
def test_unknown_historical_waveform_returns_missing(segment, monkeypatch):
    import pandas as pd
    from fastapi import HTTPException
    from src.api.routes import explanation
    frame = pd.DataFrame({"channel": ["CADC0872"] * 3, "segment": [1] * 3,
        "anomaly": [1] * 3, "timestamp": ["2022-01-01", "2022-01-02", "2022-01-03"], "value": [1e-5, 2e-5, 1e-5]})
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: frame)
    with pytest.raises(HTTPException) as error:
        explanation.get_waveform(channel="CADC0872", segment=segment, context_size=200)
    assert error.value.status_code == 404
