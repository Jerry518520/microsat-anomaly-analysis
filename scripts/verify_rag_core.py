"""Audit actual indexed pages and exercise event explanations through HTTP."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify(output, database, generate, only_event=False):
    from pypdf import PdfReader
    from src.rag.embedding import get_embedder
    from src.rag.vectorstore import FAISSVectorStore
    from src.rag.pipeline import RAGPipeline
    from src.rag.index_bundle import read
    _, manifest, bundle_id = read(ROOT / "data/vectorstore")
    embedder = get_embedder()
    store = FAISSVectorStore(embedder=embedder)
    assert store.load(), store.load_error
    page_samples = []
    for source in manifest["sources"]:
        name = source["file_path"]
        path = Path(name)
        path = path if path.is_absolute() else ROOT / path
        positions = [i for i, metadata in enumerate(store.document_metadata) if metadata["source_id"] == name]
        assert positions, name
        selected, pages = [], set()
        for i in positions:
            page = store.document_metadata[i]["page_index"]
            if page in pages and path.suffix.lower() == ".pdf":
                continue
            selected.append(i)
            pages.add(page)
            if len(selected) == 2:
                break
        reader = PdfReader(str(path)) if path.suffix.lower() == ".pdf" else None
        for i in selected:
            metadata = store.document_metadata[i]
            content = store._content_at(i)
            original = reader.pages[metadata["page_index"]].extract_text() if reader else path.read_text(encoding="utf-8")
            assert "".join(content.split()) in "".join(original.split()), (name, metadata["page_index"])
            if reader:
                assert metadata["page_label"] == str(metadata["page_index"] + 1)
            page_samples.append({"source": name, "source_sha256": source["sha256"],
                "chunk_id": metadata["chunk_id"], "page_index": metadata["page_index"],
                "page_label": metadata["page_label"], "excerpt": content[:180]})
    queries = []
    for query in ("CADC0872 I_B_FB_MM_0 的轴向和单位是什么", "CADC0873 I_B_FB_MM_1",
                  "CADC0874 I_B_FB_MM_2", "项目当前工程检测 F1 0.6204"):
        context, sources = store.search_with_context(query, top_k=5, max_tokens=1600)
        assert context and sources
        assert all(source["content"] in context and source["metadata"]["chunk_id"] in context for source in sources)
        queries.append({"query": query, "context": context, "sources": sources})
    write_json(output / "index_and_pages.json", {"status": "passed", "bundle_id": bundle_id,
        "chunks": len(store.documents), "source_count": len(manifest["sources"]),
        "page_samples": page_samples, "queries": queries})
    write_json(output / "index_manifest.json", manifest)
    if not generate:
        return {"index_and_pages": "passed", "generation": "not_requested"}
    write_json(output / "live_service_budget.json", {"planned_calls": 1 if only_event else 2, "maximum_calls": 1 if only_event else 3,
        "max_output_tokens_per_call": 1024, "attempts_per_call": 1, "timeout_seconds": 45})
    from src.rag.llm_client import get_llm_client
    llm = get_llm_client()
    llm.max_tokens, llm.max_retries, llm.timeout = 1024, 1, 45
    pipeline = RAGPipeline(embedder=embedder, vectorstore=store, llm_client=llm)
    pipeline.max_tokens = 1600
    if not only_event:
        facts = pipeline.query("请核对 CADC0872、CADC0873、CADC0874 与 I_B_FB_MM_0/1/2 的对应关系。资料是否明确轴向和单位？", top_k=5)
        write_json(output / "real_fact_query.json", facts)
        assert facts["status"] == "ready", facts.get("metadata")

    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes import stream, explanation
    from src.api.security import require_api_key, rate_limit
    from src.streaming.engine import StreamEngine
    from src.streaming.event_store import EventStore
    from src.detection.frozen import load_bundle
    if database.exists():
        raise ValueError("use a new isolated event database")
    runtime = StreamEngine(store=EventStore(database))
    runtime.judges, runtime.models_ready = load_bundle(), True
    async def create_event():
        sid = (await runtime.start_live("real-rag-verification"))["session_id"]
        await runtime.ingest_batch(sid, str(uuid4()), [{"ts": float(i * 5), "channel": "CADC0884",
            "value": value, "segment_id": 7, "sampling": 5} for i, value in enumerate([1e9, -1e9, 1e9])])
        record = (await runtime.end_live(sid))["closed_events"][0]
        await runtime.shutdown()
        return record
    event = asyncio.run(create_event())
    app = FastAPI(dependencies=[Depends(require_api_key), Depends(rate_limit)])
    app.include_router(explanation.router, prefix="/api/explanation")
    key = str(uuid4())
    with patch.dict(os.environ, {"API_AUTH_KEYS": key, "API_AUTH_DISABLED": "false"}), \
         patch.object(stream, "engine", runtime), patch.object(explanation, "_event_pipeline", lambda: pipeline):
        with TestClient(app) as client:
            url = f"/api/explanation/events/{event['event_id']}"
            headers = {"X-API-Key": key}
            assert client.get(url).status_code == 401
            before = client.get(url, headers=headers).json()
            assert before["status"] == "not_requested"
            response = client.post(url, headers=headers, json={})
            record = response.json()
            write_json(output / "real_event_explanation.json", record)
            assert response.status_code == 200 and record["status"] == "ready", record
            assert record["input"]["event_id"] == event["event_id"]
            assert record["input"]["features"] == event["features"]
            assert record["input"]["waveform"] == event["waveform"]
            assert client.get(url, headers=headers).json() == record
            assert client.post(url, headers=headers, json={}).json() == record
    restarted = EventStore(database)
    assert restarted.get_explanation(event["event_id"]) == record
    result = {"status": "passed", "index_bundle_id": bundle_id, "indexed_chunks": len(store.documents),
              "source_count": len(manifest["sources"]), "sampled_pages_and_fragments": len(page_samples),
              "real_generation_calls": llm.total_requests, "llm_model": llm.model,
              "event_id": event["event_id"], "same_event_features_waveform": True,
              "http_auth_enabled": True, "cached_retry": True, "restart_read": True}
    write_json(output / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/results/acceptance/A11-A12")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--only-event", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.database, args.generate, args.only_event), ensure_ascii=False, indent=2))
