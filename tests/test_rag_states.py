"""Evidence absence and technical failures must have different states."""
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("state", ["ready", "insufficient_knowledge", "retrieval_failed", "generation_failed"])
def test_pipeline_states_and_actual_prompts(state):
    from src.rag.pipeline import RAGPipeline
    source = {"content": "current evidence", "metadata": {"chunk_id": "chunk-one"}, "score": .9}
    vectorstore = Mock()
    vectorstore.index = object()
    vectorstore.bundle_id = "index-one"
    vectorstore.search_with_context.return_value = ("current evidence", [source])
    llm = Mock()
    llm.model = "test-model"
    llm.generate.return_value = "evidence-based explanation"
    if state == "insufficient_knowledge":
        vectorstore.search_with_context.return_value = ("", [])
    if state == "retrieval_failed":
        vectorstore.search_with_context.side_effect = RuntimeError("retrieval unavailable")
    if state == "generation_failed":
        llm.generate.side_effect = RuntimeError("generation unavailable")
    pipeline = RAGPipeline(embedder=Mock(), vectorstore=vectorstore, llm_client=llm)
    result = pipeline.query("question", score_threshold=0.)
    assert result["status"] == state
    assert vectorstore.search_with_context.call_args.kwargs["score_threshold"] == 0.
    if state == "ready":
        assert result["prompts"]["user"] == llm.generate.call_args.kwargs["prompt"]
        assert result["prompts"]["system"] == llm.generate.call_args.kwargs["system_prompt"]
        assert result["metadata"]["index_bundle_id"] == "index-one"
    if state in {"insufficient_knowledge", "retrieval_failed"}:
        llm.generate.assert_not_called()
    if state == "generation_failed":
        assert result["sources"] == [source] and result["context"] == "current evidence"


def test_low_confidence_and_citations_use_actual_context():
    from src.rag.pipeline import RAGPipeline
    source = {"content": "actual", "low_confidence": True,
              "metadata": {"filename": "actual.pdf", "page_label": "1", "chunk_id": "abc"}}
    store = Mock(index=object(), bundle_id="index")
    store.search_with_context.return_value = ("actual context", [source])
    llm = Mock(model="fake")
    llm.generate.return_value = "analysis\n\n**【来源】**\n1. [invented.pdf p.99]"
    result = RAGPipeline(embedder=Mock(), vectorstore=store, llm_client=llm).query("question")
    assert result["metadata"]["retrieval_confidence"] == "low"
    assert len(result["metadata"]["prompt_sha256"]) == 64
    assert result["retrieved_candidates"] == [source]
    assert "invented.pdf" not in result["answer"] and "actual.pdf" in result["answer"]
    assert "invented.pdf" in result["raw_answer"]
