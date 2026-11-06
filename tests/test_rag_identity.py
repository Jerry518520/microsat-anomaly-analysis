"""Index publication, stale-cache and actual-context source regressions."""
import json
from pathlib import Path
from unittest.mock import Mock

import faiss
import numpy as np
import pytest
import yaml
from langchain_core.documents import Document


class Embedder:
    model_name = "test-encoder"
    normalize_embeddings = True
    max_length = 512

    def get_embedding_dimension(self):
        return 2

    def encode(self, texts, **kwargs):
        return np.tile([[1., 0.]], (1 if isinstance(texts, str) else len(texts), 1)).astype("float32")

    def clear_cache(self):
        pass


@pytest.fixture
def store(tmp_path):
    from src.rag.vectorstore import FAISSVectorStore
    source = tmp_path / "source.md"
    source.write_text("source content", encoding="utf-8")
    config = {"vectorstore": {"persist_directory": str(tmp_path / "index"), "auto_save": False},
              "document": {"knowledge_base": [{"name": "source", "file_path": str(source), "priority": 0}]}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    instance = FAISSVectorStore(str(config_path), embedder=Embedder())
    instance.index = faiss.IndexFlatIP(2)
    instance.index.add(np.array([[1., 0.]], dtype="float32"))
    instance.documents = [Document(page_content="source content", metadata={"source": str(source), "filename": "source.md"})]
    instance.document_metadata = [dict(instance.documents[0].metadata)]
    return instance, config_path, source


def test_atomic_bundle_identity_stale_source_and_bm25_invalidation(store):
    from src.rag.vectorstore import FAISSVectorStore
    instance, config, source = store
    instance.save()
    pointer = Path(instance.persist_directory) / "CURRENT"
    assert pointer.exists()
    old_pointer = pointer.read_bytes()
    loaded = FAISSVectorStore(str(config), embedder=Embedder())
    loaded._bm25 = object()
    assert loaded.load()
    assert loaded._bm25 is None
    assert len(loaded.document_metadata[0]["chunk_id"]) == 64
    assert loaded.bundle_id == instance.bundle_id
    instance.index.add(np.array([[1., 0.]], dtype="float32"))
    with pytest.raises(ValueError):
        instance.save()
    assert pointer.read_bytes() == old_pointer
    source.write_text("changed source", encoding="utf-8")
    assert not loaded.load()
    assert loaded.index is None and loaded.documents == [] and loaded._bm25 is None
    assert "source" in loaded.load_error.lower()


def test_embedding_cache_separates_normalization():
    from src.rag.embedding import BGE_M3_Embedder
    embedder = BGE_M3_Embedder.__new__(BGE_M3_Embedder)
    embedder.cache = {}
    embedder.batch_size = 1
    embedder.embedding_config = {}
    embedder.normalize_embeddings = True
    embedder.model_name = "test"
    embedder.max_length = 512
    embedder.model = Mock()
    embedder.model.encode.side_effect = lambda texts, **kwargs: np.array([[.6, .8] if kwargs["normalize_embeddings"] else [3., 4.]])
    first = embedder.encode("same", normalize=False)
    second = embedder.encode("same", normalize=True)
    np.testing.assert_array_equal(first, [[3., 4.]])
    np.testing.assert_array_equal(second, [[.6, .8]])


def test_keyword_only_candidate_receives_boost_once(store, monkeypatch):
    instance, _, _ = store
    instance.hybrid_search = True
    instance.keyword_boost = .02
    instance.filter_low_quality = False
    instance.documents = ["first", "second"]
    instance.document_metadata = [{}, {}]
    bm25 = Mock()
    bm25.top_n.return_value = [(0, 1.), (1, 1.)]
    monkeypatch.setattr(instance, "_ensure_bm25", lambda: bm25)
    result = instance._merge_keyword_candidates("x", [{"_pos": 0, "score": .5, "content": "first", "metadata": {}}])
    assert result[0]["score"] == .52
    assert result[1]["score"] == .02


def test_context_returns_only_used_sources_and_one_based_page(store, monkeypatch):
    instance, _, _ = store
    results = [{"content": "a" * 40, "score": .9, "metadata": {"filename": "x.pdf", "page": 0, "chunk_id": "one"}},
               {"content": "b" * 1000, "score": .8, "metadata": {"filename": "y.pdf", "page": 3, "chunk_id": "two"}}]
    search = Mock(return_value=results)
    monkeypatch.setattr(instance, "search", search)
    context, sources = instance.search_with_context("query", top_k=2, max_tokens=80, score_threshold=0.)
    assert len(sources) == 1 and sources[0]["metadata"]["chunk_id"] == "one"
    assert len(sources.candidates) == 2
    assert "第 1 页" in context and "y.pdf" not in context
    assert search.call_args.kwargs["score_threshold"] == 0.
    assert instance.search_with_context("query", max_tokens=0) == ("", [])


def test_index_bundle_handles_chinese_directory(store, tmp_path):
    instance, _, _ = store
    directory = tmp_path / "中文索引"
    instance.save(str(directory))
    assert instance.load(str(directory)), instance.load_error


def test_prompt_and_llm_cache_follow_configuration(tmp_path, monkeypatch):
    from src.rag import prompts, llm_client
    config = tmp_path / "config.yaml"
    monkeypatch.setattr(prompts, "_global_templates", None)
    monkeypatch.setattr(llm_client, "_global_llm_client", None)
    factory = Mock(side_effect=[object(), object()])
    monkeypatch.setattr(llm_client, "LLMClient", factory)
    config.write_text(yaml.safe_dump({"prompt": {"system_prompt": "first"},
                                     "llm": {"api_key_env": "RAG_TEST_ROTATION"}}), encoding="utf-8")
    assert prompts.get_prompt_templates(str(config)).system_prompt == "first"
    config.write_text(yaml.safe_dump({"prompt": {"system_prompt": "second"},
                                     "llm": {"api_key_env": "RAG_TEST_ROTATION"}}), encoding="utf-8")
    assert prompts.get_prompt_templates(str(config)).system_prompt == "second"
    monkeypatch.setenv("RAG_TEST_ROTATION", "first")
    first = llm_client.get_llm_client(str(config))
    monkeypatch.setenv("RAG_TEST_ROTATION", "second")
    assert llm_client.get_llm_client(str(config)) is not first


def test_append_reload_clear_rebuild_keyword_locations(store, monkeypatch, tmp_path):
    instance, _, _ = store
    instance.hybrid_search = True
    assert instance._ensure_bm25() is not None
    second = tmp_path / "second.pdf"
    second.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(instance, "_load_pdf", lambda path: [Document(page_content="uniquewordtest",
        metadata={"source": str(second), "filename": "second.pdf", "page": 0})])
    assert instance.add_documents(str(second)) == 1
    assert instance._bm25 is None
    assert instance._ensure_bm25().top_n("uniquewordtest", 1)[0][0] == 1
    instance.save()
    assert instance.load()
    assert instance._bm25 is None
    assert instance._ensure_bm25().top_n("uniquewordtest", 1)[0][0] == 1
    instance.clear()
    assert instance._bm25 is None and instance._ensure_bm25() is None
    assert instance.add_documents(str(second)) == 1
    assert instance._ensure_bm25().top_n("uniquewordtest", 1)[0][0] == 0


def test_failed_publication_keeps_previous_generation(store, monkeypatch):
    from src.rag import index_bundle
    instance, _, _ = store
    instance.save()
    pointer = Path(instance.persist_directory) / "CURRENT"
    previous = pointer.read_bytes()
    def fail_replace(*args):
        raise OSError("injected publication failure")
    monkeypatch.setattr(index_bundle.os, "replace", fail_replace)
    with pytest.raises(OSError):
        instance.save()
    assert pointer.read_bytes() == previous
    assert instance.load()


def test_encoder_change_is_rejected_on_load(store):
    instance, _, _ = store
    instance.save()
    instance.embedder.model_name = "different-encoder"
    assert not instance.load()
    assert "encoder" in instance.load_error
