"""Versioned index publication. CURRENT changes only after all files are ready."""
from collections import Counter
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import yaml
from uuid import uuid4

import faiss

ROOT = Path(__file__).resolve().parents[2]


def configuration_key(config_path=None, *, embedding_only=False):
    path = Path(config_path) if config_path else ROOT / "configs/rag_config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_path = os.getenv("EMBEDDING_MODEL_PATH", "")
    if embedding_only:
        config = config.get("embedding", {})
        weights = []
        if model_path and Path(model_path).is_dir():
            for pattern in ("*.bin", "*.safetensors", "*.json", "*/*.json"):
                weights.extend((str(item), file_hash(item)) for item in sorted(Path(model_path).glob(pattern)))
        return _json([config, model_path, weights])
    directory = Path(config.get("vectorstore", {}).get("persist_directory", "data/vectorstore"))
    directory = directory if directory.is_absolute() else ROOT / directory
    pointer = directory / "CURRENT"
    key_name = config.get("llm", {}).get("api_key_env", "NVIDIA_API_KEY")
    credential_id = hashlib.sha256(os.getenv(key_name, "").encode()).hexdigest()
    return _json([str(path.resolve()), config, model_path, credential_id,
                  file_hash(pointer) if pointer.exists() else None])


@lru_cache(maxsize=128)
def _cached_hash(path, size, modified):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hash(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _cached_hash(str(path), stat.st_size, stat.st_mtime_ns)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _path_name(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def source_manifest(config, metadata):
    paths = {str(item["file_path"]): dict(item) for item in config.get("document", {}).get("knowledge_base", [])}
    for item in metadata:
        source = item.get("source")
        if source and Path(source).is_file():
            paths.setdefault(str(source), {"file_path": str(source)})
    by_path = {}
    for entry in paths.values():
        path = Path(entry["file_path"])
        path = path if path.is_absolute() else ROOT / path
        name = _path_name(path)
        # Configured sources must all exist. A partial build must fail.
        by_path.setdefault(name, {**entry, "file_path": name, "sha256": file_hash(path)})
    return [by_path[name] for name in sorted(by_path)]


def embedding_manifest(embedder):
    name = getattr(embedder, "model_name", type(embedder).__name__)
    name = name if isinstance(name, str) else type(embedder).__name__
    model = Path(name)
    files = {}
    if model.is_dir():
        for pattern in ("*.json", "*.bin", "*.safetensors", "*/*.json"):
            for path in sorted(model.glob(pattern)):
                files[path.relative_to(model).as_posix()] = file_hash(path)
    return {"model": model.name if model.is_dir() else name, "files": files,
            "dimension": int(embedder.get_embedding_dimension()),
            "normalize": bool(getattr(embedder, "normalize_embeddings", True)),
            "max_length": int(getattr(embedder, "max_length", 512)),
            "encoder_source": file_hash(ROOT / "src/rag/embedding.py"),
            "libraries": {name: importlib.metadata.version(name) for name in ("sentence-transformers", "transformers", "torch", "numpy")}}


def assign_chunk_ids(documents, metadata, sources):
    hashes = {item["file_path"]: item["sha256"] for item in sources}
    occurrences = Counter()
    result = []
    for document, original in zip(documents, metadata):
        content = document if isinstance(document, str) else document.page_content
        item = dict(original)
        source = item.get("source")
        source_name = _path_name(source) if source and Path(source).is_file() else item.get("filename", "inline")
        source_sha = hashes.get(source_name, hashlib.sha256(content.encode()).hexdigest())
        page = item.get("page") if str(item.get("filename", "")).lower().endswith(".pdf") else None
        occurrence = occurrences[(source_name, page)]
        occurrences[(source_name, page)] += 1
        identity = [source_name, source_sha, page, occurrence, content]
        item.update({"source_sha256": source_sha, "source_id": source_name,
                     "chunk_id": hashlib.sha256(_json(identity).encode()).hexdigest(),
                     "page_index": page, "page_label": str(page + 1) if page is not None else None})
        result.append(item)
    return result


def publish(directory, index, documents, metadata, config, embedder):
    if index is None or index.ntotal != len(documents) or len(documents) != len(metadata):
        raise ValueError("index/document/metadata counts disagree")
    sources = source_manifest(config, metadata)
    tagged = assign_chunk_ids(documents, metadata, sources)
    dimension = embedder.get_embedding_dimension()
    if index.d != dimension:
        raise ValueError("index and encoder dimensions disagree")
    directory = Path(directory).resolve()
    generations = directory / "generations"
    generation = generations / uuid4().hex
    generation.mkdir(parents=True)
    (generation / "faiss_index.bin").write_bytes(faiss.serialize_index(index).tobytes())
    with (generation / "documents_metadata.pkl").open("wb") as stream:
        pickle.dump({"documents": documents, "metadata": tagged}, stream)
    manifest = {"schema": 1, "count": len(documents), "dimension": index.d,
                "embedding": embedding_manifest(embedder), "sources": sources,
                "chunking": config.get("document", {}).get("chunking", {}),
                "metric": config.get("vectorstore", {}).get("metric", "cosine"),
                "files": {name: file_hash(generation / name) for name in ("faiss_index.bin", "documents_metadata.pkl")}}
    raw = _json(manifest).encode()
    (generation / "manifest.json").write_bytes(raw)
    pointer = {"generation": generation.name, "manifest_sha256": hashlib.sha256(raw).hexdigest()}
    temporary = directory / f"CURRENT.{uuid4().hex}.tmp"
    temporary.write_text(_json(pointer), encoding="utf-8")
    os.replace(temporary, directory / "CURRENT")
    return manifest, pointer["manifest_sha256"], tagged


def read(directory, config=None, embedder=None):
    directory = Path(directory).resolve()
    pointer = json.loads((directory / "CURRENT").read_text(encoding="utf-8"))
    generation = directory / "generations" / pointer["generation"]
    if generation.resolve().parent != (directory / "generations").resolve():
        raise ValueError("invalid index generation")
    raw = (generation / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != pointer["manifest_sha256"]:
        raise ValueError("index manifest checksum mismatch")
    manifest = json.loads(raw)
    if manifest["schema"] != 1:
        raise ValueError("unsupported index schema")
    for name, expected in manifest["files"].items():
        if file_hash(generation / name) != expected:
            raise ValueError(f"index artifact checksum mismatch: {name}")
    validate_sources(manifest)
    if config is not None:
        if manifest["chunking"] != config.get("document", {}).get("chunking", {}):
            raise ValueError("index chunking mismatch")
        if manifest["metric"] != config.get("vectorstore", {}).get("metric", "cosine"):
            raise ValueError("index metric mismatch")
        expected_sources = source_manifest(config, [])
        actual_sources = {item["file_path"]: item for item in manifest["sources"]}
        if any(actual_sources.get(item["file_path"]) != item for item in expected_sources):
            raise ValueError("configured index sources mismatch")
    if embedder is not None and manifest["embedding"] != embedding_manifest(embedder):
        raise ValueError("index encoder mismatch")
    return generation, manifest, pointer["manifest_sha256"]


def validate_sources(manifest):
    for item in manifest["sources"]:
        path = Path(item["file_path"])
        path = path if path.is_absolute() else ROOT / path
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"index source changed: {item['file_path']}")
