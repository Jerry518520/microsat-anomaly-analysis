"""Build and atomically publish a source-verified knowledge index."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import numpy as np
import yaml
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from src.rag.embedding import get_embedder
from src.rag.vectorstore import FAISSVectorStore
from src.rag.retrieval_policy import is_low_quality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rag_config.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    os.chdir(ROOT)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    embedder = get_embedder(str(args.config))
    store = FAISSVectorStore(str(args.config), embedder=embedder)
    documents = []
    for source in config["document"]["knowledge_base"]:
        path = Path(source["file_path"])
        path = path if path.is_absolute() else ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"configured source missing: {path}")
        if path.suffix.lower() == ".pdf":
            pages = PyPDFLoader(str(path)).load()
        elif path.suffix.lower() == ".md":
            pages = [Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})]
        else:
            raise ValueError(f"unsupported source type: {path}")
        if not pages:
            raise ValueError(f"source has no readable pages: {path}")
        for page in pages:
            page.metadata.update({"source": str(path), "source_name": source["name"],
                "filename": path.name, "priority": source["priority"], "language": source.get("language", "unknown")})
        documents.extend(pages)
        print(f"Loaded {source['name']}: {len(pages)} pages", flush=True)
    chunks = store.text_splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if not is_low_quality(chunk.page_content,
        store.min_chunk_chars, store.max_junk_ratio)]
    if not chunks:
        raise ValueError("no usable chunks")
    print(f"Encoding {len(chunks)} chunks", flush=True)
    vectors = embedder.encode([chunk.page_content for chunk in chunks],
        batch_size=args.batch_size, normalize=embedder.normalize_embeddings, show_progress_bar=True)
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    if not np.isfinite(vectors).all():
        raise ValueError("encoder returned nonfinite vectors")
    store.index = store._create_faiss_index(vectors)
    store.documents = chunks
    store.document_metadata = [dict(chunk.metadata) for chunk in chunks]
    output = args.output or Path(store.persist_directory)
    store.save(str(output))
    assert store.load(str(output)), store.load_error
    print(f"Published verified index {store.bundle_id}: {len(store.documents)} chunks", flush=True)


if __name__ == "__main__":
    main()
