"""Fixed factual retrieval comparisons. This script makes no LLM calls."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from src.evaluation.retrieval_metrics import mean_reciprocal_rank
from src.rag.vectorstore import get_vectorstore


def evaluate():
    spec_path = ROOT / "docs/contracts/current_retrieval_queries_v1.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    store = get_vectorstore()
    assert store.load(), store.load_error
    original_caps = store.priority_caps
    original_reserved, original_priority = store.max_reserved, store.reserve_priority_max
    profiles = [("vector", False, False, False), ("keyword", True, False, False),
                ("runtime_filter", True, True, False), ("source_policy", True, True, True)]
    all_rows, summary = [], []
    expected = set(spec["expected_source_ids"])
    def covered(chunk, groups):
        return chunk["metadata"].get("source_id") in expected and all(
            any(term.lower() in chunk["content"].lower() for term in group) for group in groups)
    for name, hybrid, filtered, policy in profiles:
        store.hybrid_search, store.filter_low_quality = hybrid, filtered
        store.keyword_boost = spec["keyword_boost"]
        store.priority_caps = original_caps if policy else {priority: 100 for priority in range(10)}
        store.max_reserved = original_reserved if policy else 0
        store.reserve_priority_max = original_priority if policy else -1
        rows = []
        for query in spec["queries"]:
            context, sources = store.search_with_context(query["q"], top_k=spec["top_k"], max_tokens=spec["context_budget"])
            candidates = sources.candidates
            first = next((rank for rank, chunk in enumerate(candidates, 1) if covered(chunk, query["answers"])), None)
            actual = next((rank for rank, chunk in enumerate(sources, 1) if covered(chunk, query["answers"])), None)
            row = {"profile": name, "query_id": query["id"], "query": query["q"],
                   "source_hit": any(chunk["metadata"].get("source_id") in expected for chunk in candidates),
                   "candidate_answer_hit": first is not None, "context_answer_hit": actual is not None,
                   "first_retrieved_rank": first, "first_context_rank": actual,
                   "candidates": candidates, "context_sources": list(sources), "context": context}
            rows.append(row)
        summary.append({"profile": name, "n": len(rows),
                        "source_hit_rate": sum(row["source_hit"] for row in rows) / len(rows),
                        "candidate_answer_coverage": sum(row["candidate_answer_hit"] for row in rows) / len(rows),
                        "context_answer_coverage": sum(row["context_answer_hit"] for row in rows) / len(rows),
                        "candidate_mrr": mean_reciprocal_rank(row["first_retrieved_rank"] for row in rows),
                        "context_mrr": mean_reciprocal_rank(row["first_context_rank"] for row in rows)})
        all_rows.extend(rows)
    output = ROOT / "data/results/acceptance/A10"
    output.mkdir(parents=True, exist_ok=True)
    report = {"query_version": spec["version"], "query_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
              "scope": spec["scope"], "build_filter_limitation": spec["index_build_filter"],
              "index_bundle_id": store.bundle_id, "config_sha256": hashlib.sha256((ROOT / "configs/rag_config.yaml").read_bytes()).hexdigest(),
              "top_k": spec["top_k"], "context_budget": spec["context_budget"], "keyword_boost": spec["keyword_boost"],
              "source_policy": {"priority_caps": original_caps, "max_reserved": original_reserved, "reserve_priority_max": original_priority},
              "summary": summary, "results": all_rows}
    (output / "current_retrieval_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "current_retrieval_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    # Independently recompute all aggregate values from the saved rows.
    saved = json.loads((output / "current_retrieval_comparison.json").read_text(encoding="utf-8"))
    for aggregate in saved["summary"]:
        rows = [row for row in saved["results"] if row["profile"] == aggregate["profile"]]
        assert aggregate["candidate_mrr"] == mean_reciprocal_rank(row["first_retrieved_rank"] for row in rows)
        assert aggregate["context_mrr"] == mean_reciprocal_rank(row["first_context_rank"] for row in rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    evaluate()
