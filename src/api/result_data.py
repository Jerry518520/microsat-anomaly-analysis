"""One verified result package for the current API metrics."""
import hashlib
import json
from pathlib import Path

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data/results/graduation-confirmation-v1"


def confirmed_results():
    try:
        manifest = json.loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
        raw = (RESULTS / "metrics.json").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if manifest["status"] != "complete" or manifest["artifacts"]["metrics.json"] != digest:
            raise ValueError("current result checksum mismatch")
        metrics = json.loads(raw)
        source = {"protocol": manifest["protocol"], "feature_version": "18d-canonical-v1",
                  "detector_version": "a02-engineering-gate-v1",
                  "bundle_id": manifest["engineering_bundle"]["bundle_id"],
                  "benchmark_previously_seen": manifest["benchmark_previously_seen"],
                  "benchmark_rows": manifest["benchmark_rows"], "metrics_sha256": digest,
                  "file": "data/results/graduation-confirmation-v1/metrics.json"}
        return metrics, source
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail={"error": "current_results_unavailable", "message": str(exc)}) from exc
