"""Recompute metrics and audit split isolation without training any detector."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from src.evaluation.saved_predictions import evaluate_file


def verify(folder, repeat):
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    for path, digest in manifest["inputs_and_sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
    for filename, metrics_name, expected_methods, expected_rows in (
        ("predictions.csv", "metrics.json", 8, 529), ("nested_predictions.csv", "nested_metrics.json", 7, 1594)):
        predictions = pd.read_csv(folder / filename, float_precision="round_trip")
        assert predictions.method.nunique() == expected_methods
        assert predictions.groupby("method").size().eq(expected_rows).all()
        stored = json.loads((folder / metrics_name).read_text(encoding="utf-8"))
        recomputed = evaluate_file(folder / filename)
        assert all(stored[key] == value for key, value in recomputed.items())
        for method, rows in predictions.groupby("method"):
            metric = stored["overall"][method]
            reference = {"precision": precision_score(rows.truth, rows.final_pred, zero_division=0),
                         "recall": recall_score(rows.truth, rows.final_pred, zero_division=0),
                         "f1": f1_score(rows.truth, rows.final_pred, zero_division=0),
                         "accuracy": accuracy_score(rows.truth, rows.final_pred),
                         "mcc": matthews_corrcoef(rows.truth, rows.final_pred)}
            assert all(abs(metric[key] - value) <= 1e-12 for key, value in reference.items())
    splits = json.loads((folder / "split_manifest.json").read_text(encoding="utf-8"))
    fixed = splits["fixed"]
    train_ids = set(fixed["fit_ids"]) | set(fixed["val_ids"])
    test_ids = set(fixed["test_ids"])
    seen = set()
    for fold in splits["nested"]:
        fit, val, training, held = [set(fold[key]) for key in ("inner_fit_ids", "inner_val_ids", "outer_train_ids", "held_ids")]
        assert not fit & val and fit | val == training
        assert not training & held and training | held == train_ids
        assert not (training | held) & test_ids and not held & seen
        seen |= held
    assert seen == train_ids
    stream = pd.read_csv(ROOT / "data/results/acceptance/A05/segment_verdicts.csv")
    engineering = pd.read_csv(folder / "predictions.csv")
    engineering = engineering[engineering.method == "engineering_gate"]
    engineering["segment"] = engineering.segment.astype(str)
    stream["segment"] = stream.segment.astype(str)
    joined = engineering.merge(stream, on=["segment", "channel"], suffixes=("_experiment", "_stream"), validate="one_to_one")
    assert len(joined) == 529
    assert (joined.final_pred_experiment.astype(int) == joined.final_pred_stream.astype(int)).all()
    compared = []
    if repeat is not None:
        for name in manifest["artifacts"]:
            assert (folder / name).read_bytes() == (repeat / name).read_bytes(), name
            compared.append(name)
    return {"status": "passed", "benchmark_segments": 529, "nested_unique_held_segments": 1594,
            "benchmark_methods": 8, "nested_methods": 7, "split_isolation": True,
            "independent_metric_tolerance": 1e-12, "all_saved_metrics_recomputed": True,
            "engineering_matches_real_closed_stream": True, "repeat_equal_artifacts": compared}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=ROOT / "data/results/graduation-confirmation-v1")
    parser.add_argument("--repeat", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.folder, args.repeat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
