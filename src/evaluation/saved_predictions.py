"""Independent evaluation of saved predictions. This module never fits models."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics(truth, predicted, score=None):
    truth, predicted = np.asarray(truth, dtype=int), np.asarray(predicted, dtype=int)
    tn = int(((truth == 0) & (predicted == 0)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    tp = int(((truth == 1) & (predicted == 1)).sum())
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** .5
    result = {"n": len(truth), "positives": int(truth.sum()),
              "precision": tp / (tp + fp) if tp + fp else 0.,
              "recall": tp / (tp + fn) if tp + fn else 0.,
              "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
              "accuracy": (tp + tn) / len(truth),
              "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.,
              "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
              "roc_auc": None, "average_precision": None}
    if score is not None and np.isfinite(score).all():
        if len(np.unique(truth)) == 2:
            result["roc_auc"] = float(roc_auc_score(truth, score))
        if truth.sum():
            result["average_precision"] = float(average_precision_score(truth, score))
    return result


def validate_predictions(frame):
    required = {"method", "segment", "channel", "truth", "final_pred", "score"}
    if not required.issubset(frame.columns) or frame[list(required - {"score"})].isna().any().any():
        raise ValueError("prediction fields are missing")
    if frame.duplicated(["method", "segment"]).any():
        raise ValueError("duplicate segment within a method")
    if not frame.truth.isin([0, 1]).all() or not frame.final_pred.isin([0, 1]).all():
        raise ValueError("labels must be binary")
    reference = None
    for _, rows in frame.groupby("method", sort=True):
        identity = rows.sort_values("segment")[["segment", "channel", "truth"]].reset_index(drop=True)
        if reference is not None and not identity.equals(reference):
            raise ValueError("methods have different samples or truth")
        reference = identity


def _bootstrap_f1(frame, repetitions=2000):
    counts = np.array([list(metrics(rows.truth, rows.final_pred)["confusion"].values())
                       for _, rows in frame.groupby("channel", sort=True)], dtype=float)
    indices = np.random.default_rng(42).integers(0, len(counts), size=(repetitions, len(counts)))
    sums = counts[indices].sum(axis=1)
    denominator = 2 * sums[:, 3] + sums[:, 1] + sums[:, 2]
    return np.divide(2 * sums[:, 3], denominator, out=np.zeros(repetitions), where=denominator != 0)


def paired_interval(first, second, repetitions=2000):
    order = ["channel", "segment"]
    first, second = first.sort_values(order), second.sort_values(order)
    if not first[order + ["truth"]].reset_index(drop=True).equals(second[order + ["truth"]].reset_index(drop=True)):
        raise ValueError("paired comparison needs the same events")
    difference = _bootstrap_f1(first, repetitions) - _bootstrap_f1(second, repetitions)
    return np.quantile(difference, [.025, .975]).tolist()


def evaluate_file(path):
    frame = pd.read_csv(path, float_precision="round_trip")
    validate_predictions(frame)
    overall, per_channel = {}, {}
    for method, rows in frame.groupby("method", sort=True):
        score = rows.score.to_numpy(dtype=float)
        overall[method] = metrics(rows.truth, rows.final_pred, score)
        overall[method]["f1_channel_bootstrap_95"] = np.quantile(_bootstrap_f1(rows), [.025, .975]).tolist()
        per_channel[method] = {}
        for channel, subset in rows.groupby("channel", sort=True):
            per_channel[method][channel] = {**metrics(subset.truth, subset.final_pred, subset.score.to_numpy()),
                                            "small_sample": len(subset) < 30}
    paired = {}
    if "gate" in overall:
        for baseline in ("rule", "channel_if"):
            paired[f"gate_minus_{baseline}"] = {
                "difference": overall["gate"]["f1"] - overall[baseline]["f1"],
                "channel_bootstrap_95": paired_interval(frame[frame.method == "gate"], frame[frame.method == baseline])}
    return {"prediction_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "overall": overall, "per_channel": per_channel, "paired_f1": paired,
            "bootstrap": {"unit": "whole channel", "repetitions": 2000, "seed": 42}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_file(args.predictions)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
