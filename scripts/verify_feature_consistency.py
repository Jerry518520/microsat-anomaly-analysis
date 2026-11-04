"""Independent full A01 verification; never trains or modifies historical data.

Run: python scripts/verify_feature_consistency.py
Outputs summary, every differing cell, and compressed point/smoothing traces.
The historical extraction below is a frozen diagnostic reference, not a second
production extraction path. It must not be imported by application code.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.extract_segments_18d import generate_dataset
from src.features.segment_spec import (
    CANONICAL_FILENAME, COUNT_COLS, FEATURE_COLS, FEATURE_TOLERANCES,
    feature_manifest, smooth_values,
)
from src.streaming.segment_features import extract_segment_features, feature_vector


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_features(values, seconds, sampling):
    """Frozen pre-A01 code, for reproducing the audit and measuring migration."""
    def peaks(a):
        return len(find_peaks(a, prominence=.1 * np.ptp(a))[0]) if len(a) else 0
    result = {
        "duration": float(int(round(seconds[-1] - seconds[0])) % 86400),
        "len": float(len(values)), "mean": float(np.mean(values)),
        "var": float(np.var(values)), "std": float(np.std(values)),
        "kurtosis": float(kurtosis(values)), "skew": float(skew(values)),
        "n_peaks": float(peaks(values)),
        "smooth10_n_peaks": float(peaks(np.convolve(values, np.ones(10) / 10, "same"))),
        "smooth20_n_peaks": float(peaks(np.convolve(values, np.ones(20) / 20, "same"))),
        "diff_peaks": float(peaks(np.diff(values))),
        "diff2_peaks": float(peaks(np.diff(values, n=2))),
        "diff_var": float(np.var(np.diff(values))),
        "diff2_var": float(np.var(np.diff(values, n=2))),
        "gaps_squared": float(sum((int(round(d)) % 86400) ** 2 for d in np.diff(seconds))),
    }
    result["len_weighted"] = float(sampling) * result["len"]
    result["var_div_duration"] = result["var"] / result["duration"] if result["duration"] else 0.
    result["var_div_len"] = result["var"] / result["len"]
    return result


def compare(left, right, name, records):
    if not left.index.equals(right.index):
        raise AssertionError(f"{name}: segment identities differ")
    mask = np.column_stack([
        ~np.isclose(left[col], right[col], equal_nan=False, **FEATURE_TOLERANCES[col])
        for col in FEATURE_COLS
    ])
    for i, j in np.argwhere(mask):
        sid, col = left.index[i], FEATURE_COLS[j]
        records.append({"comparison": name, "segment": int(sid),
                        "channel": str(left.loc[sid, "channel"]),
                        "train": int(left.loc[sid, "train"]), "feature": col,
                        "left": float(left.loc[sid, col]), "right": float(right.loc[sid, col])})
    return {
        "segments": len(left), "different_segments": int(mask.any(axis=1).sum()),
        "different_cells": int(mask.sum()),
        "per_feature": {col: {"different_cells": int(mask[:, j].sum()),
                              "max_abs_difference": float(np.max(np.abs(left[col] - right[col])))}
                        for j, col in enumerate(FEATURE_COLS)},
        "exact_count_match": bool(np.array_equal(left[COUNT_COLS], right[COUNT_COLS])),
    }


def trace_signal(signal):
    indices, properties = find_peaks(signal, prominence=.1 * np.ptp(signal))
    return {"values": signal.tolist(), "peak_indices": indices.tolist(),
            "prominence_threshold": float(.1 * np.ptp(signal)),
            "prominences": properties["prominences"].tolist()}


def predict(judges, frame):
    """Existing model only; judge has no training side effects."""
    return np.array([int(judges[row.channel].judge(
        {name: float(getattr(row, name)) for name in FEATURE_COLS}).anomaly)
        for row in frame.itertuples()], dtype=int)


def run(out, model_path):
    out.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / "data/raw/segments.csv"
    official_path = next((ROOT / "data/raw").glob("dataset-*.csv"))
    processed_path = ROOT / "data/processed/segments_18d.csv"
    protected = {p: sha(p) for p in (raw_path, official_path, processed_path, model_path)}
    raw = pd.read_csv(raw_path)
    official = pd.read_csv(official_path).set_index("segment").sort_index()
    processed = pd.read_csv(processed_path).set_index("segment").sort_index()
    # Register tolerances before computing or comparing any new dataset.
    (out / "registered_manifest.json").write_text(
        json.dumps(feature_manifest(), indent=2, ensure_ascii=False), encoding="utf-8")
    offline = generate_dataset(raw).set_index("segment").sort_index()
    streams, historical = [], []
    groups = {}
    for sid, frame in raw.groupby("segment", sort=True):
        # Independent timestamp conversion; do not use the canonical time helper.
        seconds = (pd.to_datetime(frame.timestamp, utc=True, format="ISO8601") -
                   pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
        values = frame.value.to_numpy()
        meta = {c: frame[c].iloc[0] for c in ("segment", "channel", "anomaly", "train", "sampling")}
        streams.append({**meta, **extract_segment_features(values, seconds, meta["sampling"])})
        historical.append({**meta, **historical_features(values, seconds, meta["sampling"])})
        groups[int(sid)] = (frame, seconds)
    stream = pd.DataFrame(streams).set_index("segment").sort_index()
    old = pd.DataFrame(historical).set_index("segment").sort_index()
    records, comparisons = [], {}
    for name, left, right in (
        ("official_vs_processed", official, processed),
        ("historical_stream_vs_official", old, official),
        ("historical_stream_vs_processed", old, processed),
        ("canonical_offline_vs_stream", offline, stream),
        ("canonical_vs_historical_stream", offline, old),
        ("canonical_vs_official", offline, official),
        ("canonical_vs_processed", offline, processed),
    ):
        comparisons[name] = compare(left, right, name, records)
    pd.DataFrame(records).to_csv(out / "all_differences.csv", index=False)
    canonical_path = ROOT / "data/processed" / CANONICAL_FILENAME
    offline.reset_index().to_csv(canonical_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    # Explicit round-trip parsing makes this saved file retain the model vector.
    disk = pd.read_csv(canonical_path, float_precision="round_trip").set_index("segment").sort_index()
    vectors_equal = np.array_equal(offline[FEATURE_COLS], stream[FEATURE_COLS])
    disk_vectors_equal = np.array_equal(offline[FEATURE_COLS], disk[FEATURE_COLS])
    cache = joblib.load(model_path)
    judges = cache["judges"]
    # Predict both independently generated vectors with exactly the same model.
    po, ps = predict(judges, offline), predict(judges, stream)
    ph, pp = predict(judges, old), predict(judges, processed)
    prediction_records = pd.DataFrame({
        "segment": offline.index, "channel": offline.channel.to_numpy(),
        "train": offline.train.to_numpy(), "historical_stream": ph,
        "processed": pp, "canonical_offline": po, "canonical_stream": ps,
    })
    prediction_records.to_csv(out / "same_model_predictions.csv", index=False)
    affected = sorted({r["segment"] for r in records if r["feature"].startswith("smooth")})
    traces, layout_changes, parser_changes = [], [], []
    roundtrip = pd.read_csv(raw_path, float_precision="round_trip")
    assert np.array_equal(raw.value, roundtrip.value)  # audit's already excluded cause
    for sid in affected:
        frame, seconds = groups[sid]
        values = frame.value.to_numpy()
        subset = raw.loc[raw.segment == sid, "value"].to_numpy()
        strided = np.column_stack([values, values])[:, 0]
        # Test layout/content hypotheses independently of production normalization.
        smooth = np.convolve(values, np.ones(20) / 20, "same")
        if not np.array_equal(smooth, np.convolve(strided, np.ones(20) / 20, "same")):
            layout_changes.append(sid)
        if not np.array_equal(smooth, np.convolve(subset, np.ones(20) / 20, "same")):
            parser_changes.append(sid)
        traces.append({
            "segment": sid, "channel": str(frame.channel.iloc[0]),
            "raw_csv_row_indices_zero_based": frame.index.tolist(),
            "timestamp_original": frame.timestamp.tolist(), "unix_seconds": seconds.tolist(),
            "raw_values": values.tolist(), "dtype": str(values.dtype),
            "stride_bytes": list(values.strides), "strided_stride_bytes": list(strided.strides),
            "group_offset_mod64": values.ctypes.data % 64, "subset_offset_mod64": subset.ctypes.data % 64,
            "point_order_nondecreasing": bool(np.all(np.diff(seconds) >= 0)),
            "dtype_experiment": {
                "float64_old_smooth20_count": len(find_peaks(smooth, prominence=.1 * np.ptp(smooth))[0]),
                "float32_old_smooth20_count": len(find_peaks(
                    np.convolve(values.astype(np.float32), np.ones(20, dtype=np.float32) / 20, "same"),
                    prominence=.1 * np.ptp(np.convolve(values.astype(np.float32), np.ones(20, dtype=np.float32) / 20, "same")))[0]),
                "float32_input_cells_changed": int(np.count_nonzero(values != values.astype(np.float32).astype(np.float64))),
                "interpretation": "float32 changes input values and arithmetic; sensitivity evidence only, not proof of historical dtype",
            },
            "stored_official": {f: float(official.loc[sid, f]) for f in ("smooth10_n_peaks", "smooth20_n_peaks")},
            "stored_processed": {f: float(processed.loc[sid, f]) for f in ("smooth10_n_peaks", "smooth20_n_peaks")},
            "historical_peak_indices": "not recoverable from stored counts; generation provenance unavailable",
            "signals": {str(w): {
                "old_numpy": trace_signal(np.convolve(values, np.ones(w) / w, "same")),
                "canonical_fsum": trace_signal(smooth_values(values, w)),
                "max_abs_smoothing_change": float(np.max(abs(np.convolve(values, np.ones(w) / w, "same") - smooth_values(values, w)))),
            } for w in (10, 20)},
        })
    # Deterministic gzip; keep full point evidence small enough to review/store.
    (out / "point_peak_traces.json.gz").write_bytes(gzip.compress(
        json.dumps(traces, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
    historical_test_changes = prediction_records[(prediction_records.train == 0) & (ph != pp)]
    summary = {
        "raw_points": len(raw), "segments": len(offline),
        "identity": feature_manifest(), "input_sha256": {str(p.relative_to(ROOT)): s for p, s in protected.items()},
        "canonical_dataset": str(canonical_path.relative_to(ROOT)), "canonical_dataset_sha256": sha(canonical_path),
        "tolerances_registered_before_run": FEATURE_TOLERANCES, "comparisons": comparisons,
        "all_18_model_vector_values_exact_equal": bool(vectors_equal),
        "saved_csv_roundtrip_model_vector_exact_equal": bool(disk_vectors_equal),
        "same_model_prediction_differences": int(np.count_nonzero(po != ps)),
        "historical_test_prediction_changes_vs_processed": historical_test_changes.to_dict("records"),
        "canonical_prediction_changes_vs_historical": prediction_records[po != ph].to_dict("records"),
        "hypotheses": {"default_vs_roundtrip_value_changes": 0,
                       "layout_changed_convolution_segments": layout_changes,
                       "group_vs_subset_changed_convolution_segments": parser_changes},
        "traced_segments": affected,
        "historical_provenance": "unknown: no verified environment/input order/smoothed vectors/peak indices at historical generation time; do not claim official reconstruction",
        "protected_files_unchanged": all(sha(p) == s for p, s in protected.items()),
    }
    canonical_path.with_suffix(".manifest.json").write_text(json.dumps({
        **feature_manifest(), "dataset_sha256": sha(canonical_path), "raw_sha256": sha(raw_path), "rows": len(offline)
    }, indent=2, allow_nan=False), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"segments": len(offline), "comparisons": {k: {c: v[c] for c in ("different_segments", "different_cells")} for k, v in comparisons.items()},
                      "exact_vectors": bool(vectors_equal), "exact_disk_vectors": bool(disk_vectors_equal),
                      "same_model_prediction_differences": summary["same_model_prediction_differences"],
                      "protected_files_unchanged": summary["protected_files_unchanged"]}, indent=2))
    assert comparisons["canonical_offline_vs_stream"]["different_cells"] == 0
    assert vectors_equal and disk_vectors_equal and np.array_equal(po, ps)
    assert summary["protected_files_unchanged"]
    assert len(offline) == len(official) == 2123


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data/results/acceptance/A01")
    parser.add_argument("--model", type=Path, default=ROOT / "data/models/stage2_detector.joblib")
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run(args.out, args.model)
