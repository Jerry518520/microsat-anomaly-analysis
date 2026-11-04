"""Versioned 18D batch adapter. Historical CSV files are never overwritten."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, PROJECT_ROOT)
from src.features.segment_spec import (  # noqa: E402
    CANONICAL_FILENAME, FEATURE_COLS, FEATURE_TOLERANCES, TRANSFORMATIONS,
    extract_features, feature_manifest, timestamp_seconds,
    number_of_peaks_finding, smooth10_n_peaks, smooth20_n_peaks,
    diff_peaks, diff2_peaks, diff_var, diff2_var, seconds_component,
)

META_COLS = ["segment", "anomaly", "train", "channel", "sampling"]


def duration(df):
    timestamps = timestamp_seconds(df.timestamp.to_numpy())
    return seconds_component(timestamps[-1] - timestamps[0])


def gaps_squared(df):
    timestamps = timestamp_seconds(df.timestamp.to_numpy())
    return sum(seconds_component(d) ** 2 for d in np.diff(timestamps))


def generate_dataset(source_df):
    """Preserve segment point order; both adapters call extract_features."""
    if source_df.empty:
        raise ValueError("point table must not be empty")
    required = META_COLS + ["value", "timestamp"]
    if source_df[required].isna().any().any():
        raise ValueError("segment metadata, values and timestamps must be complete")
    rows = []
    for seg_id, frame in source_df.groupby("segment", sort=True):
        for col in META_COLS:
            if frame[col].nunique(dropna=False) != 1:
                raise ValueError(f"segment {seg_id}: inconsistent {col}")
        meta = {name: frame[name].iloc[0] for name in META_COLS}
        feats = extract_features(frame.value.to_numpy(),
                                 timestamp_seconds(frame.timestamp.to_numpy()),
                                 meta["sampling"])
        rows.append({**meta, **feats})
    return pd.DataFrame(rows, columns=META_COLS + FEATURE_COLS)


def validate_against_dataset(generated_df, dataset_csv_path):
    """Full historical comparison. Differences do not imply canonical failure."""
    official = pd.read_csv(dataset_csv_path).set_index("segment").sort_index()
    generated = generated_df.set_index("segment").sort_index()
    if not generated.index.equals(official.index):
        print("[Validation] segment identities differ")
        return False
    differences = {}
    for name in FEATURE_COLS:
        tol = FEATURE_TOLERANCES[name]
        different = ~np.isclose(generated[name], official[name], equal_nan=False, **tol)
        differences[name] = int(different.sum())
    print(f"[Validation] full historical comparison ({len(generated)} segments): {differences}")
    return not any(differences.values())


def main():
    raw_path = Path(PROJECT_ROOT) / "data/raw/segments.csv"
    official_path = next((Path(PROJECT_ROOT) / "data/raw").glob("dataset-*.csv"))
    out_path = Path(PROJECT_ROOT) / "data/processed" / CANONICAL_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(pd.read_csv(raw_path))
    df.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    manifest = feature_manifest()
    import hashlib
    manifest["dataset_sha256"] = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    manifest["rows"] = len(df)
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[Extract18D] canonical output: {out_path}")
    validate_against_dataset(df, official_path)
    return df


if __name__ == "__main__":
    main()
