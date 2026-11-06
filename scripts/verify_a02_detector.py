"""Offline acceptance: official test and unseen segments, batch vs closed stream."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detection.frozen import (VERSION, load_bundle, point_features, predict_features,
                                  SPLIT_JSON, DEFAULT_BUNDLE, sha256)
from src.features.segment_spec import FEATURE_COLS, timestamp_seconds
from src.streaming.segment_features import extract_segment_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/results/acceptance/A02")
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    judges = load_bundle()
    # Exercise the real closed-segment engine, with persistence isolated in memory.
    from src.streaming.event_store import EventStore
    from src.streaming.engine import StreamEngine
    from src.streaming.segmenter import ClosedSegment
    engine = StreamEngine(store=EventStore(":memory:"))
    engine.judges, engine.models_ready = judges, True
    raw = pd.read_csv(ROOT / "data/raw/segments.csv", float_precision="round_trip")
    split = json.loads((ROOT / SPLIT_JSON).read_text(encoding="utf-8"))
    official = raw[raw.segment.isin(split["test_ids"])].copy()
    synthetic = []
    for ch in judges:
        synthetic.append(pd.DataFrame({"channel": ch, "segment": f"unseen-{ch}", "sampling": 5,
            "timestamp": pd.date_range("2026-11-01", periods=16, freq="5s").astype(str),
            "value": np.tile([1e9, -1e9], 8)}))
    supplied = pd.concat([official, *synthetic], ignore_index=True)
    features = point_features(supplied)
    batch = predict_features(features, judges)
    single = pd.concat([predict_features(features.iloc[[i]], judges) for i in range(len(features))], ignore_index=True)
    pd.testing.assert_frame_equal(batch, single, check_exact=True)
    stream = []
    for (ch, sid), frame in supplied.groupby(["channel", "segment"], sort=False):
        feats = extract_segment_features(frame.value.to_numpy(), timestamp_seconds(frame.timestamp.to_numpy()), frame.sampling.iloc[0])
        feature_row = features[(features.channel == ch) & (features.segment == sid)].iloc[0]
        assert np.array_equal(np.array([feats[c] for c in FEATURE_COLS]), feature_row[FEATURE_COLS].to_numpy(dtype=float))
        engine._judge_segment(ClosedSegment(ch, sid, True, frame.value.to_numpy(),
                              timestamp_seconds(frame.timestamp.to_numpy()), frame.sampling.iloc[0]))
        stream.append(engine.latest[ch]["judgement"])
    for name in stream[0]:
        assert batch[name].tolist() == [row[name] for row in stream], name
    batch.to_csv(output / "segment_verdicts.csv", index=False, float_format="%.17g")
    summary = {"detector_version": VERSION, "official_test_rows": len(split["test_ids"]),
               **next(iter(judges.values())).bundle_identity.as_fields(),
               "unseen_rows": len(synthetic), "total_rows": len(batch),
               "batch_single_exact": True, "batch_closed_stream_exact": True,
               "actual_stream_engine_exercised": True,
               "all_18_features_exact": True, "test_used_for_tuning": False,
               "postprocess": "identity", "manifest_sha256": sha256(DEFAULT_BUNDLE / "model_manifest.json"),
               "official_test_predicted_anomalies": int(batch.iloc[:len(split["test_ids"])].final_pred.sum()),
               "unseen_0884_can_be_anomalous": bool(batch[(batch.channel == "CADC0884") & batch.segment.eq("unseen-CADC0884")].final_pred.iloc[0])}
    assert summary["official_test_rows"] == 529
    assert summary["unseen_0884_can_be_anomalous"]
    (output / "consistency.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
