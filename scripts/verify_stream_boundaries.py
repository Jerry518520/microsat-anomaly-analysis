"""Verify all official segment boundaries and point order without fitting models."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.streaming.segmenter import OnlineSegmenter


def verify(output):
    raw = pd.read_csv(ROOT / "data/raw/segments.csv", float_precision="round_trip")
    raw["ts"] = (pd.to_datetime(raw.timestamp, utc=True, format="ISO8601") -
                 pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    segmenter = OnlineSegmenter(official=True, session_id="boundary-verification",
                                source_id="official-replay")
    closed = []
    for row in raw.sort_values("ts", kind="stable").itertuples(index=False):
        result = segmenter.feed(float(row.ts), row.channel, float(row.value),
                                int(row.segment), int(row.sampling))
        assert result.accepted, (row.segment, result.reason)
        if result.closed is not None:
            closed.append(result.closed)
    closed.extend(segmenter.flush(reason="replay_end"))
    groups = raw.groupby("segment", sort=False)
    assert len(closed) == groups.ngroups == 2123
    assert len({segment.source_segment_id for segment in closed}) == 2123
    for segment in closed:
        frame = groups.get_group(segment.source_segment_id)
        assert np.array_equal(segment.values, frame.value.to_numpy())
        assert np.array_equal(segment.timestamps, frame.ts.to_numpy())
        assert segment.official and segment.completeness == "complete"
    summary = {"raw_points": len(raw), "official_segments": len(closed),
               "exact_values_and_timestamps": True, "source_ids_unique": True,
               "replay_end_flush": True, "runtime_training_calls": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output)