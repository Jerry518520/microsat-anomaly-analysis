"""Versioned 18D contract. Batch and stream must call extract_features.

This is a new reproducible feature version, not an exact reconstruction of the
official historical CSV. See docs/A01_FEATURE_SPEC.md before changing semantics.
"""
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

FEATURE_VERSION = "18d-canonical-v1"
CANONICAL_FILENAME = f"segments_18d_{FEATURE_VERSION}.csv"
FEATURE_COLS = [
    "duration", "len", "mean", "var", "std", "kurtosis", "skew",
    "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks", "diff_peaks",
    "diff2_peaks", "diff_var", "diff2_var", "gaps_squared", "len_weighted",
    "var_div_duration", "var_div_len",
]
COUNT_COLS = ["len", "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks",
              "diff_peaks", "diff2_peaks"]
# Normative acceptance is EXACT for all 18 values. Auxiliary relative checks
# have no absolute floor: magnetometer variances can be far below 1e-12 V^2.
FEATURE_TOLERANCES = {
    name: {"atol": 0., "rtol": 1e-10}
    for name in FEATURE_COLS
}
for _name in COUNT_COLS + ["duration", "gaps_squared", "len_weighted"]:
    FEATURE_TOLERANCES[_name] = {"atol": 0., "rtol": 0.}

_SPEC = {
    "version": FEATURE_VERSION, "columns": FEATURE_COLS,
    "input": "1D contiguous float64; same lengths; N>=3; finite; nondecreasing unix seconds; positive integer sampling seconds; preserve duplicates and order",
    "time": "floor(delta_seconds) modulo 86400; reject negative deltas; sum squared adjacent components",
    "moments": "numpy mean/var/std ddof=0; scipy kurtosis Fisher,bias=True and skew bias=True; exact constant kurtosis/skew=0",
    "peaks": "scipy.find_peaks prominence=0.1*(max-min); exact equal plateau midpoint rounded down; exclude endpoints; no quantization",
    "smoothing": "math.fsum of each zero-padded window divided by 10 or 20; full convolution crop start=(min(N,W)-1)//2; length=max(N,W)",
    "differences": "numpy.diff n=1/2; population variance and same peak rule",
    "derived": "sampling*len; var/duration (0 when duration=0); var/len",
    "output": "18 float64-compatible finite values or explicit ValueError; no imputation downstream",
    "acceptance": "all 18 values exactly equal; tolerances are auxiliary diagnostics only",
    "tolerances": FEATURE_TOLERANCES,
}
FEATURE_SPEC_SHA256 = hashlib.sha256(
    json.dumps(_SPEC, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def feature_manifest():
    """Identity used by dataset writers and model-cache validation."""
    root = Path(__file__).resolve().parents[2]
    sources = ["src/features/segment_spec.py", "src/features/extract_segments_18d.py",
               "src/streaming/segment_features.py"]
    return {
        "feature_version": FEATURE_VERSION, "feature_cols": list(FEATURE_COLS),
        "feature_spec_sha256": FEATURE_SPEC_SHA256, "spec": _SPEC,
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
        "dependencies": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scipy": scipy.__version__},
    }


def timestamp_seconds(timestamps):
    """Parse batch datetimes to the same float64 unix seconds used by streams."""
    parsed = pd.to_datetime(timestamps, utc=True, format="ISO8601")
    return np.asarray((parsed - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds(), dtype=np.float64)


def seconds_component(delta):
    """Historical pandas .seconds compatibility for nonnegative durations."""
    return math.floor(float(delta)) % 86400


def peak_indices(array):
    array = np.asarray(array, dtype=np.float64)
    if array.size < 3:
        return np.empty(0, dtype=np.intp)
    return find_peaks(array, prominence=0.1 * (np.max(array) - np.min(array)))[0]


def number_of_peaks_finding(array):
    return len(peak_indices(array))


def smooth_values(array, window):
    """Fixed arithmetic; NumPy 'same' alignment, including N < even window."""
    array = np.asarray(array, dtype=np.float64)
    n = len(array)
    start = (min(n, window) - 1) // 2
    return np.array([
        math.fsum(array[max(0, i - window + 1):min(n, i + 1)]) / window
        for i in range(start, start + max(n, window))
    ], dtype=np.float64)


def smooth10_n_peaks(array):
    return number_of_peaks_finding(smooth_values(array, 10))


def smooth20_n_peaks(array):
    return number_of_peaks_finding(smooth_values(array, 20))


def diff_peaks(array):
    return number_of_peaks_finding(np.diff(array))


def diff2_peaks(array):
    return number_of_peaks_finding(np.diff(array, n=2))


def diff_var(array):
    return np.var(np.diff(array))


def diff2_var(array):
    return np.var(np.diff(array, n=2))


def _kurtosis(array):
    return 0. if np.all(array == array[0]) else kurtosis(array, fisher=True, bias=True)


def _skew(array):
    return 0. if np.all(array == array[0]) else skew(array, bias=True)


TRANSFORMATIONS = {
    "len": len, "mean": np.mean, "var": np.var, "std": np.std,
    "kurtosis": _kurtosis, "skew": _skew,
    "n_peaks": number_of_peaks_finding, "smooth10_n_peaks": smooth10_n_peaks,
    "smooth20_n_peaks": smooth20_n_peaks, "diff_peaks": diff_peaks,
    "diff2_peaks": diff2_peaks, "diff_var": diff_var, "diff2_var": diff2_var,
}


def extract_features(values, timestamps, sampling=1):
    """The only normative extraction entry; adapters do no feature arithmetic."""
    values = np.ascontiguousarray(values, dtype=np.float64)
    timestamps = np.ascontiguousarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or timestamps.ndim != 1 or len(values) != len(timestamps):
        raise ValueError("values and timestamps must be equally sized 1D arrays")
    if len(values) < 3:
        raise ValueError("at least 3 points are required; shorter segments are not applicable")
    if not np.isfinite(values).all() or not np.isfinite(timestamps).all():
        raise ValueError("values and timestamps must be finite and complete")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps must be nondecreasing; input is not reordered")
    if not np.isscalar(sampling) or not np.isfinite(sampling) or sampling <= 0 or int(sampling) != sampling:
        raise ValueError("sampling must be a positive integer in seconds")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        feats = {"duration": float(seconds_component(timestamps[-1] - timestamps[0]))}
        feats.update({name: float(fn(values)) for name, fn in TRANSFORMATIONS.items()})
        feats["gaps_squared"] = float(sum(seconds_component(d) ** 2 for d in np.diff(timestamps)))
        feats["len_weighted"] = float(sampling) * feats["len"]
        feats["var_div_duration"] = feats["var"] / feats["duration"] if feats["duration"] else 0.
        feats["var_div_len"] = feats["var"] / feats["len"]
    if not np.isfinite(list(feats.values())).all():
        raise ValueError("segment produces undefined or non-finite features")
    return feats
