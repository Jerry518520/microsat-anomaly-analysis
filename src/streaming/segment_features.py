"""Thin streaming adapter for the versioned shared 18D feature contract."""
import numpy as np

from src.features.segment_spec import FEATURE_COLS, extract_features, seconds_component

# Compatibility import; the shared implementation follows pandas floor semantics.
_seconds_component = seconds_component


def extract_segment_features(values: np.ndarray, timestamps: np.ndarray,
                             sampling: int = 1) -> dict[str, float]:
    return extract_features(values, timestamps, sampling)


def feature_vector(feats: dict[str, float]) -> np.ndarray:
    """dict to a float64 vector in the fixed 18-column model input order."""
    return np.array([feats[c] for c in FEATURE_COLS], dtype=np.float64)
