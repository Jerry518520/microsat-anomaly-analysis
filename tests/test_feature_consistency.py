"""A01: shared feature contract, including cases not present in the raw data."""
import json

import numpy as np
import pandas as pd
import pytest

from src.features.extract_segments_18d import generate_dataset
from src.streaming.segment_features import FEATURE_COLS, extract_segment_features, feature_vector


def points(values, timestamps=None, sampling=1):
    n = len(values)
    seconds = np.arange(n, dtype=float) if timestamps is None else np.asarray(timestamps)
    return pd.DataFrame({
        "segment": [1] * n, "anomaly": [0] * n, "train": [1] * n,
        "channel": ["CADC0872"] * n, "sampling": [sampling] * n,
        "value": values, "timestamp": pd.to_datetime(seconds, unit="s", utc=True),
    })


def test_float32_input_is_normalized_before_all_formulas():
    values = np.array([0, .1, .5, .9, .2, 0], dtype=np.float32)
    actual = feature_vector(extract_segment_features(values, np.arange(len(values))))
    expected = feature_vector(extract_segment_features(values.astype(np.float64), np.arange(len(values))))
    np.testing.assert_array_equal(actual, expected)


def test_zero_duration_and_duplicate_timestamps_match_offline():
    frame = points([0., 2., 0., 1.], [10, 10, 10, 10])
    offline = generate_dataset(frame).iloc[0]
    stream = extract_segment_features(frame.value.to_numpy(), np.array([10.] * 4))
    np.testing.assert_array_equal(offline[FEATURE_COLS].to_numpy(dtype=float), feature_vector(stream))
    assert stream["duration"] == stream["gaps_squared"] == stream["var_div_duration"] == 0


def test_seconds_component_floors_fractional_seconds_and_keeps_cross_day_compatibility():
    result = extract_segment_features(np.array([0., 1., 0.]), np.array([0., 86400.9, 86401.8]))
    assert result["duration"] == 1
    assert result["gaps_squared"] == 0


@pytest.mark.parametrize("values,timestamps,sampling", [
    ([], [], 1), ([1], [0], 1), ([1, 2], [0, 1], 1),
    ([1, np.nan, 2], [0, 1, 2], 1), ([1, np.inf, 2], [0, 1, 2], 1),
    ([1, 2, 3], [0, np.nan, 2], 1), ([1, 2, 3], [0, 2, 1], 1),
    ([1, 2, 3], [0, 1], 1), ([[1, 2, 3]], [0, 1, 2], 1),
    ([1, 2, 3], [0, 1, 2], -1), ([1, 2, 3], [0, 1, 2], 0),
    ([1, 2, 3], [0, 1, 2], np.nan),
    ([1, 2, 3], [0, 1, 2], True),
    ([1, 2, 3], [0, 1, 2], np.bool_(True)),
    ([1, 2, 3], [0, 1, 2], "1"),
])
def test_invalid_or_unsupported_segments_are_rejected(values, timestamps, sampling):
    with pytest.raises(ValueError):
        extract_segment_features(np.asarray(values), np.asarray(timestamps), sampling)


@pytest.mark.parametrize("values,timestamps", [
    ([1e308] * 3, [0, 1, 2]),
    ([0, 1, 0], [-1e308, 0, 1e308]),
])
def test_finite_inputs_that_overflow_have_explicit_value_error_and_cause(values, timestamps):
    with pytest.raises(ValueError, match="overflow") as error:
        extract_segment_features(np.asarray(values), np.asarray(timestamps))
    assert isinstance(error.value.__cause__, (OverflowError, FloatingPointError))


def test_constant_segment_has_explicit_finite_moment_sentinels():
    result = extract_segment_features(np.ones(8), np.arange(8))
    assert result["kurtosis"] == result["skew"] == 0
    assert np.isfinite(feature_vector(result)).all()


@pytest.mark.parametrize("count", [3, 8])
def test_exact_decimal_constant_has_exact_mean_and_zero_variance(count):
    result = extract_segment_features(np.full(count, .1), np.arange(count))
    assert result["mean"] == .1
    assert result["var"] == 0.
    assert result["std"] == 0.


def test_near_constant_is_not_zeroed_by_a_threshold():
    result = extract_segment_features(np.array([.1, .1 + 1e-10, .1]), np.arange(3))
    assert 0 < result["var"] < 1e-20
    assert result["std"] > 0
    assert result["mean"] != .1


def test_raw_and_smoothing_peak_plateaus_keep_scipy_semantics():
    from src.features.segment_spec import peak_indices, smooth_values
    assert peak_indices(np.array([0., 1., 1., 1., 0.])).tolist() == [2]
    assert peak_indices(np.array([0., 1., np.nextafter(1., 0.), 1., 0.])).tolist() == [1, 3]
    for n in (3, 8, 10, 19, 20, 21, 24):
        values = np.arange(n, dtype=float)
        for window in (10, 20):
            np.testing.assert_allclose(smooth_values(values, window),
                                       np.convolve(values, np.ones(window) / window, "same"),
                                       rtol=2e-15, atol=1e-15)


def test_fixed_sum_does_not_split_the_675_smoothing_plateau():
    values = np.array([0, 0, .0192109, .0439221, .0686601, .115515, .187709,
                       .25807, .315298, .353065, .355991, .321077, .263751,
                       .193299, .132108, .0494165, .0219557, .00548852, 0, 0, 0, 0])
    result = extract_segment_features(values, np.arange(len(values)))
    assert result["smooth20_n_peaks"] == 1


def test_offline_rejects_inconsistent_metadata():
    frame = points([0., 1., 0.])
    frame.loc[1, "sampling"] = 2
    with pytest.raises(ValueError, match="sampling"):
        generate_dataset(frame)


def test_offline_rejects_empty_point_table():
    with pytest.raises(ValueError, match="empty"):
        generate_dataset(points([]))


def test_even_window_crop_has_independent_hand_calculated_values():
    from src.features.segment_spec import smooth_values
    np.testing.assert_array_equal(smooth_values(np.array([1., 2., 3.]), 10),
                                  np.array([3.] + [6.] * 8 + [5.]) / 10)
    np.testing.assert_array_equal(smooth_values(np.array([1., 2., 3.]), 20),
                                  np.array([3.] + [6.] * 18 + [5.]) / 20)


def test_manifest_and_column_order_are_stable():
    from src.features.segment_spec import FEATURE_VERSION, FEATURE_SPEC_SHA256, feature_manifest
    manifest = feature_manifest()
    assert FEATURE_VERSION == "18d-canonical-v1"
    assert list(extract_segment_features(np.array([0., 1., 0.]), np.arange(3))) == FEATURE_COLS
    assert len(FEATURE_COLS) == 18
    assert manifest["feature_spec_sha256"] == FEATURE_SPEC_SHA256
    assert len(FEATURE_SPEC_SHA256) == 64
    assert manifest["feature_cols"] == FEATURE_COLS
    assert set(manifest["source_sha256"]) == {
        "src/features/segment_spec.py", "src/features/extract_segments_18d.py",
        "src/streaming/segment_features.py",
    }
    json.dumps(manifest, allow_nan=False)


def test_canonical_acceptance_requires_exact_values_without_absolute_floor():
    from src.features.segment_spec import FEATURE_TOLERANCES, feature_manifest
    assert all(tolerance["atol"] == 0 for tolerance in FEATURE_TOLERANCES.values())
    assert feature_manifest()["spec"]["acceptance"] == "all 18 values exactly equal; tolerances are auxiliary diagnostics only"
    tiny = extract_segment_features(np.array([0., 1e-12, 0.]), np.arange(3))["var"]
    assert 0 < tiny < 1e-12
    assert not np.isclose(tiny, 0., **FEATURE_TOLERANCES["var"])
