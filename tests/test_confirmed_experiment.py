"""Small regressions for the saved-prediction experiment entry points."""
import numpy as np
import pandas as pd
import pytest


def test_independent_metrics_and_paired_interval():
    from src.evaluation.saved_predictions import metrics, paired_interval
    result = metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    assert result["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["precision"] == result["recall"] == result["f1"] == result["accuracy"] == .5
    assert result["mcc"] == 0.
    rows = pd.DataFrame({"segment": [1, 2, 3, 4], "channel": ["a", "a", "b", "b"], "truth": [0, 0, 1, 1],
                         "final_pred": [0, 1, 0, 1]})
    assert paired_interval(rows, rows, repetitions=20) == [0., 0.]


def test_selection_and_prediction_exclude_evaluation_labels():
    from src.experiments.confirmed import select_recipe, predict_recipe
    from src.features.segment_spec import FEATURE_COLS
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.normal(size=(60, 18)), columns=FEATURE_COLS)
    data["segment"] = np.arange(60)
    data["channel"] = np.repeat(["CADC0884", "CADC0892"], 30)
    data["anomaly"] = np.tile([0, 0, 0, 1, 1], 12)
    fit, val = data.iloc[:20].copy(), data.iloc[20:40].copy()
    recipe = select_recipe(fit, val)
    evaluation = data.iloc[40:].drop(columns="anomaly")
    one = predict_recipe(fit, evaluation, recipe)
    two = predict_recipe(fit, evaluation, recipe)
    pd.testing.assert_frame_equal(one, two, check_exact=True)
    assert "truth" not in one and "anomaly" not in one
    assert set(one.method) == {"global_if", "channel_if", "rule", "AND", "OR", "gate", "soft"}
    assert len(one) == 7 * len(evaluation)
    assert recipe["contamination"] in [.05, .2, .35]
    assert set(one.channel) == {"CADC0892"}
    assert one.if_fallback.all()  # This channel is absent from the fit fixture.


def test_saved_prediction_validation_rejects_duplicate_and_nonbinary():
    from src.evaluation.saved_predictions import validate_predictions
    frame = pd.DataFrame({"method": ["rule"], "segment": [1], "channel": ["a"],
                          "truth": [0], "final_pred": [1], "score": [.2]})
    validate_predictions(frame)
    with pytest.raises(ValueError):
        validate_predictions(pd.concat([frame, frame]))
    frame["truth"] = 2
    with pytest.raises(ValueError):
        validate_predictions(frame)
