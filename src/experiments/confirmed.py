"""Uniform graduation comparisons. Selection receives fit and validation only."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

from src.detection.frozen import EngineeringJudge
from src.features.segment_spec import FEATURE_COLS
from src.utils.stat_rules import compute_stat_thresholds

CONTAMINATIONS = [.05, .20, .35]
GATES = ["rule", "AND", "if", "OR"]
METHODS = ["global_if", "channel_if", "rule", "AND", "OR", "gate", "soft"]


def _fit_if(frame, contamination):
    model = IsolationForest(n_estimators=100, max_samples=min(128, len(frame)),
                            random_state=42, n_jobs=1, contamination=contamination)
    return model.fit(frame[FEATURE_COLS].to_numpy(dtype=float))


def _components(fit, samples, contamination):
    if len(fit) < 5 or not np.isfinite(fit[FEATURE_COLS].to_numpy(dtype=float)).all():
        raise ValueError("at least five finite fit rows are required")
    if not np.isfinite(samples[FEATURE_COLS].to_numpy(dtype=float)).all():
        raise ValueError("evaluation features must be finite")
    result = samples[["segment", "channel"]].reset_index(drop=True).copy()
    pooled = fit.assign(channel="__global__")
    thresholds = compute_stat_thresholds(fit, FEATURE_COLS)
    pooled_thresholds = compute_stat_thresholds(pooled, FEATURE_COLS)["__global__"]
    global_model = _fit_if(fit, contamination)
    global_scores = -global_model.decision_function(samples[FEATURE_COLS].to_numpy(dtype=float))
    result["global_if_score"] = global_scores
    result["global_if_pred"] = (global_scores > 0).astype(int)
    for channel in result.channel.unique():
        mask = result.channel == channel
        rows = samples.iloc[np.flatnonzero(mask)]
        training = fit[fit.channel == channel]
        fallback = len(training) < 5
        model = global_model if fallback else _fit_if(training, contamination)
        calibration = fit if fallback else training
        scores = -model.decision_function(rows[FEATURE_COLS].to_numpy(dtype=float))
        fit_scores = np.sort(-model.decision_function(calibration[FEATURE_COLS].to_numpy(dtype=float)))
        result.loc[mask, "if_score"] = scores
        result.loc[mask, "if_pred"] = (scores > 0).astype(int)
        result.loc[mask, "if_percentile"] = np.searchsorted(fit_scores, scores, side="right") / len(fit_scores)
        result.loc[mask, "if_fallback"] = fallback
        result.loc[mask, "if_fit_rows"] = len(calibration)
        judge = EngineeringJudge(channel, None,
            pooled_thresholds if fallback else thresholds[channel], k=1, gate_op="rule")
        result.loc[mask, "rule_hits"] = [judge.rule_eval(row) for row in rows[FEATURE_COLS].to_dict("records")]
        result.loc[mask, "rule_fallback"] = fallback
    for field in ("if_pred", "rule_hits", "if_fit_rows"):
        result[field] = result[field].astype(int)
    for field in ("if_fallback", "rule_fallback"):
        result[field] = result[field].astype(bool)
    return result


def _gate(op, rule, isolation):
    return {"rule": rule, "if": isolation, "AND": rule & isolation, "OR": rule | isolation}[op]


def select_recipe(fit, validation):
    """Choose only on the supplied validation rows. Ties retain candidate order."""
    truth = validation.anomaly.to_numpy(dtype=int)
    candidates = [_components(fit, validation.drop(columns="anomaly"), c) for c in CONTAMINATIONS]
    index = max(range(len(candidates)), key=lambda i: f1_score(truth, candidates[i].if_pred, zero_division=0))
    values = candidates[index]
    k = max(range(1, 5), key=lambda candidate: f1_score(truth, values.rule_hits >= candidate, zero_division=0))
    rule = (values.rule_hits.to_numpy() >= k).astype(int)
    isolation = values.if_pred.to_numpy(dtype=int)
    choices = {}
    for channel in values.channel.unique():
        mask = values.channel.to_numpy() == channel
        if mask.sum() < 5 or len(np.unique(truth[mask])) < 2:
            choices[channel] = "rule"
        else:
            choices[channel] = max(GATES, key=lambda op: f1_score(
                truth[mask], _gate(op, rule[mask], isolation[mask]), zero_division=0))
    soft_candidates = [(alpha, threshold) for alpha in (.25, .50, .75) for threshold in (.25, .50, .75)]
    alpha, threshold = max(soft_candidates, key=lambda pair: f1_score(truth,
        pair[0] * values.if_percentile + (1 - pair[0]) * values.rule_hits / len(FEATURE_COLS) >= pair[1],
        zero_division=0))
    return {"contamination": CONTAMINATIONS[index], "k": k, "gate_choice": choices,
            "soft_alpha": alpha, "soft_threshold": threshold,
            "validation_counts": {str(ch): int(n) for ch, n in validation.channel.value_counts().items()},
            "fit_counts": {str(ch): int(n) for ch, n in fit.channel.value_counts().items()}}


def predict_recipe(fit, samples, recipe):
    """Return predictions without reading or returning evaluation labels."""
    if "anomaly" in samples or "truth" in samples:
        raise ValueError("remove evaluation labels before prediction")
    values = _components(fit, samples, recipe["contamination"])
    rule = (values.rule_hits.to_numpy() >= recipe["k"]).astype(int)
    isolation = values.if_pred.to_numpy(dtype=int)
    operators = values.channel.map(recipe["gate_choice"]).fillna("rule").to_numpy()
    gate = np.array([_gate(op, r, i) for op, r, i in zip(operators, rule, isolation)])
    soft = recipe["soft_alpha"] * values.if_percentile + (1 - recipe["soft_alpha"]) * values.rule_hits / len(FEATURE_COLS)
    predictions = {"global_if": values.global_if_pred, "channel_if": isolation, "rule": rule,
                   "AND": rule & isolation, "OR": rule | isolation, "gate": gate,
                   "soft": (soft >= recipe["soft_threshold"]).astype(int)}
    scores = {"global_if": values.global_if_score, "channel_if": values.if_score,
              "rule": values.rule_hits, "soft": soft}
    records = []
    for method in METHODS:
        result = values.copy()
        result["method"] = method
        result["rule_pred"] = rule
        result["gate_op"] = operators if method == "gate" else method
        result["final_pred"] = np.asarray(predictions[method], dtype=int)
        result["score"] = scores.get(method, np.nan)
        records.append(result)
    return pd.concat(records, ignore_index=True)
