"""A02 engineering contract. No network or LLM calls."""
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def api():
    return importlib.import_module("src.detection.frozen")


@pytest.mark.parametrize("op", ["rule", "if", "AND", "OR"])
@pytest.mark.parametrize("rule,forest", [(False, False), (False, True), (True, False), (True, True)])
def test_gate_truth_table(op, rule, forest):
    expected = {"rule": rule, "if": forest, "AND": rule and forest, "OR": rule or forest}[op]
    assert api().gate(op, rule, forest) is expected


@pytest.mark.parametrize("op,rule,forest", [("if", True, None), ("AND", False, None),
                                          ("OR", True, None), ("rule", None, False)])
def test_missing_required_component_is_unavailable(op, rule, forest):
    with pytest.raises(api().DetectorUnavailable):
        api().gate(op, rule, forest)


def test_rule_counts_each_feature_once_and_null_if_is_supported():
    m = api()
    th = {"mean": {"mean": 0., "std": 1., "q1": -1., "q3": 1., "iqr": 2.}}
    judge = m.EngineeringJudge("CADC0884", None, th, k=1, gate_op="rule")
    feats = dict.fromkeys(m.FEATURE_COLS, 0.)
    feats["mean"] = 100.
    v = judge.judge(feats)
    assert v.rule_hits == 1 and v.final_pred and v.if_pred is None
    assert v.final_pred == v.gate_pred == v.anomaly
    assert m.gate("rule", True, None) is True
    judge.thresholds = {}
    with pytest.raises(m.DetectorUnavailable):
        judge.judge(feats)


def points(segment="new-segment", channel="CADC0884", values=(0., 1., 0., 1.)):
    return pd.DataFrame({"channel": channel, "segment": segment,
                         "timestamp": pd.date_range("2026-01-01", periods=len(values), freq="5s"),
                         "value": values, "sampling": 5})


def test_new_points_preserve_identity_and_labels_are_optional():
    m = api()
    p = pd.concat([points(), points(77, "CADC0886", (4., 5., 6., 7.))], ignore_index=True)
    features = m.point_features(p)
    assert list(features.segment) == ["new-segment", 77]
    assert list(features["mean"]) == [0.5, 5.5]
    changed = p.assign(anomaly=1, train=1)
    pd.testing.assert_frame_equal(features[m.FEATURE_COLS], m.point_features(changed)[m.FEATURE_COLS])
    same_id = pd.concat([points("same"), points("same", "CADC0886")])
    assert len(m.point_features(same_id)) == 2


@pytest.mark.parametrize("segment", ["", None, True, 1.5])
def test_invalid_segment_identity_fails(segment):
    with pytest.raises(ValueError):
        api().point_features(points(segment))


def test_missing_bundle_never_trains(tmp_path, monkeypatch):
    import sklearn.ensemble
    monkeypatch.setattr(sklearn.ensemble.IsolationForest, "fit", lambda *a, **k: pytest.fail("request trained"))
    with pytest.raises(api().DetectorUnavailable, match="missing"):
        api().load_bundle(tmp_path)


def test_verified_runtime_and_stage2_alias_never_fit(frozen, monkeypatch):
    import sklearn.ensemble
    from src.streaming import stage2
    m = api()
    monkeypatch.setattr(m, "DEFAULT_BUNDLE", frozen)
    monkeypatch.setattr(sklearn.ensemble.IsolationForest, "fit", lambda *a, **k: pytest.fail("runtime fit"))
    monkeypatch.setattr(stage2, "train_detector", lambda: pytest.fail("legacy runtime training"))
    judges = stage2.load_or_train()
    assert len(judges) == 9
    assert m.predict_features(m.point_features(points()), judges).iloc[0].detector_version == m.VERSION


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    target = tmp_path_factory.mktemp("a02-package") / "bundle"
    api().freeze_bundle(target)
    return target


def test_frozen_batch_single_and_stream_match_and_display_cannot_change_label(frozen, monkeypatch):
    m = api()
    judges = m.load_bundle(frozen)
    p = pd.concat([points(), points(77, "CADC0886", (4., 5., 6., 7.))], ignore_index=True)
    features = m.point_features(p)
    batch = m.predict_features(features, judges)
    singles = pd.concat([m.predict_features(features.iloc[[i]], judges) for i in range(2)], ignore_index=True)
    pd.testing.assert_frame_equal(batch, singles)
    for i, row in features.iterrows():
        from src.streaming.segment_features import extract_segment_features
        frame = p[(p.channel == row.channel) & (p.segment == row.segment)]
        from src.features.segment_spec import timestamp_seconds
        feats = extract_segment_features(frame.value, timestamp_seconds(frame.timestamp.to_numpy()), row.sampling)
        verdict = judges[row.channel].judge(feats)
        assert verdict.final_pred == batch.iloc[i].final_pred
        assert verdict.rule_hits == batch.iloc[i].rule_hits
    monkeypatch.setattr(m.EngineeringJudge, "display_score", lambda *args: 0.)
    low = m.predict_features(features, judges)
    monkeypatch.setattr(m.EngineeringJudge, "display_score", lambda *args: 1.)
    high = m.predict_features(features, judges)
    assert list(low.final_pred) == list(high.final_pred) == list(batch.final_pred)


@pytest.mark.parametrize("field", ["feature", "sources", "dependencies", "inputs", "recipe_sha256", "model_sha256"])
def test_manifest_tamper_fails_before_joblib_load(frozen, tmp_path, monkeypatch, field):
    import shutil
    import joblib
    shutil.copytree(frozen, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "model_manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = "tampered"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(joblib, "load", lambda *a: pytest.fail("deserialized before validation"))
    with pytest.raises(api().DetectorUnavailable, match="mismatch"):
        api().load_bundle(tmp_path)


def test_corrupt_model_fails_before_deserialization(frozen, tmp_path, monkeypatch):
    import shutil
    import joblib
    shutil.copytree(frozen, tmp_path, dirs_exist_ok=True)
    (tmp_path / "detector.joblib").write_bytes(b"corrupt")
    monkeypatch.setattr(joblib, "load", lambda *a: pytest.fail("deserialized corrupt model"))
    with pytest.raises(api().DetectorUnavailable, match="model_sha256"):
        api().load_bundle(tmp_path)


def test_pipeline_predicts_given_points_only_with_optional_labels(frozen, monkeypatch):
    from src.integration.anomaly_rag_pipeline import AnomalyRAGPipeline
    m = api()
    monkeypatch.setattr(m, "DEFAULT_BUNDLE", frozen)
    p = AnomalyRAGPipeline()
    new = points("customer-unique", values=(1e9, -1e9, 2e9, -2e9))
    result = p.detect(new)
    assert len(p.last_seg_results) == 1 and p.last_seg_results.iloc[0].segment == "customer-unique"
    assert result and result[0].segment == "customer-unique"
    first = p.last_seg_results.copy()
    p.detect(new.assign(anomaly=0, train=0))
    pd.testing.assert_frame_equal(first.drop(columns="y_true", errors="ignore"),
                                  p.last_seg_results.drop(columns="y_true", errors="ignore"))
    p.detect(new.assign(anomaly=1, train=1))
    assert p.last_seg_results.iloc[0].final_pred == first.iloc[0].final_pred
    precise = points("precision", values=(.123456789012345, 1., 0., 1.))
    p.detect(precise)
    digest = p.last_input_sha256
    precise.loc[0, "value"] += 1e-14
    p.detect(precise)
    assert p.last_input_sha256 != digest


def test_small_fit_has_explicit_component_failure():
    m = api()
    recipe = m.load_recipe()
    channel = "CADC0874"  # IF gate requires IF; fewer than 10 fit rows cannot train it.
    frame = pd.DataFrame([{"channel": channel, **dict.fromkeys(m.FEATURE_COLS, 0.)}] * 4)
    judges = m.fit_judges(frame, recipe)
    with pytest.raises(m.DetectorUnavailable, match="IF"):
        judges[channel].judge(dict.fromkeys(m.FEATURE_COLS, 0.))


def test_explanation_keeps_low_display_formal_anomaly(frozen, monkeypatch):
    from types import SimpleNamespace
    from src.integration.anomaly_rag_pipeline import AnomalyRAGPipeline
    m = api()
    monkeypatch.setattr(m, "DEFAULT_BUNDLE", frozen)
    monkeypatch.setattr(m.EngineeringJudge, "display_score", lambda *args: 0.)
    p = AnomalyRAGPipeline()
    monkeypatch.setattr(p, "_init_rag", lambda: None)
    monkeypatch.setattr(p, "explain", lambda a: SimpleNamespace(answer=p._build_anomaly_description(a), sources=[], retrieval_time=0, generation_time=0))
    result = p.detect_and_explain(points("new-low-display", values=(1e9, -1e9, 1e9, -1e9)))
    assert len(result) == 1
    assert result[0].final_pred and result[0].anomaly_score == 0.
    assert result[0].gate_op == "rule"
    assert result[0].bundle_id == p.last_bundle_identity.manifest_sha256
    assert result[0].model_sha256 == p.last_bundle_identity.model_sha256


@pytest.mark.parametrize("file", ["detector_recipe.json", "model_manifest.json", "detector.joblib"])
def test_missing_package_part_and_legacy_cache_never_train(frozen, tmp_path, monkeypatch, file):
    import shutil
    import joblib
    import sklearn.ensemble
    shutil.copytree(frozen, tmp_path, dirs_exist_ok=True)
    (tmp_path / file).unlink()
    joblib.dump({"version": "stage2-ipynb-aligned-2", "judges": {}}, tmp_path / "stage2_detector.joblib")
    monkeypatch.setattr(sklearn.ensemble.IsolationForest, "fit", lambda *a, **k: pytest.fail("load trained"))
    with pytest.raises(api().DetectorUnavailable, match="missing"):
        api().load_bundle(tmp_path)


def test_rule_minimum_samples_and_zero_iqr_semantics():
    m = api()
    recipe = m.load_recipe()
    frame = pd.DataFrame([{"channel": "CADC0884", **dict.fromkeys(m.FEATURE_COLS, 0.)}] * 4)
    assert not m.fit_judges(frame, recipe)["CADC0884"].thresholds
    judge = m.fit_judges(pd.concat([frame, frame.iloc[[0]]]), recipe)["CADC0884"]
    assert judge.thresholds["mean"]["std"] == 1.
    assert judge.thresholds["mean"]["iqr"] == 0.
    feats = dict.fromkeys(m.FEATURE_COLS, 0.)
    feats["mean"] = 1e-8
    assert judge.rule_eval(feats) == 1


def test_explicit_build_rejects_broken_canonical_feature_provenance(tmp_path, monkeypatch):
    import shutil
    m = api()
    csv = tmp_path / "canonical.csv"
    shutil.copyfile(m.ROOT / m.TRAINING_CSV, csv)
    shutil.copyfile((m.ROOT / m.TRAINING_CSV).with_suffix(".manifest.json"), csv.with_suffix(".manifest.json"))
    csv.write_bytes(csv.read_bytes() + b"\n")
    monkeypatch.setattr(m, "TRAINING_CSV", str(csv))
    with pytest.raises(m.DetectorUnavailable, match="canonical dataset"):
        m.freeze_bundle(tmp_path / "output")


def test_default_result_save_uses_versioned_run_and_preserves_history(frozen, tmp_path, monkeypatch):
    from src.integration import anomaly_rag_pipeline as module
    monkeypatch.setattr(api(), "DEFAULT_BUNDLE", frozen)
    pipeline = module.AnomalyRAGPipeline()
    pipeline.detect(points())
    monkeypatch.setattr(module, "project_root", tmp_path)
    history = tmp_path / "data/results/anomaly_rag_results.json"
    history.parent.mkdir(parents=True)
    history.write_bytes(b"historical-output")
    output = pipeline.to_json([])
    assert history.read_bytes() == b"historical-output"
    assert api().VERSION in str(output)
    assert json.loads(Path(output).read_text(encoding="utf-8"))["detector_version"] == api().VERSION
    second = pipeline.to_json([])
    assert second != output
    with pytest.raises(FileExistsError):
        pipeline.to_json([], str(history))
    assert history.read_bytes() == b"historical-output"
