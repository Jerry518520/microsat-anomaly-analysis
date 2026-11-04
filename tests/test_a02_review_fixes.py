"""Real regressions for the three A02 specification-review counterexamples."""
import importlib
import json
from pathlib import Path
import shutil

import joblib
import pytest


def api():
    return importlib.import_module("src.detection.frozen")


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    target = tmp_path_factory.mktemp("review-package") / "bundle"
    api().freeze_bundle(target)
    return target


def copy_package(source, target):
    shutil.copytree(source, target)
    return target


def manifest(path):
    return json.loads((path / "model_manifest.json").read_text(encoding="utf-8"))


def write_manifest(path, value):
    (path / "model_manifest.json").write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("field,bad", [("fit_rows", 0), ("fit_ids_sha256", "wrong-split"),
    ("fit_rows_per_channel", {}), ("effective_if_max_samples", {})])
@pytest.mark.parametrize("remove", [False, True])
def test_fit_identity_fields_are_required_before_deserialization(package, tmp_path, monkeypatch, field, bad, remove):
    target = copy_package(package, tmp_path / "bundle")
    value = manifest(target)
    if remove:
        del value[field]
    else:
        value[field] = bad
    write_manifest(target, value)
    monkeypatch.setattr(joblib, "load", lambda *a: pytest.fail("fit identity must be checked before load"))
    with pytest.raises(api().DetectorUnavailable, match=field):
        api().load_bundle(target)


@pytest.mark.parametrize("field,bad", [("channel", "CADC9999"), ("k", 99), ("gate_op", "rule"),
    ("detector_version", "wrong"), ("n_features_in_", 1), ("_n_samples", 1),
    ("max_samples_", 1), ("n_estimators", 1), ("max_samples", 1), ("random_state", 1),
    ("n_jobs", 99), ("contamination", .01), ("estimators_", []), ("model", None), ("thresholds", {}),
    ("feature_cols", ("mean",))])
def test_loaded_judges_match_recipe_and_actual_fit_population(package, tmp_path, field, bad):
    target = copy_package(package, tmp_path / "bundle")
    payload = joblib.load(target / "detector.joblib")
    judge = payload["judges"]["CADC0874"]  # IF gate
    setattr(judge if field in {"channel", "k", "gate_op", "detector_version", "model", "thresholds", "feature_cols"} else judge.model, field, bad)
    joblib.dump(payload, target / "detector.joblib")
    value = manifest(target)
    value["model_sha256"] = api().sha256(target / "detector.joblib")
    write_manifest(target, value)
    with pytest.raises(api().DetectorUnavailable, match="payload"):
        api().load_bundle(target)


@pytest.mark.parametrize("attribute,bad", [("offset_", "remove"), ("offset_", float("nan")),
    ("offset_", float("inf")), ("estimators_features_", "remove"), ("_decision_path_lengths", "remove")])
def test_load_rejects_missing_or_invalid_inference_state(package, tmp_path, attribute, bad):
    target = copy_package(package, tmp_path / "bundle")
    payload = joblib.load(target / "detector.joblib")
    model = payload["judges"]["CADC0874"].model
    if bad == "remove":
        delattr(model, attribute)
    else:
        setattr(model, attribute, bad)
    joblib.dump(payload, target / "detector.joblib")
    value = manifest(target)
    value["model_sha256"] = api().sha256(target / "detector.joblib")
    write_manifest(target, value)
    with pytest.raises(api().DetectorUnavailable, match="payload"):
        api().load_bundle(target)


def test_load_rejects_nonfinite_probe_score(package, monkeypatch):
    m = api()
    monkeypatch.setattr(m.IsolationForest, "decision_function", lambda self, values: m.np.full(len(values), m.np.nan))
    with pytest.raises(m.DetectorUnavailable, match="probe"):
        m.load_bundle(package)


def test_load_runs_one_finite_probe_per_channel_without_training(package, monkeypatch):
    m = api()
    calls = []
    original = m.EngineeringJudge.judge
    def track(judge, features):
        calls.append((judge.channel, len(features), all(m.np.isfinite(list(features.values())))))
        return original(judge, features)
    monkeypatch.setattr(m.EngineeringJudge, "judge", track)
    monkeypatch.setattr(m.IsolationForest, "fit", lambda *args, **kwargs: pytest.fail("load trained"))
    judges = m.load_bundle(package)
    assert len(calls) == len(judges) == 9
    assert {ch for ch, _, _ in calls} == set(judges)
    assert all(count == 18 and finite for _, count, finite in calls)


def test_default_build_refuses_existing_target_before_fit_and_preserves_sentinels(tmp_path, monkeypatch):
    target = tmp_path / "existing"
    target.mkdir()
    original = {name: ("sentinel-" + name).encode() for name in
                ["detector.joblib", "detector_recipe.json", "model_manifest.json"]}
    for name, data in original.items():
        (target / name).write_bytes(data)
    monkeypatch.setattr(api(), "fit_judges", lambda *a: pytest.fail("refusal must precede fit"))
    with pytest.raises(FileExistsError):
        api().freeze_bundle(target)
    assert {name: (target / name).read_bytes() for name in original} == original


def test_replace_archives_old_complete_package_and_publishes_new(package, tmp_path):
    target = copy_package(package, tmp_path / "bundle")
    old = {path.name: path.read_bytes() for path in target.iterdir()}
    api().freeze_bundle(target, replace=True)
    archives = list(tmp_path.glob("bundle.archive-*"))
    assert len(archives) == 1
    assert {path.name: path.read_bytes() for path in archives[0].iterdir()} == old
    assert len(api().load_bundle(archives[0])) == len(api().load_bundle(target)) == 9


def test_replace_build_failure_preserves_old_package(package, tmp_path, monkeypatch):
    target = copy_package(package, tmp_path / "bundle")
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    def fail(*args):
        raise RuntimeError("test build failure")
    monkeypatch.setattr(api(), "fit_judges", fail)
    with pytest.raises(RuntimeError, match="test build failure"):
        api().freeze_bundle(target, replace=True)
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before
    assert not list(tmp_path.glob("bundle.archive-*"))


def test_publish_failure_restores_original_directory(package, tmp_path, monkeypatch):
    m = api()
    target = copy_package(package, tmp_path / "bundle")
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    rename = m.os.rename
    def fail_stage_publish(source, destination):
        if Path(source).name.startswith(".bundle.staging-") and Path(destination) == target:
            raise OSError("test publish failure")
        return rename(source, destination)
    monkeypatch.setattr(m.os, "rename", fail_stage_publish)
    with pytest.raises(OSError, match="test publish failure"):
        m.freeze_bundle(target, replace=True)
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before
    assert not list(tmp_path.glob("bundle.archive-*"))


def test_build_rejects_unsafe_roots_before_fit(monkeypatch):
    monkeypatch.setattr(api(), "fit_judges", lambda *a: pytest.fail("unsafe target reached fit"))
    for target in [api().ROOT, api().ROOT / "data", api().ROOT / "data/models", Path(api().ROOT.anchor)]:
        with pytest.raises(ValueError, match="unsafe"):
            api().freeze_bundle(target, replace=True)


def alternate_serialization(source, target):
    copy_package(source, target)
    payload = joblib.load(target / "detector.joblib")
    joblib.dump(payload, target / "detector.joblib", compress=3)
    value = manifest(target)
    value["model_sha256"] = api().sha256(target / "detector.joblib")
    write_manifest(target, value)
    return target


def test_actual_loaded_identity_distinguishes_same_version_packages_and_survives_file_switch(package, tmp_path):
    m = api()
    first = copy_package(package, tmp_path / "first")
    second = alternate_serialization(package, tmp_path / "second")
    a, b = m.load_bundle(first), m.load_bundle(second)
    feats = dict.fromkeys(m.FEATURE_COLS, 0.)
    va, vb = a["CADC0884"].judge(feats), b["CADC0884"].judge(feats)
    assert va.detector_version == vb.detector_version
    assert va.model_sha256 != vb.model_sha256
    assert va.manifest_sha256 != vb.manifest_sha256
    assert va.recipe_sha256 == vb.recipe_sha256
    assert va.bundle_id == va.manifest_sha256
    # Files can be replaced after load; the held judge and produced verdict keep their identity.
    for path in second.iterdir():
        shutil.copyfile(path, first / path.name)
    assert a["CADC0884"].judge(feats).bundle_id == va.bundle_id
    assert m.load_bundle(first)["CADC0884"].judge(feats).bundle_id == vb.bundle_id


def test_real_closed_stream_context_retains_loaded_identity(package, monkeypatch):
    m = api()
    persistence = importlib.import_module("src.streaming.persistence")
    monkeypatch.setattr(persistence, "init_db", lambda: None)
    runtime = importlib.import_module("src.streaming.engine")
    monkeypatch.setattr(runtime, "init_alert_db", lambda: None)
    stored = []
    monkeypatch.setattr(runtime, "save_verdict", lambda **kwargs: stored.append(kwargs))
    monkeypatch.setattr(runtime, "save_alert", lambda *a, **k: None)
    engine = runtime.StreamEngine()
    engine.judges = m.load_bundle(package)
    engine.models_ready = True
    from src.streaming.segmenter import ClosedSegment
    seg = ClosedSegment("CADC0884", "identity-test", False, [1e9, -1e9, 1e9, -1e9], [0., 5., 10., 15.], 5)
    engine._judge_segment(seg)
    identity = engine.judges["CADC0884"].bundle_identity.as_fields()
    facts = stored[0]["features"]["_detector"]
    assert all(facts[name] == value for name, value in identity.items())
    assert all(engine.latest["CADC0884"]["judgement"][name] == value for name, value in identity.items())


def test_pipeline_and_empty_run_save_loaded_identity_without_rereading_switched_files(package, tmp_path, monkeypatch):
    from src.integration.anomaly_rag_pipeline import AnomalyRAGPipeline
    m = api()
    first = copy_package(package, tmp_path / "first")
    second = alternate_serialization(package, tmp_path / "second")
    monkeypatch.setattr(m, "DEFAULT_BUNDLE", first)
    pipeline = AnomalyRAGPipeline()
    points = m.pd.DataFrame({"channel": "CADC0884", "segment": "new", "sampling": 5,
        "timestamp": m.pd.date_range("2026-01-01", periods=4, freq="5s"), "value": (1e9, -1e9, 1e9, -1e9)})
    anomalies = pipeline.detect(points)
    assert anomalies
    identity = anomalies[0].bundle_id
    assert pipeline.last_seg_results.iloc[0].bundle_id == identity
    for path in second.iterdir():
        shutil.copyfile(path, first / path.name)
    output = pipeline.to_json(anomalies, tmp_path / "nonempty.json")
    assert json.loads(output.read_text(encoding="utf-8"))["bundle_id"] == identity
    # Select a real normal fit segment and run its actual points for the zero-anomaly case.
    canonical = m.pd.read_csv(m.ROOT / m.TRAINING_CSV, float_precision="round_trip")
    candidates = canonical[canonical.channel == "CADC0884"].head(10)
    predicted = m.predict_features(candidates, m.load_bundle(second))
    normal = predicted[~predicted.final_pred].iloc[0]
    raw = m.pd.read_csv(m.ROOT / "data/raw/segments.csv", float_precision="round_trip")
    assert pipeline.detect(raw[(raw.channel == normal.channel) & (raw.segment == normal.segment)]) == []
    second_id = m.load_bundle(second)["CADC0884"].judge(dict.fromkeys(m.FEATURE_COLS, 0.)).bundle_id
    for path in package.iterdir():
        shutil.copyfile(path, first / path.name)
    empty_output = pipeline.to_json([], tmp_path / "empty.json")
    empty = json.loads(empty_output.read_text(encoding="utf-8"))
    assert empty["results"] == []
    assert empty["bundle_id"] == second_id != identity
    assert empty["input_sha256"] == pipeline.last_input_sha256
