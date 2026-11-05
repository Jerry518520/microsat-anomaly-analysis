"""Explicit build and strict load of the A02 engineering detector.

No inference entry point fits a model. Historical stage2 pickles stay separate.
"""
from dataclasses import asdict, dataclass
import ast
import hashlib
import importlib.metadata
import io
import json
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import stat
import tempfile
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.features.segment_spec import (CANONICAL_FILENAME, FEATURE_COLS, FEATURE_VERSION,
                                       extract_features, feature_manifest, timestamp_seconds)
from src.utils.stat_rules import compute_stat_thresholds, sigma_rule, iqr_rule

ROOT = Path(__file__).resolve().parents[2]
VERSION = "a02-engineering-gate-v1"
DEFAULT_BUNDLE = ROOT / "data/models" / VERSION
RECIPE_PATH = ROOT / "configs/detectors" / f"{VERSION}.json"
TRAINING_CSV = f"data/processed/{CANONICAL_FILENAME}"
SPLIT_JSON = "data/results/v3/split_indices.json"
HISTORICAL_FUSION = "data/results/v3/fusion.json"
INPUT_FILES = [TRAINING_CSV, TRAINING_CSV.replace(".csv", ".manifest.json"),
               "data/raw/segments.csv", SPLIT_JSON, HISTORICAL_FUSION]
# Only the detector, feature and adapter source files affect this package.
SOURCE_FILES = ["src/detection/frozen.py", "src/features/segment_spec.py",
                "src/features/extract_segments_18d.py", "src/streaming/segment_features.py",
                "src/utils/stat_rules.py", "src/streaming/stage2.py"]
SOURCE_METHODS = {"src/integration/anomaly_rag_pipeline.py": ["AnomalyRAGPipeline.detect"],
                  "src/streaming/engine.py": ["StreamEngine._judge_segment"]}


class DetectorUnavailable(RuntimeError):
    """The fixed detector or a required component is unavailable."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_recipe():
    """Load the committed engineering choice; no parameter search or fallback."""
    recipe = _json(RECIPE_PATH)
    if recipe["detector_version"] != VERSION or recipe["feature_version"] != FEATURE_VERSION:
        raise DetectorUnavailable("recipe version mismatch")
    historical = _json(ROOT / HISTORICAL_FUSION)["results"]["selection"]
    expected = {"best_k": historical["best_k"],
                "contamination": historical["best_contamination_per_channel"],
                "gate_choice": {ch: item["op"] for ch, item in historical["gate_perchannel_choice"].items()}}
    for name, value in expected.items():
        if recipe[name] != value:
            raise DetectorUnavailable(f"historical parameter source mismatch: {name}")
    if recipe["postprocess"] != "identity":
        raise DetectorUnavailable("unsupported postprocess")
    return recipe


def gate(op, rule_pred, if_pred):
    if op not in {"rule", "if", "AND", "OR"}:
        raise DetectorUnavailable(f"unsupported gate: {op}")
    if op in {"rule", "AND", "OR"} and rule_pred is None:
        raise DetectorUnavailable(f"{op}: rule thresholds unavailable")
    if op in {"if", "AND", "OR"} and if_pred is None:
        raise DetectorUnavailable(f"{op}: IF unavailable")
    return bool({"rule": rule_pred, "if": if_pred, "AND": rule_pred and if_pred,
                 "OR": rule_pred or if_pred}[op])


@dataclass(frozen=True)
class BundleIdentity:
    """A snapshot of the bytes validated by one successful load."""
    model_sha256: str
    recipe_sha256: str
    manifest_sha256: str

    def as_fields(self):
        return {**asdict(self), "bundle_id": self.manifest_sha256}


@dataclass
class EngineeringVerdict:
    channel: str
    anomaly: bool
    if_pred: bool | None
    rule_hits: int
    rule_pred: bool | None
    score: float | None
    anomaly_score: float
    severity: str
    rule_total: int
    gate_op: str
    gate_pred: bool
    final_pred: bool
    postprocess: str
    display_score: float
    detector_version: str = VERSION
    feature_version: str = FEATURE_VERSION
    model_sha256: str | None = None
    recipe_sha256: str | None = None
    manifest_sha256: str | None = None
    bundle_id: str | None = None


@dataclass
class EngineeringJudge:
    channel: str
    model: IsolationForest | None
    thresholds: dict
    k: int = 2
    gate_op: str = "rule"
    score_tau95: float = 0.
    score_std: float = 1.
    detector_version: str = VERSION
    feature_cols: tuple = tuple(FEATURE_COLS)
    bundle_identity: BundleIdentity | None = None

    def _rules(self, feats):
        hits = total = 0
        for col, th in self.thresholds.items():
            value = feats[col]
            total += 1
            hits += int(sigma_rule(value, th["mean"], th["std"]) or
                        iqr_rule(value, th["q1"], th["q3"], th["iqr"]))
        return hits, total

    def rule_eval(self, feats):
        return self._rules(feats)[0]

    def continuous_score(self, vec):
        if self.model is None:
            return None
        return float(-self.model.decision_function(np.asarray(vec).reshape(1, -1))[0])

    def display_score(self, feats, hits, score):
        if score is None:
            return round(min(1., hits / 6.), 4)
        return round(min(1., max(0., score - self.score_tau95) / (3 * self.score_std)), 4)

    def judge(self, feats):
        vec = np.array([feats[c] for c in FEATURE_COLS], dtype=float)
        if not np.isfinite(vec).all():
            raise ValueError("features must be finite; no inference imputation")
        hits, total = self._rules(feats)
        rule_pred = bool(hits >= self.k) if total else None
        if_pred = bool(self.model.predict(vec.reshape(1, -1))[0] == -1) if self.model is not None else None
        score = self.continuous_score(vec)
        final = gate(self.gate_op, rule_pred, if_pred)
        display = self.display_score(feats, hits, score)
        severity = "nominal" if not final else ("critical" if hits >= 4 else "warning")
        return EngineeringVerdict(self.channel, final, if_pred, hits, rule_pred, score,
                                  display, severity, total, self.gate_op, final, final,
                                  "identity", display,
                                  **(self.bundle_identity.as_fields() if self.bundle_identity is not None else {}))


def fit_judges(fit, recipe):
    """Build only. Insufficient data leaves components explicitly unavailable."""
    thresholds = compute_stat_thresholds(fit, FEATURE_COLS)
    judges = {}
    for ch in fit.channel.unique():
        frame = fit[fit.channel == ch]
        model = None
        tau, std = 0., 1.
        contamination = recipe["contamination"][ch]
        if contamination is not None and len(frame) >= recipe["minimum_if_fit_rows"]:
            X = frame[FEATURE_COLS].to_numpy(dtype=float)
            if not np.isfinite(X).all():
                raise ValueError("fit features must be finite")
            model = IsolationForest(contamination=contamination, **recipe["isolation_forest"])
            model.fit(X)
            scores = -model.decision_function(X)
            tau, std = float(np.quantile(scores, .95)), float(np.std(scores)) or 1.
        judges[ch] = EngineeringJudge(ch, model, thresholds.get(ch, {}), recipe["best_k"],
                                      recipe["gate_choice"][ch], tau, std)
    return judges


def _identity(recipe):
    sources = {p: sha256(ROOT / p) for p in SOURCE_FILES}
    for filename, methods in SOURCE_METHODS.items():
        source = (ROOT / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for method in methods:
            cls, name = method.split(".")
            node = next(fn for item in tree.body if isinstance(item, ast.ClassDef) and item.name == cls
                        for fn in item.body if isinstance(fn, ast.FunctionDef) and fn.name == name)
            sources[f"{filename}#{method}"] = hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
    return {"detector_version": VERSION, "feature": feature_manifest(),
            "inputs": {p: sha256(ROOT / p) for p in INPUT_FILES},
            "sources": sources,
            "dependencies": {"python": platform.python_version(),
                             **{p: importlib.metadata.version(p) for p in
                                ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]}},
            "recipe_sha256": sha256(RECIPE_PATH), "recipe": recipe}


def _training_fit(recipe):
    """Reconstruct the expected fit population from verified canonical inputs."""
    provenance = _json((ROOT / TRAINING_CSV).with_suffix(".manifest.json"))
    for key, value in feature_manifest().items():
        if provenance.get(key) != value:
            raise DetectorUnavailable(f"canonical feature provenance mismatch: {key}")
    if provenance.get("dataset_sha256") != sha256(ROOT / TRAINING_CSV):
        raise DetectorUnavailable("canonical dataset hash mismatch; regenerate A01 explicitly")
    if provenance.get("raw_sha256") != sha256(ROOT / "data/raw/segments.csv"):
        raise DetectorUnavailable("canonical raw source hash mismatch")
    data = pd.read_csv(ROOT / TRAINING_CSV, float_precision="round_trip")
    if provenance.get("rows") != len(data) or not np.isfinite(data[FEATURE_COLS].to_numpy(dtype=float)).all():
        raise DetectorUnavailable("canonical dataset row/feature mismatch")
    split = _json(ROOT / SPLIT_JSON)
    sets = [set(split[name]) for name in ("fit_ids", "val_ids", "test_ids")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("split sets overlap")
    if data.segment.duplicated().any() or set(data.segment) != set.union(*sets):
        raise ValueError("training dataset/split identities mismatch")
    fit = data[data.segment.isin(sets[0])].sort_values("segment").reset_index(drop=True)
    if len(fit) != recipe["fit_rows"] or not fit.train.eq(1).all():
        raise ValueError("fit population mismatch")
    return fit


def _fit_identity(fit, recipe):
    counts = {str(k): int(v) for k, v in fit.groupby("channel").size().items()}
    return {"fit_rows": len(fit),
            "fit_ids_sha256": hashlib.sha256(json.dumps(sorted(fit.segment.tolist())).encode()).hexdigest(),
            "fit_rows_per_channel": counts,
            "effective_if_max_samples": {
                ch: min(recipe["isolation_forest"]["max_samples"], n)
                if recipe["contamination"][ch] is not None and n >= recipe["minimum_if_fit_rows"] else None
                for ch, n in counts.items()}}


def _validate_judges(judges, fit, recipe, fit_identity):
    """Check stored components and one fit vector per channel; never refit."""
    if not isinstance(judges, dict) or set(judges) != set(recipe["gate_choice"]):
        raise DetectorUnavailable("detector payload channel-set mismatch")
    thresholds = compute_stat_thresholds(fit, FEATURE_COLS)
    for ch, judge in judges.items():
        expected = {"channel": ch, "k": recipe["best_k"], "gate_op": recipe["gate_choice"][ch],
                    "feature_cols": tuple(FEATURE_COLS), "detector_version": VERSION,
                    "thresholds": thresholds[ch]}
        if type(judge) is not EngineeringJudge:
            raise DetectorUnavailable(f"detector payload judge-type mismatch: {ch}")
        for name, value in expected.items():
            if name not in vars(judge) or vars(judge)[name] != value:
                raise DetectorUnavailable(f"detector payload mismatch: {ch}.{name}")
        for name in ["score_tau95", "score_std", "model"]:
            if name not in vars(judge):
                raise DetectorUnavailable(f"detector payload missing: {ch}.{name}")
        if not np.isfinite([judge.score_tau95, judge.score_std]).all() or judge.score_std <= 0:
            raise DetectorUnavailable(f"detector payload display calibration invalid: {ch}")
        model = judge.model
        samples = fit_identity["effective_if_max_samples"][ch]
        if samples is None:
            if model is not None:
                raise DetectorUnavailable(f"detector payload unexpected IF: {ch}")
            continue
        if type(model) is not IsolationForest:
            raise DetectorUnavailable(f"detector payload missing/invalid IF: {ch}")
        params = IsolationForest(contamination=recipe["contamination"][ch], **recipe["isolation_forest"]).get_params(deep=False)
        if model.get_params(deep=False) != params:
            raise DetectorUnavailable(f"detector payload IF parameters mismatch: {ch}")
        offset = getattr(model, "offset_", None)
        if not isinstance(offset, Real) or not np.isfinite(offset):
            raise DetectorUnavailable(f"detector payload missing/invalid IF offset_: {ch}")
        attrs = {"n_features_in_": len(FEATURE_COLS), "_n_samples": fit_identity["fit_rows_per_channel"][ch],
                 "max_samples_": samples}
        if any(getattr(model, name, None) != value for name, value in attrs.items()):
            raise DetectorUnavailable(f"detector payload IF fit dimensions mismatch: {ch}")
        trees = getattr(model, "estimators_", [])
        if len(trees) != params["n_estimators"] or any(tree.n_features_in_ != len(FEATURE_COLS) for tree in trees):
            raise DetectorUnavailable(f"detector payload IF trees mismatch: {ch}")
    for ch, judge in judges.items():
        try:
            vector = fit[fit.channel == ch].iloc[0][FEATURE_COLS].to_dict()
            verdict = judge.judge(vector)
            scores = [verdict.display_score, verdict.anomaly_score]
            if judge.model is not None:
                scores.append(verdict.score)
            if not np.isfinite(scores).all():
                raise ValueError("nonfinite inference score")
        except Exception as exc:
            raise DetectorUnavailable(f"detector payload inference probe failed: {ch}: {exc}") from exc


def _build_package(bundle):
    """Write a new private staging directory, never a published package."""
    recipe = load_recipe()
    identity = _identity(recipe)
    fit = _training_fit(recipe)
    judges = fit_judges(fit, recipe)
    if set(judges) != set(recipe["gate_choice"]):
        raise DetectorUnavailable("fit channels mismatch")
    # Check required components now, before publishing a package.
    for ch, judge in judges.items():
        judge.judge(fit[fit.channel == ch].iloc[0][FEATURE_COLS].to_dict())
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "detector_recipe.json").write_bytes(RECIPE_PATH.read_bytes())
    joblib.dump({"detector_version": VERSION, "judges": judges}, bundle / "detector.joblib")
    manifest = {**identity, "model_sha256": sha256(bundle / "detector.joblib"),
                **_fit_identity(fit, recipe)}
    _write_json(bundle / "model_manifest.json", manifest)
    return manifest


PACKAGE_FILES = {"model_manifest.json", "detector_recipe.json", "detector.joblib"}


def _safe_target(path):
    """Refuse roots, containers, symlinks and Windows junction/reparse paths."""
    path = Path(path).expanduser()
    if ".." in path.parts:
        raise ValueError(f"unsafe package target traversal: {path}")
    path = path.absolute()
    reserved = {Path(path.anchor), Path.home().absolute(), ROOT, ROOT / "data",
                ROOT / "data/models", ROOT / "data/raw", ROOT / "data/processed"}
    if path in reserved or path in ROOT.parents:
        raise ValueError(f"unsafe package root/container target: {path}")
    for part in [*reversed(path.parents), path]:
        if os.path.lexists(part):
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"unsafe package link/reparse path: {part}")
            if part != path and not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"unsafe package parent: {part}")
    return path


def _replaceable_package(path):
    if not path.is_dir() or {item.name for item in path.iterdir()} != PACKAGE_FILES:
        raise ValueError(f"unsafe replacement target; expected only three package files: {path}")
    for item in path.iterdir():
        _safe_target(item)
        if not item.is_file():
            raise ValueError(f"unsafe package member: {item}")


def freeze_bundle(bundle=None, *, replace=False):
    """Explicit build; default refuses existing targets before reading or fitting.

    Replacement builds and verifies a sibling stage first, archives the entire
    old directory, then publishes by rename. Failed stages are kept for audit.
    """
    target = _safe_target(bundle if bundle is not None else DEFAULT_BUNDLE)
    if os.path.lexists(target):
        if not replace:
            raise FileExistsError(f"package target exists; use --replace to archive it: {target}")
        _replaceable_package(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(f".{target.name}.build.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    stage = archive = None
    try:
        # Recheck while holding the per-target publisher lock.
        _safe_target(target)
        if os.path.lexists(target):
            if not replace:
                raise FileExistsError(f"package target exists: {target}")
            _replaceable_package(target)
        stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
        manifest = _build_package(stage)
        load_bundle(stage)
        _safe_target(target)
        if os.path.lexists(target):
            if not replace:
                raise FileExistsError(f"package target appeared during build: {target}")
            _replaceable_package(target)
            archive = target.with_name(f"{target.name}.archive-{uuid4().hex}")
            os.rename(target, archive)
        try:
            if os.path.lexists(target):
                raise FileExistsError(f"package target appeared before publish: {target}")
            os.rename(stage, target)
        except Exception:
            if archive is not None and not os.path.lexists(target):
                os.rename(archive, target)
            raise
        return {**manifest, "archived_package": str(archive) if archive is not None else None}
    except Exception as exc:
        if stage is not None:
            exc.add_note(f"A02 stage retained if present: {stage}; previous package/archive: {archive or target}")
        raise
    finally:
        os.close(descriptor)
        lock.unlink()


def load_bundle(bundle=None):
    """Verify every expected identity and model bytes before joblib.load."""
    bundle = Path(bundle) if bundle is not None else DEFAULT_BUNDLE
    try:
        for name in ["model_manifest.json", "detector_recipe.json", "detector.joblib"]:
            if not (bundle / name).is_file():
                raise DetectorUnavailable(f"detector package missing: {bundle / name}; run scripts/freeze_detector.py explicitly")
        recipe = load_recipe()
        expected = _identity(recipe)
        manifest_bytes = (bundle / "model_manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        for name, value in expected.items():
            if manifest.get(name) != value:
                raise DetectorUnavailable(f"detector manifest mismatch: {name}")
        fit = _training_fit(recipe)
        fit_identity = _fit_identity(fit, recipe)
        for name, value in fit_identity.items():
            if manifest.get(name) != value:
                raise DetectorUnavailable(f"detector manifest mismatch: {name}")
        recipe_bytes = (bundle / "detector_recipe.json").read_bytes()
        recipe_digest = hashlib.sha256(recipe_bytes).hexdigest()
        if json.loads(recipe_bytes) != recipe or recipe_digest != manifest["recipe_sha256"]:
            raise DetectorUnavailable("detector recipe mismatch")
        model_bytes = (bundle / "detector.joblib").read_bytes()
        if manifest.get("model_sha256") != hashlib.sha256(model_bytes).hexdigest():
            raise DetectorUnavailable("detector manifest mismatch: model_sha256")
        payload = joblib.load(io.BytesIO(model_bytes))
        if payload.get("detector_version") != VERSION or set(payload["judges"]) != set(recipe["gate_choice"]):
            raise DetectorUnavailable("detector payload mismatch")
        _validate_judges(payload["judges"], fit, recipe, fit_identity)
        identity = BundleIdentity(manifest["model_sha256"], recipe_digest,
                                  hashlib.sha256(manifest_bytes).hexdigest())
        for judge in payload["judges"].values():
            judge.bundle_identity = identity
        return payload["judges"]
    except DetectorUnavailable:
        raise
    except Exception as exc:
        raise DetectorUnavailable(f"detector validation/load failed: {exc}") from exc


def point_features(points):
    """Extract supplied points by (channel, segment), without label dependence."""
    required = ["channel", "segment", "timestamp", "value", "sampling"]
    missing = sorted(set(required) - set(points.columns))
    if missing or points.empty:
        raise ValueError(f"nonempty point table required; missing columns: {missing}")
    if points[required].isna().any().any():
        raise ValueError("point inputs must be complete")
    for sid in points.segment:
        if isinstance(sid, (bool, np.bool_)) or not (isinstance(sid, Integral) or isinstance(sid, str) and sid.strip()):
            raise ValueError("segment must be an integer or nonempty string")
    rows = []
    for (ch, sid), frame in points.groupby(["channel", "segment"], sort=False):
        if frame.sampling.nunique() != 1:
            raise ValueError(f"{ch}/{sid}: inconsistent sampling")
        feats = extract_features(frame.value.to_numpy(), timestamp_seconds(frame.timestamp.to_numpy()), frame.sampling.iloc[0])
        label = frame.anomaly.iloc[0] if "anomaly" in frame and frame.anomaly.nunique(dropna=False) == 1 else None
        rows.append({"channel": ch, "segment": sid, "sampling": frame.sampling.iloc[0], "y_true": label, **feats})
    return pd.DataFrame(rows)


def predict_features(features, judges):
    rows = []
    for _, row in features.iterrows():
        if row.channel not in judges:
            raise DetectorUnavailable(f"channel unavailable: {row.channel}")
        verdict = judges[row.channel].judge(row[FEATURE_COLS].to_dict())
        rows.append({"segment": row.segment, "sampling": row.sampling,
                     "y_true": row.get("y_true", row.get("anomaly", None)), **asdict(verdict),
                     "is_anomaly": verdict.final_pred, "n_windows": 1})
    result = pd.DataFrame(rows)
    # Preserve None and segment identities even for a one-segment request.
    for name in ["segment", "y_true", "score", "if_pred", "rule_pred"]:
        result[name] = pd.Series([row[name] for row in rows], dtype=object)
    return result
