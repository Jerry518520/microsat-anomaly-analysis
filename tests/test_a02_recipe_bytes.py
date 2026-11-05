"""A frozen recipe keeps the exact bytes whose identity was registered."""
import hashlib
import importlib
import json

import pytest


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("indent", [2, 4], ids=["indent2", "indent4"])
def test_real_freeze_and_load_preserve_recipe_bytes(tmp_path, monkeypatch, newline, indent):
    m = importlib.import_module("src.detection.frozen")
    recipe = m.load_recipe()
    raw = (json.dumps(recipe, ensure_ascii=False, indent=indent) + "\n").replace("\n", newline).encode("utf-8")
    source = tmp_path / "input-recipe.json"
    source.write_bytes(raw)
    monkeypatch.setattr(m, "RECIPE_PATH", source)
    assert m.load_recipe() == recipe

    target = tmp_path / "bundle"
    manifest = m.freeze_bundle(target)
    assert (target / "detector_recipe.json").read_bytes() == source.read_bytes() == raw
    assert manifest["recipe_sha256"] == hashlib.sha256(raw).hexdigest()

    # The explicit build uses real fitting. Runtime loading must only validate.
    monkeypatch.setattr(m.IsolationForest, "fit", lambda *a, **k: pytest.fail("load trained"))
    judges = m.load_bundle(target)
    assert set(judges) == set(recipe["gate_choice"])
    assert all(judge.bundle_identity.recipe_sha256 == manifest["recipe_sha256"] for judge in judges.values())
