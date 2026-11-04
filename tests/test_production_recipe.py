"""Current engineering recipe contract; historical artifacts stay immutable."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_fixed_engineering_recipe_preserves_historical_parameters_and_nulls():
    from src.detection.frozen import load_recipe
    recipe = load_recipe()
    historical = json.loads((ROOT / "data/results/v3/fusion.json").read_text(encoding="utf-8"))["results"]["selection"]
    assert recipe["best_k"] == historical["best_k"]
    assert recipe["contamination"] == historical["best_contamination_per_channel"]
    assert recipe["gate_choice"] == {ch: v["op"] for ch, v in historical["gate_perchannel_choice"].items()}
    assert recipe["contamination"]["CADC0886"] is None
    assert recipe["contamination"]["CADC0890"] is None
    assert recipe["status"] == "engineering_fixed"
    assert recipe["postprocess"] == "identity"
    assert recipe["fit_rows"] == 1275
    assert "no new-feature search" in recipe["parameter_source"]


def test_pipeline_keeps_null_contamination_and_has_no_permanent_normal_channel():
    from src.integration.anomaly_rag_pipeline import AnomalyRAGPipeline
    from src.utils.constants import NO_ANOMALY_CHANNELS
    pipeline = AnomalyRAGPipeline()
    assert pipeline.best_contamination == pipeline.recipe["contamination"]
    assert not NO_ANOMALY_CHANNELS


def test_model_configuration_is_explicit_and_fixed():
    from src.detection.frozen import load_recipe
    recipe = load_recipe()
    assert recipe["isolation_forest"] == {"n_estimators": 100, "max_samples": 128, "random_state": 42, "n_jobs": 1}
    assert recipe["rules"]["minimum_nonnull_fit_rows"] == 5
    assert recipe["minimum_if_fit_rows"] == 10
    assert set(recipe["gate_choice"].values()) <= {"rule", "if", "AND", "OR"}


def test_declared_requirements_match_installed():
    """requirements.txt 里用 == 锁版本的包，实际安装版本必须一致。

    用途：干净环境按 requirements 装出来的版本应与本机验收环境相同。
    缺声明或版本漂移都会让「本机能跑」变成「别人跑不起来」。
    """
    import importlib.metadata as meta
    import re

    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    pinned = dict(re.findall(r"^([A-Za-z0-9_-]+)==([0-9.]+)$", req, re.M))

    # 三个包曾经完全未声明，导致干净环境无法启动后端
    for must in ("fastapi", "uvicorn", "requests"):
        assert must in pinned or f"{must}>=" in req, (
            f"requirements.txt 未声明 {must}，干净环境装完无法启动后端"
        )

    bad = []
    for p, v in sorted(pinned.items()):
        try:
            actual = meta.version(p)
        except meta.PackageNotFoundError:
            bad.append(f"{p}: 声明 {v} 但未安装")
            continue
        if actual != v:
            bad.append(f"{p}: 声明 {v} 实际 {actual}")
    assert not bad, "requirements 与实际安装版本不符：" + "; ".join(bad)


def test_readme_uses_python_dash_m_uvicorn():
    """README 必须写 `python -m uvicorn`，不能直接 `uvicorn`。

    实测：直接跑 uvicorn.exe 会使 sys.path[0] 变成 venv/Scripts，
    导致 `import src` 失败。新人照文档操作第一天就起不来服务。
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for line in readme.splitlines():
        stripped = line.strip()
        if stripped.startswith("uvicorn "):
            raise AssertionError(
                f"README 有裸 uvicorn 命令（会 ModuleNotFoundError: No module named 'src'）：{line.strip()[:60]}"
            )
