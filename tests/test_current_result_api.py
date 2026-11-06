import json
from pathlib import Path

import pytest
from fastapi import HTTPException


def test_current_result_endpoints_share_one_result_package():
    from src.api.routes import dashboard, detection
    root = Path(__file__).resolve().parents[1]
    metrics = json.loads((root / "data/results/graduation-confirmation-v1/metrics.json").read_text(encoding="utf-8"))
    experiments = detection.get_experiments()
    assert len(experiments["experiments"]) == 8
    assert experiments["source"]["protocol"] == "graduation-confirmation-v1"
    assert dashboard.get_core_metrics()["best_f1"] == round(metrics["overall"]["engineering_gate"]["f1"], 4)
    channels = detection.get_channel_f1()
    assert len(channels["channels"]) == 9
    for row in channels["channels"]:
        assert row["f1"] == round(metrics["per_channel"]["engineering_gate"][row["channel"]]["f1"], 4)
    params = detection.get_system_params()
    assert params["contamination"] is None
    assert len(params["contamination_by_channel"]) == 9


def test_missing_current_results_return_failure(monkeypatch, tmp_path):
    from src.api import result_data
    from src.api.routes import detection
    monkeypatch.setattr(result_data, "RESULTS", tmp_path)
    with pytest.raises(HTTPException) as failure:
        detection.get_experiments()
    assert failure.value.status_code == 503
