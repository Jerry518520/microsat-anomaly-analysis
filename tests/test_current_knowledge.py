"""Current facts and historical explanation identity."""
from pathlib import Path


def test_current_channel_labels_and_fact_boundaries():
    from src.utils.constants import CHANNEL_MAP, CHANNEL_PHYSICS
    for i, channel in enumerate(("CADC0872", "CADC0873", "CADC0874")):
        assert CHANNEL_MAP[channel] == f"磁力计{i}号"
        assert f"I_B_FB_MM_{i}" in CHANNEL_PHYSICS[channel]
    assert all("单位未确认" in text for text in CHANNEL_PHYSICS.values())
    from src.rag.prompts import PromptTemplates
    prompt = PromptTemplates().system_prompt
    assert "采样间隔以当前段" in prompt
    assert "单位未确认" in prompt


def test_current_knowledge_uses_current_result_identity():
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs/knowledge_base/experiment_knowledge.md").read_text(encoding="utf-8")
    assert "18d-canonical-v1" in text
    assert "0.6204" in text
    assert "graduation-confirmation-v1" in text


def test_legacy_detail_is_explicitly_unverified(monkeypatch):
    from src.api.routes import explanation
    monkeypatch.setattr(explanation, "load_json", lambda name: {"results": [
        {"segment": 7, "channel": "CADC0884", "rag_explanation": "历史解释"}]})
    result = explanation.get_detail(segment="7", channel="CADC0884")
    assert result["record_kind"] == "historical_cache"
    assert result["verification_status"] == "legacy_unverified"
    assert result["current_detector"] is False
