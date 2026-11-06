"""Generate on demand from one durable event. No display-segment fallback."""
import hashlib
import json

from src.streaming.contracts import StreamProtocolError


class EventExplanationService:
    def __init__(self, store, pipeline_factory):
        self.store = store
        self.pipeline_factory = pipeline_factory

    def _event(self, event_id):
        event = self.store.get_event(event_id)
        if event is None:
            raise StreamProtocolError(404, "unknown_event", "event does not exist")
        return event

    def get(self, event_id):
        self._event(event_id)
        return self.store.get_explanation(event_id) or {"event_id": event_id, "status": "not_requested"}

    def generate(self, event_id, *, retry=False):
        event = self._event(event_id)
        if event["status"] != "judged" or not event.get("judgement", {}).get("final_pred"):
            raise StreamProtocolError(422, "event_not_explainable", "a complete anomalous event is required")
        previous = self.get(event_id)
        if previous["status"] == "generating" or (previous["status"] != "not_requested" and not retry):
            return previous
        digest = hashlib.sha256(json.dumps(event, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        inputs = {key: event[key] for key in ("event_id", "session_id", "source_id", "channel", "segment",
            "first_ts", "last_ts", "n_points", "sampling", "features", "judgement", "waveform")}
        inputs["event_sha256"] = digest
        attempt, claimed = self.store.claim_explanation(event_id, inputs, retry=retry)
        if not claimed:
            return attempt
        try:
            pipeline = self.pipeline_factory()
            info = {key: value for key, value in inputs.items() if key != "waveform"}
            result = pipeline.query(
                query=f"通道 {event['channel']} 事件 {event_id} 遥测异常的形态、证据和排查建议",
                query_type="anomaly", channel_id=event["channel"], anomaly_type="telemetry anomaly",
                anomaly_description=json.dumps({"features": event["features"], "judgement": event["judgement"]}, ensure_ascii=False),
                additional_info=info)
        except Exception as exc:
            result = {"status": "retrieval_failed", "answer": "", "sources": [], "context": "",
                      "prompts": None, "metadata": {"error": str(exc)}}
        return self.store.finish_explanation(attempt["attempt_id"], result)
