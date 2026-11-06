"""Validated telemetry inputs and stream protocol errors."""
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.constants import CHANNEL_MAP


class StreamProtocolError(ValueError):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class TelemetryFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    ts: float = Field(ge=-62135596800, le=253402300799, allow_inf_nan=False)
    channel: str
    value: float = Field(allow_inf_nan=False)
    segment_id: int | None = None
    sampling: int = Field(default=1, gt=0)

    @field_validator("channel")
    @classmethod
    def known_channel(cls, value):
        if value not in CHANNEL_MAP:
            raise ValueError("unknown telemetry channel")
        return value


class SessionStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: str = Field(min_length=1, max_length=200)

    @field_validator("source_id")
    @classmethod
    def nonempty_source(cls, value):
        if not value.strip():
            raise ValueError("source_id must contain a name")
        return value


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: UUID


class IngestRequest(SessionRequest):
    batch_id: UUID
    frames: list[TelemetryFrame] = Field(min_length=1, max_length=1000)