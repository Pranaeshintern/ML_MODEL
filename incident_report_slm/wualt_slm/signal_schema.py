"""Compact backend signals in; a single interpretation out."""
from typing import Annotated, Literal
from pydantic import AwareDatetime, Field, StringConstraints, model_validator
from .schemas import RequestBase, SourceId, StrictModel
from .incident_schema import Window, Location, ChatMessage
from .physiology_schema import Quality, HRVMeasurement

Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Keyword = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]

class DetectedLabel(StrictModel):
    label: Label
    confidence: float | None = Field(default=None, strict=True, ge=0, le=1)

class SuppliedState(StrictModel):
    label: Literal["normal", "stress", "distress", "unknown"]
    confidence: float | None = Field(default=None, strict=True, ge=0, le=1)

class SoundEvent(DetectedLabel):
    at: AwareDatetime | None = None

class Keywords(StrictModel):
    stress: list[Keyword] = Field(default_factory=list, max_length=12)
    distress: list[Keyword] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def unique(self):
        for values in (self.stress, self.distress):
            if len({v.casefold() for v in values}) != len(values):
                raise ValueError("Duplicate keywords in one group")
        return self

class HRReference(StrictModel):
    bpm: float = Field(strict=True, gt=0, le=350)
    at: AwareDatetime
    comparable_context: bool | None = Field(default=None, strict=True)

class HeartRate(StrictModel):
    bpm: float = Field(strict=True, gt=0, le=350)
    at: AwareDatetime
    signal_quality: Quality = "unknown"
    reference: HRReference | None = None

    @model_validator(mode="after")
    def reference_before(self):
        if self.reference and self.reference.at > self.at:
            raise ValueError("HR reference must not be later than the measurement")
        return self

class InterpretationInput(RequestBase):
    schema_version: Literal["interpretation-v1"] = "interpretation-v1"
    request_id: SourceId | None = None
    time: AwareDatetime
    observation_window: Window | None = None
    state: SuppliedState
    transcript: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=3000)] | None = None
    transcript_reliability: Literal["usable", "limited", "unreliable", "unknown", "unavailable"] = "unknown"
    speaker_verified: bool = Field(default=False, strict=True)
    tone: DetectedLabel | None = None
    keywords: Keywords = Field(default_factory=Keywords)
    events: list[SoundEvent] = Field(default_factory=list, max_length=12)
    hr: HeartRate | None = None
    hrv: HRVMeasurement | None = None
    location: Location | None = None
    activity: Label | None = None
    chat_context: list[ChatMessage] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def coherent(self):
        if self.transcript is None and self.transcript_reliability not in ("unknown", "unavailable"):
            raise ValueError("No transcript: use unknown or unavailable reliability")
        if self.transcript is not None and self.transcript_reliability == "unavailable":
            raise ValueError("Supplied transcript cannot have unavailable reliability")
        if self.observation_window and self.observation_window.end > self.time:
            raise ValueError("Observation window must end at or before time")
        stamps = [e.at for e in self.events if e.at]
        if self.hr: stamps.append(self.hr.at)
        if any(at > self.time for at in stamps):
            raise ValueError("Observations cannot be after the snapshot time")
        if self.observation_window and any(not self.observation_window.start <= at <= self.observation_window.end for at in stamps):
            raise ValueError("Timed events and HR must be inside the supplied observation window")
        if self.hrv and self.hrv.window_end > self.time:
            raise ValueError("HRV cannot end after the snapshot time")
        if self.location and self.location.fixed_at > self.time:
            raise ValueError("Location fix cannot be after the snapshot time")
        if any(m.at > self.time for m in self.chat_context):
            raise ValueError("Chat messages cannot be after the snapshot time")
        if len(self.transcript or "") + sum(len(m.text) for m in self.chat_context) > 5000:
            raise ValueError("Select at most 5000 characters of transcript and chat")
        return self

class SignalSentence(StrictModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=700)]
    evidence_ids: list[Label] = Field(min_length=1, max_length=16)

class SignalDraft(StrictModel):
    sentences: list[SignalSentence] = Field(min_length=1, max_length=5)

class InterpretationResponse(StrictModel):
    interpretation: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1800)]
