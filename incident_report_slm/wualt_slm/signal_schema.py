"""Compact backend signals in; separate audio and distress interpretations out."""
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

Paragraph = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1800)]
Issue = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]

class AttemptDiagnostic(StrictModel):
    attempt: int = Field(strict=True, ge=1, le=2)
    issues: list[Issue] = Field(min_length=1, max_length=20)

class SectionStatus(StrictModel):
    status: Literal["generated", "not_available", "failed"]
    reason: Literal["no_audio_evidence", "no_usable_audio_evidence", "grounding_failed", "generation_incomplete", "invalid_model_response", "model_unavailable"] | None = None
    attempts: int = Field(strict=True, ge=0, le=2)
    diagnostics: list[AttemptDiagnostic] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def coherent(self):
        if self.status == "generated" and (self.reason is not None or self.attempts == 0):
            raise ValueError("Generated sections require a model attempt and no failure reason")
        if self.status == "not_available" and (self.reason not in ("no_audio_evidence", "no_usable_audio_evidence") or self.attempts or self.diagnostics):
            raise ValueError("Unavailable audio is a data status, not a model generation")
        if self.status == "failed" and (self.reason not in ("grounding_failed", "generation_incomplete", "invalid_model_response", "model_unavailable") or self.attempts == 0):
            raise ValueError("Failed sections require a failure reason and an attempt")
        attempts = [item.attempt for item in self.diagnostics]
        if attempts != sorted(set(attempts)) or any(n > self.attempts for n in attempts):
            raise ValueError("Diagnostics must identify unique completed attempts in order")
        if self.status == "failed" and (not attempts or attempts[-1] != self.attempts):
            raise ValueError("Failure diagnostics must record the final rejected attempt")
        if self.status == "generated" and any(n >= self.attempts for n in attempts):
            raise ValueError("A successful attempt cannot also be recorded as rejected")
        return self

class SectionStatuses(StrictModel):
    audio: SectionStatus
    distress: SectionStatus

class InterpretationResponse(StrictModel):
    audio_interpretation: Paragraph | None
    distress_interpretation: Paragraph | None
    sections: SectionStatuses

    @model_validator(mode="after")
    def coherent(self):
        for name in ("audio", "distress"):
            text = getattr(self, name + "_interpretation")
            section = getattr(self.sections, name)
            if (text is not None) != (section.status == "generated"):
                raise ValueError("Only a generated section can contain interpretation prose")
        if self.sections.distress.status == "not_available":
            raise ValueError("The supplied state is always available to the distress call")
        return self
