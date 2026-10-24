"""Server-to-report contract for the supplied colour-coded incident template."""
from datetime import datetime
from typing import Annotated, Literal
from pydantic import AwareDatetime, Field, field_validator, model_validator
from .schemas import RequestBase, SourceId, StrictModel
from .physiology_schema import Quality, HRVMeasurement, OtherMeasurement

ShortText = Annotated[str, Field(min_length=1, max_length=250)]


class Officer(StrictModel):
    name: ShortText | None = None
    officer_id: ShortText | None = None
    unit: ShortText | None = None


class Person(StrictModel):
    wearer_id: SourceId
    contact: ShortText | None = None
    guardian_name: ShortText | None = None
    guardian_contact: ShortText | None = None


class Window(StrictModel):
    start: AwareDatetime
    end: AwareDatetime
    context: ShortText | None = None

    @model_validator(mode='after')
    def ordered(self):
        if not 0 < (self.end-self.start).total_seconds() <= 86400:
            raise ValueError('Observation window must be positive and at most 24 hours')
        return self


class Location(StrictModel):
    latitude: float = Field(ge=-90,le=90)
    longitude: float = Field(ge=-180,le=180)
    fixed_at: AwareDatetime
    accuracy_m: float | None = Field(default=None,ge=0,le=100000)
    last_known: bool = False


class Physiology(StrictModel):
    hr_min_bpm: int | None = Field(default=None,strict=True,ge=1,le=350)
    hr_max_bpm: int | None = Field(default=None,strict=True,ge=1,le=350)
    hr_peak_bpm: int | None = Field(default=None,strict=True,ge=1,le=350)
    reference_median_bpm: int | None = Field(default=None,strict=True,ge=1,le=350)
    reference_type: Literal['pre_event','resting'] = 'pre_event'
    reference_at: AwareDatetime | None = None
    hr_signal_quality: Quality = 'unknown'
    hr_measured_at: AwareDatetime | None = None
    hrv: HRVMeasurement | None = None
    other_measurements: list[OtherMeasurement] = Field(default_factory=list, max_length=3)

    @model_validator(mode='after')
    def range_order(self):
        names = [m.name for m in self.other_measurements]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate physiological measurement name')
        if (self.hr_min_bpm is None) != (self.hr_max_bpm is None):
            raise ValueError('Supply both ends of a heart-rate range')
        if self.hr_min_bpm is not None and self.hr_min_bpm > self.hr_max_bpm:
            raise ValueError('Heart-rate range is reversed')
        if self.hr_peak_bpm is not None and self.hr_max_bpm is not None and self.hr_peak_bpm < self.hr_max_bpm:
            raise ValueError('Peak cannot be below the maximum')
        return self


class Motion(StrictModel):
    observation: ShortText | None = None


class Speaker(StrictModel):
    label: SourceId
    supplied_gender_label: Literal['Female','Male','Unknown'] = 'Unknown'


class TranscriptLine(StrictModel):
    at: AwareDatetime
    speaker: SourceId | None = None
    text: str = Field(min_length=1,max_length=500)

    @field_validator('text')
    @classmethod
    def nonblank_text(cls, value):
        if not value.strip():
            raise ValueError('Transcript text must contain words, not just whitespace')
        return value


class Audio(StrictModel):
    received: bool = Field(strict=True)
    sounds: ShortText | None = None
    tone: ShortText | None = None
    reliability: Literal['usable','limited','unreliable','unavailable'] = 'unavailable'
    source_verified: bool = False
    wearer_link_verified: bool = False
    speaker_count: int | None = Field(default=None,strict=True,ge=0,le=20)
    speakers: list[Speaker] = Field(default_factory=list,max_length=10)
    transcript: list[TranscriptLine] = Field(default_factory=list,max_length=6)
    keywords: list[ShortText] = Field(default_factory=list,max_length=8)

    @model_validator(mode='after')
    def coherent(self):
        if not self.received and (self.sounds or self.tone or self.speaker_count is not None or self.speakers or self.transcript or self.keywords):
            raise ValueError('Missing audio cannot contain sound, speaker or transcript findings')
        if not self.received and self.reliability!='unavailable':
            raise ValueError('Missing audio reliability must be unavailable')
        labels=[s.label for s in self.speakers]
        if len(set(labels))!=len(labels):raise ValueError('Duplicate speaker labels')
        if self.speaker_count is not None and len(labels)>self.speaker_count:raise ValueError('Speaker labels exceed speaker count')
        if any(t.speaker is not None and t.speaker not in labels for t in self.transcript):raise ValueError('Transcript references an unknown speaker label')
        return self


class Button(StrictModel):
    state: Literal['sos','none_recorded','unavailable']
    at: AwareDatetime | None = None
    duration_seconds: float | None = Field(default=None,gt=0,le=600)

    @model_validator(mode='after')
    def coherent(self):
        if self.state=='sos' and self.at is None:raise ValueError('SOS requires its recorded timestamp')
        if self.state!='sos' and (self.at is not None or self.duration_seconds is not None):raise ValueError('A missing press cannot have a time or duration')
        return self


class Event(StrictModel):
    source_id: SourceId
    at: AwareDatetime
    observation: ShortText


class Assessment(StrictModel):
    # A server-owned contextual assessment, not a score learned from four samples.
    context: Literal['possible_distress','exercise_consistent','moderate_concern','insufficient_context','inconclusive'] = 'insufficient_context'
    provenance: ShortText = 'Not supplied'


class Activity(StrictModel):
    """An upstream activity classification, not inferred motion detail."""
    label: Annotated[str, Field(min_length=1, max_length=60)]

    @field_validator('label')
    @classmethod
    def nonblank_label(cls, value):
        if not value.strip():
            raise ValueError('Activity label cannot be blank')
        return value


class TranscriptAudio(StrictModel):
    """Text supplied by a transcription service; no acoustic claims implied."""
    transcript: list[TranscriptLine] = Field(default_factory=list, max_length=12)
    transcript_reliability: Literal['usable', 'limited', 'unreliable', 'unavailable']
    keywords: list[ShortText] = Field(default_factory=list, max_length=12)

    @field_validator('keywords')
    @classmethod
    def nonblank_keywords(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError('Keywords cannot be blank')
        if len({value.casefold().strip() for value in values}) != len(values):
            raise ValueError('Duplicate keywords')
        return values

    @model_validator(mode='after')
    def coherent(self):
        if bool(self.transcript) == (self.transcript_reliability == 'unavailable'):
            raise ValueError('An empty transcript requires unavailable reliability; supplied text cannot be unavailable')
        return self


class ChatMessage(StrictModel):
    at: AwareDatetime
    role: Literal['user', 'assistant']
    text: str = Field(min_length=1, max_length=1000)

    @field_validator('text')
    @classmethod
    def nonblank_text(cls, value):
        if not value.strip():
            raise ValueError('Chat text must contain words, not just whitespace')
        return value


class ChatContext(StrictModel):
    messages: list[ChatMessage] = Field(default_factory=list, max_length=12)


class IncidentBundle(RequestBase):
    schema_version: Literal['incident-v1', 'incident-v2'] = 'incident-v1'
    request_id: SourceId
    incident_id: SourceId
    report_number: SourceId
    reported_at: AwareDatetime
    incident_start: AwareDatetime
    incident_type: ShortText
    incident_description: str | None = Field(default=None, max_length=2000)
    clarification_required: str | None = Field(default=None, max_length=1000)
    observation_window: Window
    officer: Officer = Field(default_factory=Officer)
    wearer: Person
    location: Location | None = None
    physiology: Physiology = Field(default_factory=Physiology)
    motion: Motion = Field(default_factory=Motion)
    activity: Activity | None = None
    audio: Audio | TranscriptAudio
    chat_context: ChatContext = Field(default_factory=ChatContext)
    button: Button
    events: list[Event] = Field(default_factory=list,max_length=15)
    assessment: Assessment = Field(default_factory=Assessment)
    limitations: list[ShortText] = Field(default_factory=list,max_length=8)
    follow_up: str | None = Field(default=None,max_length=400)
    recorded_status: str | None = Field(default=None,max_length=400)
    acknowledgement: ShortText | None = None
    synthetic: bool = False

    @field_validator('audio', mode='before')
    @classmethod
    def audio_for_schema(cls, value, info):
        # Validate the declared contract directly, so a v2 typo does not return
        # unrelated errors demanding legacy raw-audio fields such as received.
        contract = TranscriptAudio if info.data.get('schema_version') == 'incident-v2' else Audio
        return contract.model_validate(value)

    @model_validator(mode='after')
    def times(self):
        if self.schema_version == 'incident-v2':
            if not isinstance(self.audio, TranscriptAudio):
                raise ValueError('incident-v2 requires audio.transcript_reliability and transcript text')
            if self.motion.observation is not None:
                raise ValueError('incident-v2 uses activity.label instead of motion.observation')
        elif not isinstance(self.audio, Audio) or self.activity is not None or self.chat_context.messages:
            raise ValueError('Activity, chat context and transcript-only audio require incident-v2')
        w=self.observation_window
        if not w.start<=self.incident_start<=w.end:raise ValueError('Incident start must be inside the observation window')
        if self.reported_at<w.end:raise ValueError('Report time cannot precede the end of the observation window')
        stamps=[e.at for e in self.events]+[t.at for t in self.audio.transcript]+([self.button.at] if self.button.at else [])
        if any(not w.start<=at<=w.end for at in stamps):raise ValueError('Event times must be inside the observation window')
        if len({e.source_id for e in self.events})!=len(self.events):raise ValueError('Duplicate event source IDs')
        if self.location and self.location.fixed_at>self.reported_at:raise ValueError('Location fix cannot be from after report time')
        p = self.physiology
        if p.reference_at and p.reference_at > w.start:
            raise ValueError('Heart-rate reference must be from before the observation window')
        if p.hr_measured_at and not w.start <= p.hr_measured_at <= w.end:
            raise ValueError('Heart-rate measurement time must be inside the observation window')
        if p.hrv and p.hrv.window_end > self.reported_at:
            raise ValueError('HRV measurement cannot end after report time')
        if any(m.at > self.reported_at for m in p.other_measurements):
            raise ValueError('Physiological measurements cannot be from after report time')
        if self.schema_version == 'incident-v1' and (p.hrv or p.other_measurements):
            raise ValueError('HRV and additional physiological signals require incident-v2')
        if any(message.at > self.reported_at for message in self.chat_context.messages):
            raise ValueError('Chat messages cannot be from after report time')
        if self.schema_version == 'incident-v2' and sum(len(t.text) for t in self.audio.transcript) + sum(len(m.text) for m in self.chat_context.messages) > 4000:
            raise ValueError('Transcript and chat text together must be at most 4000 characters; select the relevant context')
        return self


class Narrative(StrictModel):
    text: str = Field(min_length=1,max_length=1500)
    evidence_ids: list[str] = Field(min_length=1,max_length=25)


class IncidentDraft(StrictModel):
    description: Narrative
    interpretation: Narrative
    interpretation_attempts: int = Field(default=1,strict=True,ge=1,le=2)
    interpretation_selected_attempt: int = Field(default=1,strict=True,ge=1,le=2)
    interpretation_quality_issues: list[str] = Field(default_factory=list,max_length=25)


class ContextualAnalysis(StrictModel):
    interpretation: Narrative
    clarification: Narrative


class GroundedSentence(Narrative):
    kind: Literal['interpretation', 'limitation', 'priority']


class GroundedAnalysis(StrictModel):
    sentences: list[GroundedSentence] = Field(min_length=1, max_length=5)


class InterpretationAttempt(StrictModel):
    attempt: int = Field(ge=1, le=2)
    sentences: list[GroundedSentence] = Field(min_length=1, max_length=5)
    diagnostic_issues: list[str] = Field(max_length=50)


class ContextualDraft(IncidentDraft):
    clarification: Narrative
    chat_interpretation: Narrative
    chat_interpretation_attempts: int = Field(default=1, ge=1, le=2)
    chat_interpretation_selected_attempt: int = Field(default=1, ge=1, le=2)
    interpretation_sentences: list[GroundedSentence] = Field(default_factory=list, max_length=5)
    interpretation_attempts_trace: list[InterpretationAttempt] = Field(default_factory=list, max_length=2)

    @model_validator(mode='after')
    def sentence_provenance_matches_paragraph(self):
        if self.interpretation_sentences:
            text = ' '.join(s.text.strip() for s in self.interpretation_sentences)
            ids = list(dict.fromkeys(i for s in self.interpretation_sentences for i in s.evidence_ids))
            if self.interpretation.text != text or self.interpretation.evidence_ids != ids:
                raise ValueError('Interpretation paragraph must match its model-generated sentences and citations')
        return self



class InterpretationDraft(StrictModel):
    interpretation: Narrative
    interpretation_attempts: int = Field(default=1, strict=True, ge=1, le=2)
    interpretation_selected_attempt: int = Field(default=1, strict=True, ge=1, le=2)
    interpretation_quality_issues: list[str] = Field(default_factory=list, max_length=25)
    interpretation_sentences: list[GroundedSentence] = Field(min_length=1, max_length=5)
    interpretation_attempts_trace: list[InterpretationAttempt] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def matches_sentences(self):
        text = " ".join(s.text.strip() for s in self.interpretation_sentences)
        ids = list(dict.fromkeys(i for s in self.interpretation_sentences for i in s.evidence_ids))
        if self.interpretation.text != text or self.interpretation.evidence_ids != ids:
            raise ValueError("Interpretation must match generated sentences and citations")
        return self
