"""Outputs the pipeline emits (§1 "what we emit", §7 session finalisation).

These are the external contract. The sensor team consumes `OnsetEvent`; the app
consumes `StageEstimate` and `SessionRecord`. Changing a field here is a breaking
change for another team — hence the explicit `to_dict` rather than relying on
dataclass introspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from .types import SessionOutcome, Stage

__all__ = ["OnsetEvent", "StageEstimate", "StageBreakdown", "SessionRecord", "ProvenanceStamp"]


@dataclass(frozen=True, slots=True)
class ProvenanceStamp:
    """Stamped into every SessionRecord.

    You will need this the first time pilot numbers look wrong and nobody remembers
    which model produced them.
    """

    model_l1_version: str
    model_l2_version: str
    feature_dict_version: int
    transition_matrix_version: str
    pipeline_version: str
    #: The wake channel decides every wake minute in the record, so a session whose
    #: numbers look wrong cannot be traced without knowing which scorer produced them.
    #: "absent" is a real value: a session CAN run without the channel, degraded.
    model_waso_version: str = "absent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_l1": self.model_l1_version,
            "model_l2": self.model_l2_version,
            "model_waso": self.model_waso_version,
            "feature_dict": self.feature_dict_version,
            "transition_matrix": self.transition_matrix_version,
            "pipeline": self.pipeline_version,
        }


@dataclass(frozen=True, slots=True)
class OnsetEvent:
    """T_onset, fired once per session on AWAKE->...->ASLEEP confirmation (§4).

    §5 is emphatic: onset is an INTERVAL, never a minute-level claim. The row cadence
    is 15 min, so the honest statement is [T_confirmed - 15 min, T_confirmed]. Consumers
    should render the interval, and `confirmed_at` exists only because the sensor team
    needs a single instant to switch sampling mode on.
    """

    subject_id: str
    session_id: str
    confirmed_at: datetime
    probability: float
    via_fallback: bool = False

    @property
    def interval_start(self) -> datetime:
        return self.confirmed_at - timedelta(minutes=15)

    @property
    def interval(self) -> tuple[datetime, datetime]:
        return (self.interval_start, self.confirmed_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "onset_interval": [self.interval_start.isoformat(), self.confirmed_at.isoformat()],
            "probability": round(self.probability, 4),
            "via_fallback": self.via_fallback,
        }


@dataclass(frozen=True, slots=True)
class StageEstimate:
    """Per-row staging output, post-HMM (§7).

    `confidence` is the HMM forward-backward posterior for the decoded stage, NOT the
    raw Layer 2 softmax — §7 states the posterior is the better confidence signal.
    """

    start: datetime
    stage: Stage
    confidence: float
    posterior: np.ndarray
    raw_probabilities: np.ndarray
    quality: float = 1.0
    cold_start: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "stage": self.stage.label,
            "confidence": round(self.confidence, 4),
            "cold_start": self.cold_start,
        }


@dataclass(frozen=True, slots=True)
class StageBreakdown:
    """time_in_stage_min from the §7 payload."""

    wake: int
    light: int
    deep: int
    rem: int

    def to_dict(self) -> dict[str, int]:
        return {"wake": self.wake, "light": self.light, "deep": self.deep, "rem": self.rem}

    @property
    def total_min(self) -> int:
        return self.wake + self.light + self.deep + self.rem


@dataclass(slots=True)
class SessionRecord:
    """Final session payload — mirrors the §7 JSON exactly.

    `sleep_score` is Optional on purpose, for two reasons:
      1. Naps do not get one (§4 nap protection).
      2. The Sleep Score formula is not defined anywhere in v2.1 (see decisions D-004).
    """

    session_id: str
    subject_id: str
    onset_time: datetime
    wake_time: datetime
    duration_min: int
    time_in_stage_min: StageBreakdown
    waso_min: int
    #: INTERNAL. Wake minutes from the 30-second timeline. Kept because the Sleep Score
    #: needs a number, NOT because it is presentable: measured against PSG it carries
    #: 12.3 min of error on a true mean of 21.4 for a healthy cohort, against 12.7 for
    #: printing the population median every night. The band below is what a user sees.
    stage_conf_avg: float
    outcome: SessionOutcome = SessionOutcome.COMPLETE
    sleep_score: int | None = None
    cycle_phase: str = "unknown"
    #: The user-facing WASO statement: one of minimal / some / notable / high, with the
    #: band's own bounds in minutes. None when the night could not be scored, which is a
    #: state to report rather than fill in.
    waso_band: str | None = None
    waso_band_range: tuple[float, float] | None = None
    waso_band_confidence: str | None = None
    provenance: ProvenanceStamp | None = None
    hypnogram: list[StageEstimate] = field(default_factory=list)

    @property
    def onset_interval(self) -> tuple[datetime, datetime]:
        return (self.onset_time - timedelta(minutes=15), self.onset_time)

    def to_dict(self) -> dict[str, Any]:
        """The §7 payload. Field names are frozen — downstream teams parse these."""
        return {
            "session_id": self.session_id,
            "onset_time": self.onset_time.isoformat(),
            "onset_interval": [t.isoformat() for t in self.onset_interval],
            "wake_time": self.wake_time.isoformat(),
            "duration_min": self.duration_min,
            "time_in_stage_min": self.time_in_stage_min.to_dict(),
            "sleep_score": self.sleep_score,
            "waso_min": self.waso_min,
            "waso_band": None if self.waso_band is None else {
                "band": self.waso_band,
                "range_min": None if self.waso_band_range is None else self.waso_band_range[0],
                "range_max": None if self.waso_band_range is None else self.waso_band_range[1],
                "confidence": self.waso_band_confidence,
            },
            "stage_conf_avg": round(self.stage_conf_avg, 4),
            "cycle_phase": self.cycle_phase,
            "outcome": self.outcome.value,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }
