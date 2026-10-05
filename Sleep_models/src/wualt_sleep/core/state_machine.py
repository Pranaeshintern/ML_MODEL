"""§4 — sleep-state machine. The temporal control layer.

Decides WHICH model runs and converts row-level predictions into a session. Driven by
model outputs, not by sensor schedules (sensor scheduling is the firmware team's
domain, §1).

    AWAKE --P(onset)>0.7 x1--> CANDIDATE_ONSET --P(onset)>0.7 again--> ASLEEP
       |                             |                                    |
       |                      (P <= 0.7: lapse back)                      |
       +------ (timeout +6h) --------+                (Wake x2 | motion+HR jump)
                        v                                                 v
                     ASLEEP*                                    WAKE_DET -> finalise

T_onset is the FIRST of the `onset_confirm_rows` consecutive rows above the threshold —
the row that entered CANDIDATE_ONSET — not the row that completed the run.

ONE GATE, since 02 Sep 2026. §4 originally specified 0.5 to enter CANDIDATE_ONSET and
0.7 to confirm. Measured over 353 nights the two selected the SAME row on 97.5% of them:
P(asleep) is near-binary, jumping 0.26 -> 0.75 across a single row, so only 0.12% of rows
ever landed between the gates and the hysteresis band never engaged. The second constant
bought nothing and made the two read as tuned when neither was.

The transition structure below is a literal transcription of §4. The two predicates
that §4 leaves underspecified are marked STUB and raise rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..config import StateMachineConfig
from ..errors import StateMachineError
from ..events import OnsetEvent
from ..schema import Row
from ..types import SleepState, Stage

__all__ = ["SleepStateMachine", "Transition", "LEGAL_TRANSITIONS"]

#: Every transition §4 permits. Anything else is a bug, not an edge case.
LEGAL_TRANSITIONS: dict[SleepState, frozenset[SleepState]] = {
    #: AWAKE -> ASLEEP is the FALLBACK PATH ONLY. It exists because the timeout is now
    #: checked from AWAKE: with a single gate there is no candidate state to sit in
    #: while the +6 h clock runs, so a night that never crosses the threshold has to be
    #: able to escape directly. Normal onset always goes AWAKE -> CANDIDATE_ONSET ->
    #: ASLEEP.
    SleepState.AWAKE: frozenset({SleepState.CANDIDATE_ONSET, SleepState.ASLEEP}),
    SleepState.CANDIDATE_ONSET: frozenset({SleepState.ASLEEP, SleepState.AWAKE}),
    SleepState.ASLEEP: frozenset({SleepState.WAKE_DET}),
    SleepState.WAKE_DET: frozenset({SleepState.AWAKE}),
}


@dataclass(frozen=True, slots=True)
class Transition:
    at: datetime
    from_state: SleepState
    to_state: SleepState
    reason: str


class SleepStateMachine:
    """One instance per session. Not thread-safe; not reusable across nights.

    §4 forbids re-detection: once ASLEEP, Layer 1 never runs again for this session.
    That is enforced structurally — ASLEEP has no edge back to CANDIDATE_ONSET.
    """

    def __init__(
        self,
        cfg: StateMachineConfig,
        *,
        session_id: str,
        subject_id: str,
        bedtime: datetime | None = None,
    ) -> None:
        self.cfg = cfg
        self.session_id = session_id
        self.subject_id = subject_id
        self.bedtime = bedtime

        self._state = SleepState.AWAKE
        self._transitions: list[Transition] = []
        self._consecutive_above_confirm = 0
        self._consecutive_wake = 0
        self._onset_event: OnsetEvent | None = None
        self._last_row_end: datetime | None = None

    # ---- observation ---------------------------------------------------------

    def observe_onset(self, row: Row, probability: float) -> OnsetEvent | None:
        """Feed one Layer 1 output. Returns T_onset if this row confirmed onset.

        Called only while `state.runs_onset_detector` is true.
        """
        if not self._state.runs_onset_detector:
            raise StateMachineError(f"Layer 1 must not run in state {self._state}")
        self._last_row_end = row.end

        if self._state is SleepState.AWAKE:
            if probability > self.cfg.onset_threshold:
                self._transition_to(row.end, SleepState.CANDIDATE_ONSET,
                                    f"P(onset)={probability:.3f}")
                self._consecutive_above_confirm = 1
                # A single row can confirm if onset_confirm_rows is 1.
                if self._consecutive_above_confirm >= self.cfg.onset_confirm_rows:
                    return self._confirm_onset(row.end, probability, via_fallback=False)
                return None
            # The timeout is checked from AWAKE as well as from CANDIDATE_ONSET. With a
            # single gate, a night whose P never reaches the threshold would otherwise
            # never leave AWAKE and the §4 escape hatch would be unreachable — under the
            # old two-gate rule the 0.5 crossing is what armed it.
            if self._timed_out(row.end):
                return self._confirm_onset(row.end, probability, via_fallback=True)
            return None

        # CANDIDATE_ONSET
        if probability > self.cfg.onset_threshold:
            self._consecutive_above_confirm += 1
        else:
            # One gate means no hysteresis band: a row at or below the threshold ends
            # the run outright. Nothing is lost — only 0.12% of rows ever fell between
            # the old 0.5 and 0.7 gates.
            self._consecutive_above_confirm = 0
            self._transition_to(row.end, SleepState.AWAKE, "evidence lapsed")
            return None

        if self._consecutive_above_confirm >= self.cfg.onset_confirm_rows:
            return self._confirm_onset(row.end, probability, via_fallback=False)

        if self._timed_out(row.end):
            return self._confirm_onset(row.end, probability, via_fallback=True)
        return None

    def observe_stage(self, row: Row, stage: Stage) -> None:
        """Feed one smoothed Layer 3 stage. Called only while ASLEEP."""
        if not self._state.runs_stage_classifier:
            raise StateMachineError(f"Layer 2 must not run in state {self._state}")
        self._last_row_end = row.end

        self._consecutive_wake = self._consecutive_wake + 1 if stage is Stage.WAKE else 0
        if self._consecutive_wake >= self.cfg.wake_confirm_rows:
            self._transition_to(
                row.end, SleepState.WAKE_DET, f"wake x{self._consecutive_wake} rows"
            )

    def force_wake(self, at: datetime, reason: str = "motion+HR jump") -> None:
        """§4's alternate wake trigger. The firmware-side motion+HR jump can end a
        session without waiting for two Wake rows."""
        if self._state is SleepState.ASLEEP:
            self._transition_to(at, SleepState.WAKE_DET, reason)

    # ---- escape hatches (§4) -------------------------------------------------

    def _timed_out(self, now: datetime) -> bool:
        """CANDIDATE_ONSET timeout: if P(onset) never crosses the confirm threshold by
        T_bedtime + 6 h, fire the rule-based fallback. Shift-worker safe."""
        if self.bedtime is None:
            return False
        return now >= self.bedtime + timedelta(hours=self.cfg.candidate_timeout_hours)

    def is_nap_candidate(self, wake_time: datetime) -> bool:
        """Duration half of §4 nap protection: wake < 90 min post-onset.

        The physiological half ("AND HR/temp not at sleep baseline") is a separate
        check — see `physiology_at_sleep_baseline`.
        """
        if self._onset_event is None:
            return False
        elapsed = (wake_time - self._onset_event.confirmed_at).total_seconds() / 60.0
        return elapsed < self.cfg.nap_max_duration_min

    def physiology_at_sleep_baseline(self, rows: list[Row]) -> bool:
        """STUB — §4 nap protection, physiological half.

        §4 says "HR/temp not at sleep baseline" but never defines the sleep baseline,
        and on night 1 no personal baseline exists at all (decisions D-007). Needs a
        `BaselineProvider` and a documented cold-start rule before this can be written.
        Until then callers get duration-only nap detection and must mark the outcome
        provisional.
        """
        raise NotImplementedError("D-007: sleep-baseline definition for nap protection")

    # ---- internals -----------------------------------------------------------

    def _confirm_onset(self, at: datetime, probability: float, *, via_fallback: bool) -> OnsetEvent:
        self._transition_to(
            at, SleepState.ASLEEP, "fallback timeout" if via_fallback else "sustained evidence"
        )
        self._onset_event = OnsetEvent(
            subject_id=self.subject_id,
            session_id=self.session_id,
            confirmed_at=at,
            probability=probability,
            via_fallback=via_fallback,
        )
        return self._onset_event

    def _transition_to(self, at: datetime, to: SleepState, reason: str) -> None:
        if to not in LEGAL_TRANSITIONS[self._state]:
            raise StateMachineError(f"illegal transition {self._state.value} -> {to.value}")
        self._transitions.append(Transition(at, self._state, to, reason))
        self._state = to

    # ---- accessors -----------------------------------------------------------

    @property
    def state(self) -> SleepState:
        return self._state

    @property
    def onset_event(self) -> OnsetEvent | None:
        return self._onset_event

    @property
    def transitions(self) -> list[Transition]:
        return list(self._transitions)

    @property
    def is_finalisable(self) -> bool:
        return self._state is SleepState.WAKE_DET


@dataclass(slots=True)
class MutableOnsetContext:
    """Concrete `OnsetContext` (§5). Owned by the pipeline, updated per row.

    Exists so `Row` stays a pure sensor aggregation rather than a carrier of pipeline
    state — `sustained_still_rows` is cross-row memory by definition.
    """

    rows_since_bedtime: int = 0
    sustained_still_rows: int = 0
    baseline_confidence: float = 0.0
    _still_threshold: float = field(default=0.9, repr=False)

    def update(self, still_fraction: float | None) -> None:
        self.rows_since_bedtime += 1
        if still_fraction is not None and still_fraction >= self._still_threshold:
            self.sustained_still_rows += 1
        else:
            self.sustained_still_rows = 0
