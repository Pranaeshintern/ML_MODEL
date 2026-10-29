"""The runtime — wiring for §3's architecture, revised per D-015.

    Row -> [state machine decides] -> Layer 1 | Layer 2 + wake channel -> session engine

TWO CHANNELS, ONE HYPNOGRAM
The stage classifier emits Light/Deep/REM only. Sleep-versus-wake is decided by a
separate 30-second accelerometer channel and merged in. Measured reasons:

  post-onset wake bouts average 2.8 min; a 15-min majority vote needs 7.5 to flip
  the 15-min model's WASO estimate is WORSE than predicting the population mean
  scored post-onset, the merge beats the 4-class model on every metric:
      wake F1 0.348 -> 0.471, macro-F1 0.508 -> 0.546, kappa 0.322 -> 0.353
  and it halves the flattering bias: TST +11.7 -> +6.0 min, WASO -11.7 -> -6.0

LAYER 3 IS GONE
HMM smoothing was implemented, measured and rejected (§7.5): it improved every
classification metric and degraded every derived one, buying per-row agreement by
erasing exactly the isolated wake rows the duration metrics are made of. The smoother
seam is retained in `interfaces` for the record but the runtime no longer calls it.

This is the walking skeleton: the whole path is wired and testable with stub
components, so real models drop into interfaces that already exist. It is also the
interface the sensor team and the app consume (§1).

Streaming, one row at a time, because that is how it runs on-device. The batch path
used in training is `replay.py`, which drives this same object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..config import Config
from ..core.session import SessionEngine
from ..core.state_machine import MutableOnsetContext, SleepStateMachine
from ..errors import NotFittedError
from ..events import OnsetEvent, ProvenanceStamp, SessionRecord, StageEstimate
from ..featuredict import FeatureDictionary
from ..interfaces import (OnsetDetector, StageClassifier, WakeDetector,
                          WasoBandEstimator)
from ..data.accel_epochs import ACCEL_EPOCH_FEATURES
from ..models.merge import DEFAULT_WAKE_THRESHOLD, ROW_EPOCHS, merge_hypnogram
from ..schema import Row
from ..types import SessionOutcome, SleepState, Stage

__all__ = ["SleepPipeline", "PipelineOutput"]

#: Position of `coverage` in the epoch feature vector. The band model refuses a night the
#: accelerometer barely covered, and it needs the mean coverage to make that call.
COVERAGE_COL = ACCEL_EPOCH_FEATURES.index("coverage")


@dataclass(slots=True)
class PipelineOutput:
    """What `push_row` returns. Mostly empty — events are sparse by nature."""

    state: SleepState
    onset_event: OnsetEvent | None = None
    stage_estimate: StageEstimate | None = None
    session: SessionRecord | None = None
    #: Set when this row could not be scored the normal way. "no_wake_channel" means no
    #: wake detector was injected at all; "no_motion" means this row arrived without
    #: usable accelerometer epochs. Either way the session will NOT finalise on its own
    #: (see SleepPipeline.motion_coverage) and the caller has to end it.
    degraded: str | None = None


@dataclass(slots=True)
class _AsleepBuffer:
    """Rows accumulated since onset.

    Layer 2 needs a sequence (rolling features in the MVP, T=8 context in the
    production TCN) and Layer 3 re-decodes the whole night as more rows arrive —
    Viterbi is globally optimal, so a stage decided earlier can legitimately be
    revised by later evidence. 480 ops per night; re-decoding is free.
    """

    rows: list[Row] = field(default_factory=list)
    gaps: list[bool] = field(default_factory=list)
    #: 30-second accelerometer features for each row's window, (ROW_EPOCHS, F).
    #: A row with no motion data contributes an empty array and its epochs fall back
    #: to the stage classifier alone — the merge can only ADD wake, never remove it.
    accel: list[np.ndarray] = field(default_factory=list)

    def append(self, row: Row, *, is_gap: bool = False,
               accel_epochs: np.ndarray | None = None) -> None:
        self.rows.append(row)
        self.gaps.append(is_gap)
        self.accel.append(np.empty((0, 0)) if accel_epochs is None
                          else np.asarray(accel_epochs, dtype=np.float64))

    def __len__(self) -> int:
        return len(self.rows)


class SleepPipeline:
    """One instance per session.

    Components are injected rather than constructed here: the roadmap swaps every one
    of them (rule -> tree for L1, XGBoost -> TCN for L2, fixed -> fitted HMM), and
    the eval harness needs to substitute stubs freely.
    """

    def __init__(
        self,
        cfg: Config,
        feature_dict: FeatureDictionary,
        *,
        onset_detector: OnsetDetector,
        stage_classifier: StageClassifier,
        wake_detector: WakeDetector | None = None,
        band_estimator: WasoBandEstimator | None = None,
        wake_threshold: float = DEFAULT_WAKE_THRESHOLD,
        session_id: str = "",
        subject_id: str,
        bedtime: datetime | None = None,
        cycle_phase: str = "unknown",
    ) -> None:
        self.cfg = cfg
        self.fd = feature_dict
        self.onset_detector = onset_detector
        self.stage_classifier = stage_classifier
        self.wake_detector = wake_detector
        self.band_estimator = band_estimator
        self.wake_threshold = wake_threshold
        self.cycle_phase = cycle_phase

        self.state_machine = SleepStateMachine(
            cfg.state_machine, session_id=session_id, subject_id=subject_id, bedtime=bedtime
        )
        self.session_engine = SessionEngine(provenance=self._stamp())
        self.context = MutableOnsetContext()
        self._buffer = _AsleepBuffer()
        self._last_hypnogram: np.ndarray | None = None      # 30-s stage codes
        self._last_row_probs: np.ndarray | None = None      # (R, 4)
        self._last_row_end: datetime | None = None
        self._last_pwake: np.ndarray | None = None      # 30-s P(wake), for the band
        self._rows_since_onset = 0
        self._rows_with_motion = 0

    # ---- motion availability -------------------------------------------------

    #: Share of post-onset rows that arrived with usable accelerometer epochs. Below
    #: this the session is reported as DEGRADED_NO_MOTION. Matches the WASO channel's
    #: own non-wear gate, so the two refuse the same nights.
    MIN_MOTION_COVERAGE = 0.5

    @property
    def motion_coverage(self) -> float:
        if self._rows_since_onset == 0:
            return 0.0
        return self._rows_with_motion / self._rows_since_onset

    @property
    def is_degraded(self) -> bool:
        """True when wake cannot be decided for this session.

        The sleep/wake decision belongs to the accelerometer (D-015) and the stage model
        has no Wake output, so a session without motion has no mechanism to reach
        WAKE_DET and will never finalise itself. The caller must supply `wake_time`.
        """
        return (self.wake_detector is None
                or self.motion_coverage < self.MIN_MOTION_COVERAGE)

    # ---- streaming entry point ----------------------------------------------

    def push_row(self, row: Row, *, is_gap: bool = False,
                 accel_epochs: np.ndarray | None = None) -> PipelineOutput:
        """Feed one 15-min row. This is the sensor-team-facing entry point (§1).

        `accel_epochs` is the (30, F) block of 30-second accelerometer features
        covering this row's window. The accelerometer runs continuously, so the
        device has them; without them the wake channel cannot run and the hypnogram
        falls back to the stage classifier alone, which under-reports wake by roughly
        12 minutes a night.
        """
        row.validate(self.fd, scope="shared")
        self._last_row_end = row.end
        self._update_context(row)

        state = self.state_machine.state

        if state.runs_onset_detector:
            p = self.onset_detector.predict_proba(row, self.context)
            event = self.state_machine.observe_onset(row, p)
            if event is not None:
                # The confirming row is the first row of the session (§4: T_onset is
                # the confirmation instant, and the interval reaches back 15 min).
                self._buffer.append(row, is_gap=is_gap, accel_epochs=accel_epochs)
                return PipelineOutput(state=self.state_machine.state, onset_event=event)
            return PipelineOutput(state=self.state_machine.state)

        if state.runs_stage_classifier:
            self._buffer.append(row, is_gap=is_gap, accel_epochs=accel_epochs)
            self._rows_since_onset += 1
            has_motion = accel_epochs is not None and np.asarray(accel_epochs).size > 0
            self._rows_with_motion += int(has_motion)
            degraded = ("no_wake_channel" if self.wake_detector is None
                        else None if has_motion else "no_motion")
            estimate = self._stage_row()
            # The stage fed back to the state machine is the MERGED one, so it can be
            # Wake even though the classifier itself never emits Wake. Without this the
            # 3-class model leaves `_consecutive_wake` permanently at zero and the
            # session never finalises.
            self.state_machine.observe_stage(row, estimate.stage)
            out = PipelineOutput(state=self.state_machine.state, stage_estimate=estimate,
                                 degraded=degraded)
            if self.state_machine.is_finalisable:
                out.session = self.finalise()
            return out

        return PipelineOutput(state=self.state_machine.state)

    # ---- layers 2 + 3 --------------------------------------------------------

    def _stage_row(self) -> StageEstimate:
        """Re-stage the whole session and return the newest row's estimate.

        The whole night is re-decoded each time because the stage model is a sequence
        model: later rows legitimately revise earlier ones. A night is at most ~40
        rows, so this is cheap.
        """
        probs = np.asarray(self.stage_classifier.predict_proba(self._buffer.rows),
                           dtype=np.float64)
        if probs.shape != (len(self._buffer), Stage.n_classes()):
            raise NotFittedError(
                f"stage classifier returned {probs.shape}, expected "
                f"({len(self._buffer)}, {Stage.n_classes()})"
            )
        self._last_row_probs = probs
        self._last_hypnogram = self._merged_hypnogram(probs)

        i = len(self._buffer) - 1
        # A row is Wake when the wake channel calls the MAJORITY of its 30 epochs
        # wake — the same majority rule the 15-minute grid uses everywhere else.
        epochs = self._last_hypnogram[i * ROW_EPOCHS:(i + 1) * ROW_EPOCHS]
        if epochs.size and (epochs == Stage.WAKE.value).mean() > 0.5:
            stage, conf = Stage.WAKE, float((epochs == Stage.WAKE.value).mean())
        else:
            sleep = probs[i, 1:]
            stage = Stage(int(sleep.argmax()) + 1)
            total = float(sleep.sum())
            conf = float(sleep.max() / total) if total > 0 else 0.0
        return StageEstimate(
            start=self._buffer.rows[i].start,
            stage=stage,
            confidence=conf,
            posterior=probs[i],
            raw_probabilities=probs[i],
            quality=self._buffer.rows[i].quality,
            cold_start=self.context.baseline_confidence < 1.0,
        )

    def _merged_hypnogram(self, probs: np.ndarray) -> np.ndarray:
        """(R, 4) row probabilities -> (R*30,) 30-second stage codes.

        The buffer starts at confirmed onset, so the row grid and the epoch grid share
        an origin and `epoch_offset` is zero here. That is NOT true when replaying a
        recording, where the row grid is indexed from recording start — see
        `merge_hypnogram`'s docstring, and the alignment tests.
        """
        n_rows = len(self._buffer)
        pwake = np.zeros(n_rows * ROW_EPOCHS)
        if self.wake_detector is not None:
            for i, ep in enumerate(self._buffer.accel):
                if ep.size == 0:
                    continue                     # no motion for this row: adds no wake
                p = np.asarray(self.wake_detector.predict_wake_proba(ep),
                               dtype=np.float64).ravel()
                n = min(p.size, ROW_EPOCHS)
                pwake[i * ROW_EPOCHS:i * ROW_EPOCHS + n] = p[:n]
        self._last_pwake = pwake
        return merge_hypnogram(probs, np.arange(n_rows), pwake,
                               epoch_offset=0, threshold=self.wake_threshold)

    def _waso_band(self) -> tuple[str, float, float, str] | None:
        """The night-level WASO band, from the same probabilities the timeline uses.

        The buffer starts at CONFIRMED ONSET, so every epoch here is already post-onset
        and the band's window is Layer 1's onset by construction -- one onset definition,
        without passing an index. Returns None when the estimator withholds a band or
        when no accelerometer data reached the buffer.
        """
        if self.band_estimator is None or self._last_pwake is None:
            return None
        epochs = [e for e in self._buffer.accel if e.size]
        if not epochs:
            return None
        x = np.vstack(epochs)
        n = min(len(x), len(self._last_pwake))
        if n == 0:
            return None
        cov = float(np.nanmean(x[:n, COVERAGE_COL])) if x.shape[1] > COVERAGE_COL else 1.0
        est = self.band_estimator.estimate_band(x[:n], self._last_pwake[:n], cov)
        if est is None:
            return None
        return (est.band.label, float(est.range_min), float(est.range_max),
                str(est.confidence))

    # ---- finalisation --------------------------------------------------------

    def finalise(self, wake_time: datetime | None = None, *,
                 history_nights: int | None = None) -> SessionRecord | None:
        """Build the session record (§7). Returns None if onset never confirmed.

        `history_nights` is how many nights the user has already recorded. The pipeline
        can't know it, so the caller has to pass it; without it the record's sleep_score
        stays None, because no score is given during the first 14 nights.
        """
        onset = self.state_machine.onset_event
        if onset is None or self._last_hypnogram is None:
            return None

        wake_time = wake_time or self._last_row_end or onset.confirmed_at
        outcome = SessionOutcome.COMPLETE
        if onset.via_fallback:
            outcome = SessionOutcome.TIMEOUT_FALLBACK
        # Duration-only nap check; the physiological half is stubbed (D-007), so a
        # nap flagged here is provisional.
        if self.state_machine.is_nap_candidate(wake_time):
            outcome = SessionOutcome.NAP
        # Checked last so it wins: a session scored without a usable wake channel is
        # degraded whatever else happened to it, because every wake minute in the
        # record would be missing rather than measured.
        if self.is_degraded:
            outcome = SessionOutcome.DEGRADED_NO_MOTION

        return self.session_engine.finalise(
            session_id=self.state_machine.session_id,
            subject_id=self.state_machine.subject_id,
            onset_time=onset.confirmed_at,
            wake_time=wake_time,
            hypnogram=self._hypnogram(),
            epoch_hypnogram=self._last_hypnogram,
            waso_band=self._waso_band(),
            outcome=outcome,
            cycle_phase=self.cycle_phase,
            history_nights=history_nights,
        )

    def _hypnogram(self) -> list[StageEstimate]:
        """One StageEstimate per 15-minute row, from the merged hypnogram.

        `SessionEngine` quantises time-in-stage to whole rows, so the record stays at
        15 minutes. The 30-second detail lives in `epoch_hypnogram()` — WASO and
        efficiency should be computed from THAT, because the whole point of the wake
        channel is bouts shorter than a row.
        """
        if self._last_hypnogram is None or self._last_row_probs is None:
            return []
        out: list[StageEstimate] = []
        for i, row in enumerate(self._buffer.rows):
            epochs = self._last_hypnogram[i * ROW_EPOCHS:(i + 1) * ROW_EPOCHS]
            wake_frac = float((epochs == Stage.WAKE.value).mean()) if epochs.size else 0.0
            probs = self._last_row_probs[i]
            if wake_frac > 0.5:
                stage, conf = Stage.WAKE, wake_frac
            else:
                sleep = probs[1:]
                stage = Stage(int(sleep.argmax()) + 1)
                total = float(sleep.sum())
                conf = float(sleep.max() / total) if total > 0 else 0.0
            out.append(StageEstimate(
                start=row.start, stage=stage, confidence=conf,
                posterior=probs, raw_probabilities=probs, quality=row.quality,
                cold_start=self.context.baseline_confidence < 1.0))
        return out

    def epoch_hypnogram(self) -> np.ndarray:
        """The 30-second hypnogram, indexed from confirmed onset. Empty before onset.

        This is the resolution the wake channel exists for: post-onset wake bouts
        average 2.8 minutes and are invisible in the 15-minute record above.
        """
        return (np.empty(0, dtype=np.int8) if self._last_hypnogram is None
                else self._last_hypnogram)

    # ---- internals -----------------------------------------------------------

    def _update_context(self, row: Row) -> None:
        try:
            still = row.get("still_fraction", self.fd)
        except Exception:
            still = None
        self.context.update(None if still is None or np.isnan(still) else still)

    def _stamp(self) -> ProvenanceStamp:
        return ProvenanceStamp(
            model_l1_version=getattr(self.onset_detector, "version", "unknown"),
            model_l2_version=getattr(self.stage_classifier, "version", "unknown"),
            feature_dict_version=self.fd.version,
            model_waso_version=getattr(self.wake_detector, "version", "absent"),
            transition_matrix_version=self.cfg.hmm.transition_matrix_version,
            pipeline_version=self.cfg.pipeline_version,
        )
