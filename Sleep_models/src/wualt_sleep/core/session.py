"""§7 — session engine. Integrates the smoothed hypnogram into the session record.

Sleep Score, WASO and duration feed the Readiness score as the previous-night input,
so field semantics here are load-bearing outside this repo.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ..errors import NotFittedError
from ..events import ProvenanceStamp, SessionRecord, StageBreakdown, StageEstimate
from ..types import SessionOutcome, Stage

#: The Sleep Score is its OWN distribution (`wualt-sleep-score`), because the Readiness
#: composite depends on it too and must not inherit this project's numpy/scipy runtime.
#: It is therefore optional here: without it the pipeline still streams a night and
#: returns a complete session record, with `sleep_score` null — which is already a legal
#: state, since no score is emitted during a user's first 14 nights either. Anything that
#: actually needs a score raises and names what is missing, rather than inventing one.
try:
    from Sleep_Score_Algorithm import (
        ScoreConfig,
        SleepScoreInputs,
        SleepScoreResult,
        compute_sleep_score,
    )
    from Sleep_Score_Algorithm import count_awakening_bouts as _count_bouts

    SLEEP_SCORE_INSTALLED = True
except ModuleNotFoundError:                       # pragma: no cover - packaging path
    SLEEP_SCORE_INSTALLED = False
    ScoreConfig = SleepScoreInputs = SleepScoreResult = None  # type: ignore[assignment]

    def _absent(*_args, **_kwargs):
        raise NotFittedError(
            "the Sleep Score is not installed. It ships separately as "
            "`wualt-sleep-score` (./Sleep_Score_Algorithm); install it to score a "
            "session, or leave it out and read `sleep_score` as null.")

    compute_sleep_score = _count_bouts = _absent      # type: ignore[assignment]

__all__ = ["SessionEngine", "compute_breakdown", "compute_waso",
           "count_awakening_bouts", "score_from_record", "SLEEP_SCORE_INSTALLED"]

ROW_MINUTES = 15
EPOCH_MIN = 0.5


def breakdown_from_epochs(epochs: np.ndarray) -> StageBreakdown:
    """Minutes per stage from the MERGED 30-second timeline (D-015).

    This is the canonical source. Counting wake rows instead quantises every wake bout
    to 15 minutes, and the whole reason the wake channel exists is that post-onset bouts
    average 2.8 min — so a row count cannot represent them at all. Stage IDENTITY is
    still decided per row; only the duration attribution is at 30 s, which is exactly
    what the merged timeline holds.
    """
    e = np.asarray(epochs).ravel()
    m = {s: float((e == s.value).sum()) * EPOCH_MIN for s in Stage.ordered()}
    return StageBreakdown(wake=int(round(m[Stage.WAKE])), light=int(round(m[Stage.LIGHT])),
                          deep=int(round(m[Stage.DEEP])), rem=int(round(m[Stage.REM])))


def waso_from_epochs(epochs: np.ndarray) -> int:
    """Wake minutes strictly between the first and last SLEEP epoch.

    Trailing wake before final wake-up is not WASO, same rule as the row version.
    """
    e = np.asarray(epochs).ravel()
    sleep = np.flatnonzero(np.isin(e, [s.value for s in Stage.ordered()
                                      if s is not Stage.WAKE]))
    if sleep.size == 0:
        return 0
    interior = e[sleep[0]:sleep[-1] + 1]
    return int(round(float((interior == Stage.WAKE.value).sum()) * EPOCH_MIN))


def compute_breakdown(hypnogram: list[StageEstimate]) -> StageBreakdown:
    """time_in_stage_min — a straight row count times the row duration.

    Honest by construction: we cannot resolve stage boundaries finer than a row, so
    the breakdown is quantised to 15 min. Do not interpolate to make it look precise.
    """
    counts = dict.fromkeys(Stage.ordered(), 0)
    for est in hypnogram:
        counts[est.stage] += 1
    return StageBreakdown(
        wake=counts[Stage.WAKE] * ROW_MINUTES,
        light=counts[Stage.LIGHT] * ROW_MINUTES,
        deep=counts[Stage.DEEP] * ROW_MINUTES,
        rem=counts[Stage.REM] * ROW_MINUTES,
    )


def compute_waso(hypnogram: list[StageEstimate]) -> int:
    """Wake After Sleep Onset — wake minutes strictly between first and last sleep row.

    Trailing wake before final wake-up is NOT WASO. The hypnogram passed here starts
    at confirmed onset, so leading wake should already be excluded; this trims the
    trailing run so a slow morning wake-up doesn't inflate WASO.
    """
    sleep_idx = [i for i, e in enumerate(hypnogram) if e.stage is not Stage.WAKE]
    if not sleep_idx:
        return 0
    interior = hypnogram[sleep_idx[0] : sleep_idx[-1] + 1]
    return sum(ROW_MINUTES for e in interior if e.stage is Stage.WAKE)


def count_awakening_bouts(hypnogram: list[StageEstimate]) -> int:
    """How many separate times sleep was interrupted, for the Fragmentation term.

    Delegates the run-counting to the algorithm package and only does the translation
    from `StageEstimate` to a wake flag per row, so the two can never disagree about what
    an awakening is. Same interior as `compute_waso`: leading and trailing wake are not
    awakenings FROM sleep.
    """
    return _count_bouts([e.stage is Stage.WAKE for e in hypnogram])


def score_from_record(record: SessionRecord, *,
                      midpoint_drift_min: float | None = None,
                      history_nights: int | None = None,
                      profile: str = "adult",
                      cfg: "ScoreConfig | None" = None) -> "SleepScoreResult":
    """Adapt a `SessionRecord` into `SleepScoreInputs` and score it.

    This adapter lives here rather than in the algorithm package on purpose: the package
    knows nothing about the pipeline, which is what lets it be read and tested on its own.
    Everything pipeline-shaped stays on this side of the boundary.

    `midpoint_drift_min` and `history_nights` arrive as arguments because neither lives
    on the record — both depend on the user's history, which the session engine can't
    see. Without `history_nights` no score is emitted: the first 14 nights are a warm-up.

    TST is the sum of the non-wake stages, not `duration_min` — `duration_min` is the
    onset-to-wake span and includes WASO.
    """
    if not SLEEP_SCORE_INSTALLED:
        compute_sleep_score()                     # raises, naming the distribution
    b = record.time_in_stage_min
    return compute_sleep_score(
        SleepScoreInputs(
            tst_min=float(b.light + b.deep + b.rem),
            waso_min=float(record.waso_min),
            awakening_bouts=count_awakening_bouts(record.hypnogram),
            midpoint_drift_min=midpoint_drift_min,
            deep_min=float(b.deep),
            rem_min=float(b.rem),
            profile=profile,
            history_nights=history_nights,
        ),
        cfg or ScoreConfig(),
    )


# OFF until the staging models are ready (decided 14 Sep 2026). While it's off the
# session record carries no Sleep Score, because the score's inputs would come from model
# predictions that aren't accurate enough yet. Flip this to True to turn it back on.
SLEEP_SCORE_ENABLED = False


class SessionEngine:
    """Builds the final `SessionRecord` (§7)."""

    def __init__(self, provenance: ProvenanceStamp | None = None,
                 score_config: "ScoreConfig | None" = None,
                 sleep_score_enabled: bool = SLEEP_SCORE_ENABLED) -> None:
        self.provenance = provenance
        # Stays None when the Sleep Score is not installed; the engine never reaches a
        # scoring call in that case, and constructing a default would import-error here —
        # in the constructor of the object every session goes through.
        self.score_config = (score_config if score_config is not None
                             else ScoreConfig() if SLEEP_SCORE_INSTALLED else None)
        self.sleep_score_enabled = sleep_score_enabled

    def finalise(
        self,
        *,
        session_id: str,
        subject_id: str,
        onset_time: datetime,
        wake_time: datetime,
        hypnogram: list[StageEstimate],
        epoch_hypnogram: np.ndarray | None = None,
        waso_band: tuple[str, float, float, str] | None = None,
        outcome: SessionOutcome = SessionOutcome.COMPLETE,
        cycle_phase: str = "unknown",
        midpoint_drift_min: float | None = None,
        history_nights: int | None = None,
        profile: str = "adult",
    ) -> SessionRecord:
        # The 30-second timeline is canonical when it exists; the row count is the
        # fallback for callers that have no wake channel (stubs, and the degraded path).
        if epoch_hypnogram is not None and np.asarray(epoch_hypnogram).size:
            breakdown = breakdown_from_epochs(epoch_hypnogram)
            waso = waso_from_epochs(epoch_hypnogram)
        else:
            breakdown = compute_breakdown(hypnogram)
            waso = compute_waso(hypnogram)
        conf = [e.confidence for e in hypnogram]
        record = SessionRecord(
            session_id=session_id,
            subject_id=subject_id,
            onset_time=onset_time,
            wake_time=wake_time,
            duration_min=int((wake_time - onset_time).total_seconds() // 60),
            time_in_stage_min=breakdown,
            waso_min=waso,
            stage_conf_avg=float(sum(conf) / len(conf)) if conf else 0.0,
            waso_band=None if waso_band is None else waso_band[0],
            waso_band_range=None if waso_band is None else (waso_band[1], waso_band[2]),
            waso_band_confidence=None if waso_band is None else waso_band[3],
            outcome=outcome,
            cycle_phase=cycle_phase,
            provenance=self.provenance,
            hypnogram=hypnogram,
        )
        # §4: naps do not get a Sleep Score. Nor does anything while the score is switched
        # off — see SLEEP_SCORE_ENABLED — or when the algorithm is not installed, in which
        # case the record is still complete and `sleep_score` stays null.
        if (self.sleep_score_enabled and SLEEP_SCORE_INSTALLED
                and outcome is not SessionOutcome.NAP):
            record.sleep_score = self.sleep_score(
                record,
                midpoint_drift_min=midpoint_drift_min,
                history_nights=history_nights,
                profile=profile,
            ).rounded
        return record

    def sleep_score(self, record: SessionRecord, *,
                    midpoint_drift_min: float | None = None,
                    history_nights: int | None = None,
                    profile: str = "adult") -> SleepScoreResult:
        """D-004 — closed. The algorithm lives in `Sleep_Score_Algorithm/`.

        Returns the full result, not just the number: the per-component sub-scores are
        what the §8 explanation UI needs, and `provisional` tells Readiness whether it is
        consuming a complete score or a partial one. `finalise` stores only the rounded
        integer, because the §7 payload field names are frozen.

        `history_nights` and `midpoint_drift_min` arrive from outside because the engine
        has no access to the user's history. Until `history_nights` reaches 14 — or when
        it isn't supplied at all — no score is emitted and `withheld` says why. After
        that, a missing drift still drops Timing and flags the result `provisional`.
        """
        return score_from_record(
            record,
            midpoint_drift_min=midpoint_drift_min,
            history_nights=history_nights,
            profile=profile,
            cfg=self.score_config,
        )
