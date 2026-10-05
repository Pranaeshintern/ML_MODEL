"""Core domain enums and label mappings.

Doc refs: §2 (why 4-class), §4 (state machine states).
"""

from __future__ import annotations

from enum import Enum, IntEnum

__all__ = [
    "Stage",
    "PSGStage",
    "SleepState",
    "SessionOutcome",
    "psg_to_stage",
    "PSG_TO_STAGE",
]


class Stage(IntEnum):
    """The locked 4-class scheme (§2).

    Integer values are the softmax column order and MUST NOT be reordered — the HMM
    transition matrix in `config/default.yaml` is indexed by these values.

    N1 is deliberately folded into LIGHT: N1 episodes are 1-5 min and a 15-min row
    averages them away. See §2 for the full argument.
    """

    WAKE = 0
    LIGHT = 1
    DEEP = 2
    REM = 3

    @property
    def label(self) -> str:
        return {0: "wake", 1: "light", 2: "deep", 3: "rem"}[self.value]

    @classmethod
    def n_classes(cls) -> int:
        return 4

    @classmethod
    def ordered(cls) -> tuple[Stage, ...]:
        """Canonical order — matches softmax columns and transition-matrix axes."""
        return (cls.WAKE, cls.LIGHT, cls.DEEP, cls.REM)


class PSGStage(str, Enum):
    """Raw per-sample labels as they appear in DREAMT `whole_df.Sleep_Stage`.

    PREP ("P") is the pre-sleep preparation period — explicitly-labelled wake before
    lights-out. It is 13-30% of each recording and is the primary source of Layer 1
    negatives and of the PSG-derived onset boundary (see §5 ground truth).

    The Phase-0 audit must confirm this is the complete value set across all 80
    subjects. Unmapped values raise rather than silently defaulting — see
    `psg_to_stage`.
    """

    PREP = "P"
    WAKE = "W"
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"
    REM = "R"
    #: Unscoreable epoch — NOT a stage. Present in some subjects but not all, which
    #: is exactly why `psg_to_stage` raises rather than defaulting: folding these
    #: into Wake would have silently inflated the Wake class across the dataset.
    MISSING = "Missing"

    @property
    def is_pre_sleep(self) -> bool:
        return self is PSGStage.PREP

    @property
    def is_scoreable(self) -> bool:
        return self is not PSGStage.MISSING

    @property
    def is_sleep(self) -> bool:
        return self in (PSGStage.N1, PSGStage.N2, PSGStage.N3, PSGStage.REM)


#: PSG label -> 4-class collapse (§2). PREP maps to WAKE for staging purposes, but
#: callers needing the pre-sleep distinction must use `PSGStage.is_pre_sleep`, since
#: that information is destroyed here.
PSG_TO_STAGE: dict[PSGStage, Stage] = {
    PSGStage.PREP: Stage.WAKE,
    PSGStage.WAKE: Stage.WAKE,
    PSGStage.N1: Stage.LIGHT,
    PSGStage.N2: Stage.LIGHT,
    PSGStage.N3: Stage.DEEP,
    PSGStage.REM: Stage.REM,
}


#: Codes that carry no stage. Rows built from these contribute to neither the label
#: distribution nor `n_valid_epochs`.
UNSCOREABLE: frozenset[str] = frozenset({"", "nan", "Missing"})


def is_scoreable(raw: str) -> bool:
    return raw not in UNSCOREABLE


class UnknownPSGLabelError(ValueError):
    """Raised when the dataset contains a stage label we have not mapped."""


def psg_to_stage(raw: str) -> Stage:
    """Collapse a raw PSG label to the 4-class scheme.

    Raises `UnknownPSGLabelError` on anything unmapped. This is deliberate: a silent
    default would hide dataset surprises (e.g. a "Missing" sentinel) behind a
    plausible-looking class distribution.
    """
    try:
        return PSG_TO_STAGE[PSGStage(raw)]
    except (ValueError, KeyError) as exc:
        raise UnknownPSGLabelError(
            f"unmapped Sleep_Stage value {raw!r}; "
            f"known values are {[s.value for s in PSGStage]}"
        ) from exc


class SleepState(Enum):
    """Sleep-state machine states (§4).

    The state decides WHICH model runs:
      AWAKE / CANDIDATE_ONSET -> Layer 1 only
      ASLEEP                  -> Layer 2 + Layer 3 (Layer 1 stops permanently, §4)
      WAKE_DET                -> terminal for the session; triggers finalisation
    """

    AWAKE = "awake"
    CANDIDATE_ONSET = "candidate_onset"
    ASLEEP = "asleep"
    WAKE_DET = "wake_det"

    @property
    def runs_onset_detector(self) -> bool:
        return self in (SleepState.AWAKE, SleepState.CANDIDATE_ONSET)

    @property
    def runs_stage_classifier(self) -> bool:
        return self is SleepState.ASLEEP


class SessionOutcome(str, Enum):
    """How a session ended — decides whether a Sleep Score is emitted at all.

    NAP: woke <90 min post-onset with HR/temp not at sleep baseline (§4 nap
    protection). No Sleep Score is produced.
    TIMEOUT_FALLBACK: onset never confirmed by T_bedtime + 6 h, so the rule-based
    fallback fired (§4 escape hatch). Shift-worker safe, but flag it downstream.
    """

    COMPLETE = "complete"
    NAP = "nap"
    TIMEOUT_FALLBACK = "timeout_fallback"
    ABANDONED = "abandoned"
    #: The accelerometer channel was absent or too sparse to decide wake. Sleep/wake
    #: comes from the accelerometer alone (D-015), so without it there is no wake
    #: decision to make: the session cannot end itself and WASO is not measurable.
    #: Flag it downstream rather than presenting the numbers as complete.
    DEGRADED_NO_MOTION = "degraded_no_motion"
