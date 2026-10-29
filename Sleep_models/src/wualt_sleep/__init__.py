"""WUALT sleep onset detection and 4-class sleep-stage classification.

Layered per WUALT_Sleep_Pipeline_v2_1:
    Layer 1  onset detection       (§5)
    Layer 2  4-class staging       (§6)
    Layer 3  HMM Viterbi smoothing (§7)
    plus the sleep-state machine (§4) and session engine (§7)
"""

from __future__ import annotations

from .config import Config, load_config
from .errors import WualtSleepError
from .events import OnsetEvent, SessionRecord, StageEstimate
from .featuredict import FeatureDictionary, load_feature_dictionary
from .schema import Epoch, LabelledRow, OnsetLabel, Row, StageLabel
from .types import PSGStage, SessionOutcome, SleepState, Stage

__version__ = "0.1.0"

__all__ = [
    "Config",
    "load_config",
    "FeatureDictionary",
    "load_feature_dictionary",
    "Stage",
    "PSGStage",
    "SleepState",
    "SessionOutcome",
    "Row",
    "Epoch",
    "StageLabel",
    "OnsetLabel",
    "LabelledRow",
    "OnsetEvent",
    "StageEstimate",
    "SessionRecord",
    "WualtSleepError",
    "__version__",
]
