"""Models: architectures, checkpoint loading, calibration, and baselines.

    architectures.py  the networks themselves — defined HERE, not inside training
                      scripts, so a saved checkpoint can be reconstructed
    loader.py         load a checkpoint together with the feature order and
                      normalisation constants it is meaningless without
    calibration.py    isotonic probability calibration (§6, §8)
    layer1.py         §5 rule-based onset detector (the escape-hatch fallback)
    layer2.py         protocol-conforming stage-classifier declarations
    stubs.py          null components used to test the runtime without a model
"""

from .architectures import (
    OnsetTCN,
    RowEncoder,
    StageBiLSTM,
    StageMLP,
    StageMultiHeadTCN,
    StageTCN,
    build_stage_model,
)
from .calibration import IsotonicCalibrator
from .layer1 import RuleOnsetDetector, RuleThresholds, TreeOnsetDetector
from .layer2 import RollingWindowSpec, TCNStageClassifier, XGBStageClassifier
from .loader import (
    LoadedModel,
    LoadedTreeModel,
    load_onset_model,
    load_stage_model,
    predict_onset_proba,
    predict_stages,
)
from .stubs import ConstantStageClassifier, NeverOnsetDetector, ScriptedOnsetDetector

__all__ = [
    # architectures
    "RowEncoder",
    "StageTCN",
    "StageBiLSTM",
    "StageMLP",
    "StageMultiHeadTCN",
    "OnsetTCN",
    "build_stage_model",
    # inference
    "LoadedModel",
    "LoadedTreeModel",
    "load_stage_model",
    "load_onset_model",
    "predict_stages",
    "predict_onset_proba",
    # supporting
    "IsotonicCalibrator",
    "RuleOnsetDetector",
    "RuleThresholds",
    "TreeOnsetDetector",
    "XGBStageClassifier",
    "TCNStageClassifier",
    "RollingWindowSpec",
    "NeverOnsetDetector",
    "ScriptedOnsetDetector",
    "ConstantStageClassifier",
]
