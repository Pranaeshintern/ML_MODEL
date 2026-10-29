"""Feature extraction and dataset adapters.

    hrv.py       NN intervals -> HRV features        (dataset-independent)
    motion.py    tri-axial accelerometry -> features (dataset-independent)
    dreamt.py    DREAMT CSVs   -> 15-min rows        (adapter)
    bidsleep.py  BIDSleep dirs -> 15-min rows        (adapter)

Only the two adapters know about files. Everything else takes arrays, which is what
lets the same feature code serve both corpora and, later, the ring.
"""

from .hrv import HRV_FEATURE_NAMES, NNSeries, beat_coverage, clean_nn, hrv_features
from .motion import MOTION_FEATURE_NAMES, motion_features

__all__ = [
    "NNSeries",
    "clean_nn",
    "hrv_features",
    "beat_coverage",
    "HRV_FEATURE_NAMES",
    "motion_features",
    "MOTION_FEATURE_NAMES",
]
