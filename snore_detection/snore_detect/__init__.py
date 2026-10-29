"""Snore detection - M14 / Night Sounds.

    from snore_detect import SnoreDetector

    det = SnoreDetector()
    r = det.score_window(audio, fs)      # one 10 s window -> score, snoring
    v = det.decide_night(times, flags)   # a night of results -> one verdict
"""

from snore_detect.detector import SnoreDetector  # noqa: F401
from snore_detect.aggregate import DETECTED, NOT_DETECTED  # noqa: F401

__all__ = ["SnoreDetector", "DETECTED", "NOT_DETECTED"]
__version__ = "0.1.0"
