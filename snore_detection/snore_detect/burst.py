"""Burst duration - mean width of the above-threshold events in a window.

One of the 55 features. Extracted from the original screener module, whose
hand-built detector was replaced by the trained model (FAIL-SNORE-001) and
is not shipped.
"""

from __future__ import annotations

import numpy as np

from snore_detect.periodicity import FS_ENV, envelope_periodicity


_EPS = 1e-12


def burst_duration_s(env: np.ndarray, fs: float = FS_ENV) -> np.ndarray:
    """Mean duration of the above-threshold events in each window.

    The threshold is per-window, set halfway between that window's floor and its
    peak, so it is indifferent to absolute level - which matters because hand
    position moves the level by more than the signal does.

    Caveat: "peak" is the 90th percentile, which sits inside a burst only while the
    window is reasonably occupied. For very sparse events - a handful of narrow
    bursts in a long window - the 90th percentile falls in the gap instead, the
    threshold collapses onto the noise floor, and the measurement degrades. Real
    envelopes carry a room floor that keeps this well-behaved (a tick at 4 % duty
    still measures 0.11 s correctly), but a synthetic envelope with a true zero
    floor will not.
    """
    x = np.atleast_2d(np.asarray(env, dtype=np.float64))
    floor = np.percentile(x, 10, axis=1, keepdims=True)
    peak = np.percentile(x, 90, axis=1, keepdims=True)
    above = x > (floor + 0.5 * (peak - floor))

    pad = np.zeros((x.shape[0], 1), dtype=bool)
    rises = np.diff(np.concatenate([pad, above], axis=1).astype(np.int8), axis=1) == 1
    n_bursts = rises.sum(axis=1)
    time_above = above.sum(axis=1) / fs
    return np.where(n_bursts > 0, time_above / np.maximum(n_bursts, 1), 0.0)
