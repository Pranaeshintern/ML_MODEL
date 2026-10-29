"""HRV from an NN-interval series.

Dataset-independent by design: intervals in, features out. The caller supplies them
from whatever source — DREAMT's forward-filled IBI column, a ring's firmware beat
detector, or peak detection on raw PPG. Nothing here knows about files.

Validated against DREAMT S003 (see the probe notes in `clean_nn`). Artifact
correction is not optional: uncorrected RMSSD is an artifact counter, not a
physiological measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NNSeries", "clean_nn", "hrv_features", "HRV_FEATURE_NAMES"]

#: Emitted in feature-dictionary order.
HRV_FEATURE_NAMES = (
    "rmssd",
    "sdnn",
    "pnn50",
    "poincare_sd1",
    "poincare_sd2",
    "sd1_sd2_ratio",
    "hrv_sample_entropy",
)

NN_MIN_S = 0.30      # 200 bpm
NN_MAX_S = 2.00      # 30 bpm
MALIK_FRACTION = 0.20
MIN_BEATS = 40       # below this every feature is NaN rather than noise


@dataclass(frozen=True, slots=True)
class NNSeries:
    """Cleaned NN intervals plus the mask of which survived correction.

    Both are needed downstream: successive differences may only be taken between
    beats that were ADJACENT IN THE ORIGINAL SERIES and both valid. Bridging a
    rejected beat manufactures exactly the large difference the rejection removed —
    the single easiest way to get HRV badly wrong.
    """

    intervals: np.ndarray  # seconds, after splitting missed beats
    valid: np.ndarray      # bool, same length

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())

    @property
    def artifact_rate(self) -> float:
        if self.intervals.size == 0:
            return 1.0
        return 1.0 - self.n_valid / self.intervals.size

    def successive_diffs(self) -> np.ndarray:
        """|NN[i] - NN[i-1]| in ms, over originally-adjacent valid pairs only."""
        pair_ok = self.valid[1:] & self.valid[:-1]
        return np.abs(np.diff(self.intervals))[pair_ok] * 1000.0

    def valid_intervals(self) -> np.ndarray:
        return self.intervals[self.valid]


def clean_nn(
    intervals: np.ndarray,
    *,
    nn_min: float = NN_MIN_S,
    nn_max: float = NN_MAX_S,
    malik: float = MALIK_FRACTION,
    split_missed_beats: bool = True,
) -> NNSeries:
    """Artifact-correct a raw NN series.

    Three stages, in order:

    1. **Split missed beats.** Wrist PPG drops beats; a dropped beat merges two
       intervals into one of ~2x (or 3x) the local median. Splitting restores the
       cadence. Done first, because an unsplit doubled interval would be rejected by
       the range filter and take a real beat with it.
    2. **Physiological range.** 0.3-2.0 s, i.e. 30-200 bpm.
    3. **Malik rule.** Reject a beat differing from its predecessor by >20%.

    DREAMT S003 probe: ~1% rejection during sleep, and HR recovered from the cleaned
    series matched the device's independent HR channel to within 1-2 bpm — which is
    what makes the reconstruction trustworthy.
    """
    nn = np.asarray(intervals, dtype=np.float64).ravel()
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return NNSeries(np.empty(0), np.empty(0, dtype=bool))

    if split_missed_beats:
        nn = _split_missed(nn)

    valid = (nn >= nn_min) & (nn <= nn_max)
    for i in range(1, nn.size):
        if valid[i] and valid[i - 1] and abs(nn[i] - nn[i - 1]) > malik * nn[i - 1]:
            valid[i] = False
    return NNSeries(nn, valid)


def _split_missed(nn: np.ndarray) -> np.ndarray:
    """Expand intervals that are a near-integer multiple of the local median."""
    med = float(np.median(nn))
    if not np.isfinite(med) or med <= 0:
        return nn
    out: list[float] = []
    for v in nn:
        k = int(round(v / med))
        # Only split when the result genuinely lands near the median — otherwise a
        # real tachycardic beat would be shredded into fragments.
        if 2 <= k <= 3 and abs(v / k - med) < 0.35 * med:
            out.extend([v / k] * k)
        else:
            out.append(float(v))
    return np.asarray(out, dtype=np.float64)


def hrv_features(nn: NNSeries, *, min_beats: int = MIN_BEATS) -> dict[str, float]:
    """Time-domain and nonlinear HRV. Frequency-domain is deliberately absent.

    At a 90 s burst, LF (0.04 Hz = 25 s period) gives ~3.6 cycles — too few for a
    trustworthy estimate — and HF would need beat-level data the ring can supply but
    at a window length that makes the estimate fragile. A plausible-looking number
    here would be worse than no number, so those features are not in the dictionary.
    """
    nan = {name: float("nan") for name in HRV_FEATURE_NAMES}
    if nn.n_valid < min_beats:
        return nan

    vals = nn.valid_intervals()
    diffs = nn.successive_diffs()
    if diffs.size < 10:
        return nan

    rmssd = float(np.sqrt(np.mean(diffs**2)))
    sdnn = float(np.std(vals, ddof=1) * 1000.0)
    pnn50 = float(np.mean(diffs > 50.0))

    # Poincaré. SD1 is short-term (beat-to-beat) scatter, SD2 long-term. SD1 is
    # RMSSD/sqrt(2) by construction; kept separately because SD2 and the ratio are
    # not derivable from RMSSD alone.
    sd1 = float(rmssd / np.sqrt(2.0))
    sd2_sq = 2.0 * sdnn**2 - 0.5 * rmssd**2
    sd2 = float(np.sqrt(sd2_sq)) if sd2_sq > 0 else float("nan")
    ratio = float(sd1 / sd2) if sd2 and np.isfinite(sd2) and sd2 > 0 else float("nan")

    return {
        "rmssd": rmssd,
        "sdnn": sdnn,
        "pnn50": pnn50,
        "poincare_sd1": sd1,
        "poincare_sd2": sd2,
        "sd1_sd2_ratio": ratio,
        "hrv_sample_entropy": sample_entropy(vals),
    }


def sample_entropy(x: np.ndarray, *, m: int = 2, r_factor: float = 0.2) -> float:
    """SampEn(m, r) — §9's "chaos index", the named Light<->REM discriminator.

    Negative log conditional probability that two sequences similar for m points stay
    similar at m+1, with self-matches excluded. O(N^2), but N is ~100 beats so the
    cost is irrelevant.

    Returns NaN rather than inf when no template matches: with ~100 beats that means
    the estimate is undefined, and inf would poison every downstream aggregate.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < m + 20:
        return float("nan")

    r = r_factor * float(np.std(x, ddof=1))
    if not np.isfinite(r) or r <= 0:
        return float("nan")

    def _count(dim: int) -> int:
        # templates[i] = x[i : i+dim]; Chebyshev distance between all pairs
        templates = np.lib.stride_tricks.sliding_window_view(x, dim)[: n - m]
        dist = np.abs(templates[:, None, :] - templates[None, :, :]).max(axis=2)
        np.fill_diagonal(dist, np.inf)  # exclude self-matches
        return int((dist <= r).sum())

    b, a = _count(m), _count(m + 1)
    if b == 0 or a == 0:
        return float("nan")
    return float(-np.log(a / b))


def beat_coverage(nn: NNSeries, burst_seconds: float) -> float:
    """`sqi_ppg`: observed beats vs beats expected from the mean HR over the burst.

    Catches the failure the artifact rate misses — a burst where the sensor produced
    few beats but the ones it produced were internally consistent, so nothing gets
    rejected and the data still isn't there. The DREAMT probe showed exactly this in
    pre-sleep bursts, where movement suppresses PPG.
    """
    if nn.n_valid == 0 or burst_seconds <= 0:
        return 0.0
    expected = burst_seconds / float(np.mean(nn.valid_intervals()))
    return float(min(1.0, nn.n_valid / expected)) if expected > 0 else 0.0
