"""L3 - window detections to the numbers the user actually sees.

Two decisions live here, and they are the product:

  night: {DETECTED, NOT_DETECTED}
  period: "snoring on X of Y nights", or suppressed

Two rules:

1. Count *clusters*, not positive windows (spec 4.2 escalation + 5.1 sparsity).
   After a positive the schedule tightens to 5-minute spacing for 30 minutes, so
   one snoring bout can produce six or more consecutive positives. A raw "2+
   positive windows" rule would therefore be satisfied by a single bout - or by a
   single false positive that escalated into a run. A cluster is a maximal run of
   consecutive *measured* windows that are all positive; one negative measured
   window between two positives means two separate observations of snoring, which
   is what the rule is trying to require.

   Measured on the 32 APSAA nights: counting clusters flags 2 of 6 light snorers,
   counting positive windows flags 3 of 6, and night separation falls from 0.67
   to 0.50. Every heavy snorer is caught either way.

   But "2+ clusters" alone has a hole, and it is the worst-shaped hole available:
   a night containing one long unbroken bout produces exactly one cluster and is
   reported NOT_DETECTED. Simulated with a *perfect* detector, a single 90-minute
   bout yields 18 consecutive positive windows, one cluster, and a clean bill of
   health. So sustained evidence counts too: one cluster spanning
   `min_cluster_span_s` is as good as two separate ones. A false positive cannot
   reach that span, because escalation lapses after 30 minutes of no further
   detections and an isolated false positive is bounded by a single window.

2. The period summary carries its own denominator and suppresses below a minimum
   (spec 5.1, "missing nights are not random"). A bare percentage would quietly
   divide by a denominator that is itself correlated with snoring - ring off,
   travelling, drinking.

There is no coverage floor. Every night that produced any window gets a verdict,
and a night with almost no listening returns NOT_DETECTED like any other. The
consequence is that thin coverage now reads as absence of snoring rather than as
absence of evidence: measured on real nights, a night truncated to 12 windows
returns NOT_DETECTED for 31 % of heavy snorers, and to 8 windows for 62 %.
`n_windows` and `mic_s` are still carried on the verdict so a caller that wants
to weight or reject a thin night can do so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DETECTED = "detected"
NOT_DETECTED = "not_detected"

# ---- defaults ---------------------------------------------------------------------
# Provisional. Gap 6 in the spec ("threshold definition") is closed by the sweep in
# scripts/simulate_sampling.py, not by these numbers.
MIN_CLUSTERS = 2
MIN_CLUSTER_SPAN_S = 600.0  # one cluster sustained this long is sufficient on its own
CLUSTER_GAP_S = 1800.0  # one escalation hold; beyond it, a separate episode
MIN_NIGHTS = 5


def find_clusters(
    t_start: np.ndarray,
    detected: np.ndarray,
    cluster_gap_s: float = CLUSTER_GAP_S,
) -> list[tuple[float, float]]:
    """Distinct snoring episodes observed, as (first_t, last_t) of their positives.

    A cluster breaks on either an intervening negative measured window or a gap
    wider than `cluster_gap_s` (a positive can only have extended an escalation
    that has since lapsed).
    """
    clusters: list[list[float]] = []
    in_cluster = False
    prev_t = -np.inf
    for t, d in zip(t_start.tolist(), detected.tolist()):
        if not d:
            in_cluster = False
        else:
            if not in_cluster or (t - prev_t) > cluster_gap_s:
                clusters.append([t, t])
            else:
                clusters[-1][1] = t
            in_cluster = True
        prev_t = t
    return [(a, b) for a, b in clusters]


def cluster_positives(
    t_start: np.ndarray,
    detected: np.ndarray,
    cluster_gap_s: float = CLUSTER_GAP_S,
) -> int:
    """Number of distinct snoring episodes observed."""
    return len(find_clusters(t_start, detected, cluster_gap_s))


@dataclass
class NightVerdict:
    state: str
    n_windows: int
    n_positive: int
    n_clusters: int
    max_cluster_span_s: float
    mic_s: float

    @property
    def is_flagged(self) -> bool:
        return self.state == DETECTED


def decide_night(
    t_start: np.ndarray,
    detected: np.ndarray,
    mic_s: float,
    min_clusters: int = MIN_CLUSTERS,
    cluster_gap_s: float = CLUSTER_GAP_S,
    min_cluster_span_s: float = MIN_CLUSTER_SPAN_S,
) -> NightVerdict:
    """DETECTED if snoring was observed twice, or once for long enough.

    Two positives must be *separated* by a negative measured window to count as
    two observations; two consecutive positives are one episode.

    `mic_s` no longer gates the verdict; it is carried on the result so a caller
    can judge how much listening the verdict rests on.
    """
    n_windows = int(t_start.size)
    n_positive = int(detected.sum())
    clusters = find_clusters(t_start, detected, cluster_gap_s)
    span = max((b - a for a, b in clusters), default=0.0)

    if len(clusters) >= min_clusters or span >= min_cluster_span_s:
        state = DETECTED
    else:
        state = NOT_DETECTED

    return NightVerdict(state, n_windows, n_positive, len(clusters), span, mic_s)


@dataclass
class PeriodSummary:
    snore_nights: int
    n_nights: int
    suppressed: bool

    def headline(self) -> str:
        if self.suppressed:
            return "Not enough nights yet."
        return f"Snoring on {self.snore_nights} of {self.n_nights} nights."


def summarise_period(
    verdicts: list[NightVerdict],
    min_nights: int = MIN_NIGHTS,
) -> PeriodSummary:
    return PeriodSummary(
        snore_nights=sum(v.is_flagged for v in verdicts),
        n_nights=len(verdicts),
        suppressed=len(verdicts) < min_nights,
    )
