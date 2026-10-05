"""One hypnogram from two channels (D-015).

THE PROBLEM THIS SOLVES
Two models were producing an answer to the same question. The 15-minute stage model
emits a Wake class, and post-onset Wake is WASO by definition; the accelerometer channel
independently estimates WASO at 30 seconds. Shipping both means the timeline and the
reported figure can disagree, and a user who notices has no way to tell which is wrong.

THE RESOLUTION IS NOT "PICK ONE" — the two channels are good at different things, and
the measurements say so plainly:

  wake detection, post-onset   15-min model recall 0.362 | accelerometer 0.556
                               (at matched ~0.50 precision)
  WASO on DREAMT               15-min model MAE 42.4 min, WORSE than the 41.8 of
                               simply predicting the population mean
  Deep vs REM                  only the cardiac channel can do this at all; motion
                               carries no information that separates them

So each channel owns what it is good at. The accelerometer decides SLEEP OR WAKE at
30 seconds, where wake bouts actually live. The cardiac model decides WHICH SLEEP STAGE
at 15 minutes, which is the rate its sensor arrives at. There is then exactly one wake
decision in the system and every derived metric is computed from one timeline, so they
cannot contradict each other.

WHY THE STAGE MODEL'S WAKE PROBABILITY IS DISCARDED, NOT BLENDED
Blending would reintroduce the disagreement in a softer form and would need a weighting
nobody can justify from the data. The stage model's Wake mass is removed and the
remaining three renormalised, which is exactly what a model trained with `--no-wake`
emits natively — the merge rule and the 3-class model are the same design decision seen
from two sides.
"""

from __future__ import annotations

import numpy as np

from ..types import Stage

__all__ = ["ROW_EPOCHS", "DEFAULT_WAKE_THRESHOLD", "merge_hypnogram",
           "prevalence_matched_threshold"]

ROW_EPOCHS = 30                    # 15 min / 30 s

#: Motion P(wake) above which an epoch is called wake.
#:
#: THIS NUMBER IS NOT PORTABLE. It is a cut on one particular scorer's probability
#: scale, and it moves whenever that scale moves. The previous value, 0.70, came from a
#: class-BALANCED scorer whose mean P(wake) was 0.343 against a true rate of 0.150.
#: The current scorer drops the balancing and is calibrated — mean 0.150 against the
#: same 0.150 — so its whole distribution sits about 2.3x lower. Measured over 290,122
#: epochs, 0.70 on the old scale and 0.30 on the new one both select ~15% of epochs,
#: i.e. they are the SAME operating point written two ways.
#:
#: 0.40 sits slightly above the prevalence-matching 0.30 on purpose: mildly conservative,
#: favouring precision, so the merge does not invent wake. It was chosen by sweeping
#: WASO MAE over 345 nights.
#:
#: Re-derive it, do not inherit it, whenever the epoch scorer changes — its model class,
#: its class weighting, or the wake prevalence of its training population.
#: `prevalence_matched_threshold` does this automatically.
DEFAULT_WAKE_THRESHOLD = 0.40


def prevalence_matched_threshold(pwake: np.ndarray, target_rate: float) -> float:
    """The cut that calls `target_rate` of epochs wake, whatever the scorer's scale.

    A self-calibrating alternative to a hard-coded constant. Pass the training
    population's post-onset wake fraction as `target_rate`; the returned threshold
    reproduces it on this scorer, so swapping the scorer cannot silently change how
    much wake the merge produces.
    """
    p = np.asarray(pwake, dtype=np.float64).ravel()
    p = p[np.isfinite(p)]
    if p.size == 0:
        return DEFAULT_WAKE_THRESHOLD
    return float(np.quantile(p, 1.0 - np.clip(target_rate, 0.0, 1.0)))


def merge_hypnogram(row_stage_probs: np.ndarray, row_blocks: np.ndarray,
                    pwake: np.ndarray, *, epoch_offset: int = 0,
                    threshold: float = DEFAULT_WAKE_THRESHOLD) -> np.ndarray:
    """Cardiac row probabilities + motion P(wake) -> one 30-second hypnogram.

    Args:
        row_stage_probs: (R, 4) per-15-min-row probabilities in `Stage.ordered()` order.
        row_blocks:      (R,) the `block` index of each row, counted from RECORDING
                         start — the same convention `03_unify.py` emits.
        pwake:           (E,) motion P(wake) per 30-second epoch, indexed from the first
                         SCOREABLE epoch.
        epoch_offset:    epochs dropped as unscoreable before `pwake` begins. DREAMT's
                         leading `P` block is 197-332 epochs; ignoring this offset puts
                         every row 1.5-2.5 hours from the epochs it describes, which is
                         a silent failure that has already cost this project time twice.
        threshold:       motion P(wake) cut. See DEFAULT_WAKE_THRESHOLD.

    Returns:
        (E,) int8 stage codes.

    Rows are carried forward across gaps, which is what a device does when a window
    produced no usable cardiac data; a leading gap defaults to Light rather than Wake,
    because the motion channel — not this fallback — is what decides wake.
    """
    pwake = np.asarray(pwake, dtype=np.float64).ravel()
    n_ep = pwake.size
    probs = np.asarray(row_stage_probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != Stage.n_classes():
        raise ValueError(f"expected (R, {Stage.n_classes()}) row probabilities, "
                         f"got {probs.shape}")
    if probs.shape[0] != len(row_blocks):
        raise ValueError(f"{probs.shape[0]} rows but {len(row_blocks)} blocks")

    # Sleep-stage decision per row: the cardiac model's Wake mass is dropped and the
    # remaining three renormalised. A row where all three are zero falls back to Light,
    # the majority stage, rather than to an arbitrary index.
    sleep = probs[:, 1:]
    total = sleep.sum(axis=1, keepdims=True)
    stage_of_row = np.where(total.ravel() > 0, sleep.argmax(axis=1) + 1, Stage.LIGHT.value)

    order = np.argsort(np.asarray(row_blocks))
    block_to_stage = dict(zip(np.asarray(row_blocks)[order].tolist(),
                              stage_of_row[order].tolist()))

    hyp = np.full(n_ep, Stage.LIGHT.value, dtype=np.int8)
    last = Stage.LIGHT.value
    n_blocks = int(np.ceil((n_ep + epoch_offset) / ROW_EPOCHS))
    for b in range(n_blocks):
        last = block_to_stage.get(b, last)
        lo = max(b * ROW_EPOCHS - epoch_offset, 0)
        hi = min((b + 1) * ROW_EPOCHS - epoch_offset, n_ep)
        if hi > lo:
            hyp[lo:hi] = last

    # The accelerometer has the final say on wake. One-directional by construction:
    # there is no path here that turns a motion-detected wake epoch back into sleep.
    hyp[pwake >= threshold] = Stage.WAKE.value
    return hyp
