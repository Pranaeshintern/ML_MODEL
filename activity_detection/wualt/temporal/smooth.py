"""L3 — temporal decoding over window posteriors.

Why this layer is load-bearing rather than cosmetic:

36.8 % of CAPTURE-24 windows carrying an *active* label have a motionless wrist
(`phase3/label_noise_diagnostic.json`). Some of that is annotation coarseness, but
much of it is real — a person walking with a hand in a pocket, or resting a hand on
a handlebar, genuinely produces a still wrist. No per-window classifier can recover
those from accelerometry, because the evidence is not in the window.

It *is* in the neighbourhood. A still 30 s surrounded by ten minutes of walking is
walking. That is a sequence-decoding problem, and it is why the previous phase's
attempt to fix the same thing with per-window hand rules moved accuracy by +0.002:
the rules had no more information than the classifier did.

Decoding is Viterbi over a first-order chain with a tunable self-transition boost.
The boost is the practical stand-in for an explicit-duration (semi-Markov) dwell
model: it is one parameter, it is fitted on development subjects only, and it makes
the minimum-bout behaviour directly controllable.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def fit_transitions(
    y: np.ndarray,
    session_id: np.ndarray,
    n_classes: int,
    smoothing: float = 1.0,
) -> np.ndarray:
    """Estimate a row-stochastic transition matrix from labelled window sequences.

    Sessions are treated independently — a transition is only counted between two
    windows that are genuinely adjacent in the same recording.
    """
    counts = np.full((n_classes, n_classes), smoothing, dtype=np.float64)
    order = np.argsort(session_id, kind="stable")
    y_s, sid_s = y[order], session_id[order]
    same = sid_s[:-1] == sid_s[1:]
    np.add.at(counts, (y_s[:-1][same], y_s[1:][same]), 1.0)
    # A class that never occurs leaves an all-zero row; fall back to uniform so the
    # matrix stays row-stochastic instead of producing NaN.
    row = counts.sum(axis=1, keepdims=True)
    counts = np.where(row > 0, counts, 1.0 / n_classes)
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), _EPS)


def fit_prior(y: np.ndarray, n_classes: int, smoothing: float = 1.0) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64) + smoothing
    return counts / counts.sum()


def viterbi(
    log_emission: np.ndarray,
    log_trans: np.ndarray,
    log_prior: np.ndarray,
) -> np.ndarray:
    """Standard Viterbi. log_emission is [T, K]."""
    T, K = log_emission.shape
    if T == 0:
        return np.empty(0, dtype=np.int8)
    delta = log_prior + log_emission[0]
    psi = np.zeros((T, K), dtype=np.int32)
    for t in range(1, T):
        scores = delta[:, None] + log_trans  # [K_prev, K_next]
        psi[t] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + log_emission[t]

    path = np.empty(T, dtype=np.int8)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path


def decode(
    proba: np.ndarray,
    session_id: np.ndarray,
    trans: np.ndarray,
    prior: np.ndarray,
    stickiness: float = 1.0,
    prior_correction: float = 0.0,
) -> np.ndarray:
    """Viterbi-decode each session independently.

    stickiness        multiplies the diagonal of the transition matrix before
                      renormalising. 1.0 = use the fitted matrix as-is; larger
                      values lengthen bouts and suppress isolated flips.
    prior_correction  subtracts `prior_correction * log(train_prior)` from the
                      emission, undoing part of the training class imbalance. The
                      classifier's posterior already contains the training prior;
                      leaving it in double-counts it once the transition matrix
                      supplies its own.
    """
    K = proba.shape[1]
    T = trans.copy()
    if stickiness != 1.0:
        T = T * (np.eye(K) * (stickiness - 1.0) + 1.0)
        T = T / T.sum(axis=1, keepdims=True)

    log_trans = np.log(T + _EPS)
    log_prior = np.log(prior + _EPS)
    log_em = np.log(np.clip(proba, _EPS, 1.0))
    if prior_correction:
        log_em = log_em - prior_correction * log_prior[None, :]

    out = np.empty(len(proba), dtype=np.int8)
    order = np.argsort(session_id, kind="stable")
    sid_s = session_id[order]
    bounds = np.flatnonzero(sid_s[:-1] != sid_s[1:]) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(sid_s)]):
        idx = order[lo:hi]
        out[idx] = viterbi(log_em[idx], log_trans, log_prior)
    return out


def min_dwell(
    labels: np.ndarray,
    session_id: np.ndarray,
    min_windows: int = 2,
) -> np.ndarray:
    """Absorb runs shorter than `min_windows` into the surrounding label.

    A product guard as much as an accuracy one: a single 30 s window should never
    become a reported workout.
    """
    out = labels.copy()
    order = np.argsort(session_id, kind="stable")
    sid_s = session_id[order]
    bounds = np.flatnonzero(sid_s[:-1] != sid_s[1:]) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(sid_s)]):
        idx = order[lo:hi]
        seq = out[idx]
        if len(seq) < 3:
            continue
        starts = np.r_[0, np.flatnonzero(seq[:-1] != seq[1:]) + 1, len(seq)]
        for a, b in zip(starts[:-1], starts[1:]):
            if b - a >= min_windows:
                continue
            left = seq[a - 1] if a > 0 else None
            right = seq[b] if b < len(seq) else None
            if left is not None and (right is None or left == right):
                seq[a:b] = left
            elif right is not None:
                seq[a:b] = right
        out[idx] = seq
    return out


def bouts(labels: np.ndarray, t_start: np.ndarray, window_s: float = 30.0) -> list[dict]:
    """Collapse a decoded label sequence into bouts (the timeline the app shows).

    A bout spans from the start of its first window to the *end* of its last one, so
    the reported duration is the real elapsed time. Using the hop instead would
    undercount every bout by one window length.
    """
    if len(labels) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(labels[:-1] != labels[1:]) + 1, len(labels)]
    out = []
    for a, b in zip(starts[:-1], starts[1:]):
        t_end = float(t_start[b - 1] + window_s)
        out.append(
            {
                "label": int(labels[a]),
                "t_start_s": float(t_start[a]),
                "t_end_s": t_end,
                "n_windows": int(b - a),
                "duration_s": t_end - float(t_start[a]),
            }
        )
    return out
