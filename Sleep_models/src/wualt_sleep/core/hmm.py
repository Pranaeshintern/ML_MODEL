"""§7 — Layer 3. HMM Viterbi decoding + forward-backward posteriors.

Not an ML model in the MVP: the transition matrix is fixed from PSG literature and
the decode is deterministic. Production fits A from pilot data via Baum-Welch.

Everything runs in LOG SPACE. A 30-row night in linear space underflows for
low-probability paths, and the failure is silent — you get a plausible hypnogram
computed from denormalised floats.

Cost is negligible: O(N*K^2) = 30 rows x 16 = 480 ops per night.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import HMMConfig
from ..errors import ConfigError
from ..types import Stage

__all__ = ["ViterbiResult", "ViterbiSmoother", "PassthroughSmoother", "stationary_distribution"]

_EPS = 1e-300
_K = Stage.n_classes()


def _log(x: np.ndarray) -> np.ndarray:
    """log with a floor, so a zero transition becomes very-negative, not -inf.

    §7's matrix has genuine zeros (Wake->Deep is 0.00). Keeping them finite means a
    path through a forbidden transition loses by a huge margin instead of producing
    NaN when it meets a -inf elsewhere in the arithmetic.
    """
    return np.log(np.clip(np.asarray(x, dtype=np.float64), _EPS, None))


def _logsumexp(a: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    out = np.log(np.exp(a - amax).sum(axis=axis, keepdims=True)) + amax
    return out if keepdims else np.squeeze(out, axis=axis)


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Long-run stage occupancy implied by the transition matrix.

    Used as the default class prior for the hybrid correction (D-012) when no
    empirical prior is supplied. It is the left eigenvector of A for eigenvalue 1.
    """
    values, vectors = np.linalg.eig(np.asarray(transition, dtype=np.float64).T)
    idx = int(np.argmin(np.abs(values - 1.0)))
    vec = np.real(vectors[:, idx])
    vec = np.abs(vec)
    total = vec.sum()
    if total <= 0:
        raise ConfigError("transition matrix has no valid stationary distribution")
    return vec / total


@dataclass(frozen=True, slots=True)
class ViterbiResult:
    path: list[Stage]
    posterior: np.ndarray  # (T, 4) forward-backward posterior, rows sum to 1
    log_likelihood: float

    def confidence(self) -> np.ndarray:
        """Per-row confidence: posterior mass on the decoded stage.

        §7 prefers this over the raw Layer 2 softmax, because the softmax is
        row-local and ignores the sequence constraint that produced the decision.
        """
        idx = np.array([s.value for s in self.path], dtype=np.int64)
        return self.posterior[np.arange(len(idx)), idx]

    def as_array(self) -> np.ndarray:
        return np.array([s.value for s in self.path], dtype=np.int64)


class ViterbiSmoother:
    """Layer 3. Fixed transition matrix in the MVP, Baum-Welch fitted in production.

    `decode` returns both the Viterbi path and the forward-backward posterior,
    because the runtime needs both: the path is the hypnogram, the posterior is
    `stage_confidence`. They answer different questions and can legitimately
    disagree — Viterbi optimises the whole sequence, the posterior is per-row
    marginal, so the argmax of the posterior at row t is not always path[t].
    """

    version = "viterbi_fixed_v1"

    def __init__(
        self,
        cfg: HMMConfig,
        *,
        prior_correction: bool | None = None,
        class_prior: np.ndarray | None = None,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self._log_a = _log(cfg.transition_matrix)
        self._log_pi = _log(cfg.initial_distribution)

        self.prior_correction = (
            cfg.prior_correction if prior_correction is None else prior_correction
        )
        prior = class_prior if class_prior is not None else cfg.class_prior
        if prior is None:
            prior = stationary_distribution(cfg.transition_matrix)
        self._log_prior = _log(prior)

    # ---- emissions -----------------------------------------------------------

    def _log_emissions(
        self, probabilities: np.ndarray, gaps: np.ndarray | None
    ) -> np.ndarray:
        probs = np.atleast_2d(np.asarray(probabilities, dtype=np.float64))
        if probs.ndim != 2 or probs.shape[1] != _K:
            raise ValueError(f"expected (T, {_K}) probabilities, got {probs.shape}")
        if probs.shape[0] == 0:
            raise ValueError("cannot decode an empty sequence")

        log_b = _log(probs)

        # D-012: Layer 2 emits P(stage|x) but an HMM emission wants P(x|stage).
        # Dividing by the class prior converts posterior -> scaled likelihood. Off by
        # default because §7 specifies the softmax directly.
        if self.prior_correction:
            log_b = log_b - self._log_prior

        if gaps is not None:
            gaps = np.asarray(gaps, dtype=bool)
            if gaps.shape[0] != log_b.shape[0]:
                raise ValueError(f"gaps has length {gaps.shape[0]}, expected {log_b.shape[0]}")
            # §7: a missing row still decodes, using the transition prior alone. A
            # constant emission does exactly that — it cannot prefer any state, so
            # the transition matrix carries the row. Dropping the row instead would
            # silently compress the timeline and corrupt WASO and duration.
            log_b[gaps] = 0.0

        return log_b

    # ---- decoding ------------------------------------------------------------

    def decode(self, probabilities: np.ndarray, *, gaps: np.ndarray | None = None) -> ViterbiResult:
        log_b = self._log_emissions(probabilities, gaps)
        path, best_lp = self._viterbi(log_b)
        posterior, loglik = self._forward_backward(log_b)
        del best_lp  # the sequence log-prob; loglik (all paths) is the reported one
        return ViterbiResult(
            path=[Stage(int(i)) for i in path],
            posterior=posterior,
            log_likelihood=float(loglik),
        )

    def _viterbi(self, log_b: np.ndarray) -> tuple[np.ndarray, float]:
        t_len = log_b.shape[0]
        delta = np.empty((t_len, _K), dtype=np.float64)
        psi = np.zeros((t_len, _K), dtype=np.int64)

        delta[0] = self._log_pi + log_b[0]
        for t in range(1, t_len):
            # scores[i, j] = best log-prob ending in i at t-1, then i -> j
            scores = delta[t - 1][:, None] + self._log_a
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_b[t]

        path = np.empty(t_len, dtype=np.int64)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(t_len - 2, -1, -1):
            path[t] = psi[t + 1][path[t + 1]]
        return path, float(np.max(delta[-1]))

    def _forward_backward(self, log_b: np.ndarray) -> tuple[np.ndarray, float]:
        t_len = log_b.shape[0]

        log_alpha = np.empty((t_len, _K), dtype=np.float64)
        log_alpha[0] = self._log_pi + log_b[0]
        for t in range(1, t_len):
            log_alpha[t] = _logsumexp(log_alpha[t - 1][:, None] + self._log_a, axis=0) + log_b[t]

        log_beta = np.zeros((t_len, _K), dtype=np.float64)
        for t in range(t_len - 2, -1, -1):
            log_beta[t] = _logsumexp(self._log_a + (log_b[t + 1] + log_beta[t + 1])[None, :], axis=1)

        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma), float(_logsumexp(log_alpha[-1], axis=0))

    def posterior(self, probabilities: np.ndarray, *, gaps: np.ndarray | None = None) -> np.ndarray:
        return self._forward_backward(self._log_emissions(probabilities, gaps))[0]

    def log_likelihood(self, probabilities: np.ndarray, *, gaps: np.ndarray | None = None) -> float:
        return self._forward_backward(self._log_emissions(probabilities, gaps))[1]

    # ---- production ----------------------------------------------------------

    def fit(self, sequences: list[np.ndarray], *, iterations: int = 10) -> None:
        """Baum-Welch (§7/§10). Post-pilot — the MVP matrix is fixed from literature."""
        raise NotImplementedError("Phase 7: Baum-Welch transition fitting")


class PassthroughSmoother:
    """Argmax with no smoothing. Test scaffolding and an ablation baseline only.

    §7 exists precisely because this produces implausible sequences
    (Deep -> Wake -> Deep -> Wake). Never ship it — but keep it, because the
    Viterbi-vs-passthrough delta is the cleanest evidence Layer 3 earns its place.
    """

    version = "passthrough_v1"

    def decode(self, probabilities: np.ndarray, *, gaps: np.ndarray | None = None) -> ViterbiResult:
        probs = np.atleast_2d(np.asarray(probabilities, dtype=np.float64))
        path = [Stage(int(i)) for i in probs.argmax(axis=1)]
        return ViterbiResult(path=path, posterior=probs, log_likelihood=float("nan"))
