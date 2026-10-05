"""Typed configuration, loaded from `config/default.yaml`.

Validation happens at load, not at use. A bad threshold should fail when the process
starts, not eight hours into a LOSO sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .errors import ConfigError
from .types import Stage

__all__ = ["Config", "load_config", "DEFAULT_CONFIG_PATH"]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass(frozen=True, slots=True)
class AggregationConfig:
    row_minutes: int
    epoch_seconds: int
    train_stride_minutes: int
    min_valid_epochs: int

    @property
    def epochs_per_row(self) -> int:
        return self.row_minutes * 60 // self.epoch_seconds

    @property
    def uses_overlapping_windows(self) -> bool:
        """If true, subject-grouped splitting is mandatory — overlapping windows from
        one subject leak into each other trivially."""
        return self.train_stride_minutes < self.row_minutes

    def validate(self) -> None:
        if self.row_minutes * 60 % self.epoch_seconds:
            raise ConfigError("row_minutes must be a whole number of epochs")
        if not 0 < self.min_valid_epochs <= self.epochs_per_row:
            raise ConfigError(f"min_valid_epochs must be in (0, {self.epochs_per_row}]")


@dataclass(frozen=True, slots=True)
class StateMachineConfig:
    #: ONE gate, not two. §4 originally specified a 0.5 candidate threshold and a 0.7
    #: confirm threshold, but they were measured to select the same row on 97.5% of
    #: nights: P(asleep) is near-binary, jumping 0.26 -> 0.75 across a single row, so
    #: only 0.12% of rows ever fall between them. The hysteresis band was inert and the
    #: two constants read as tuned when they were not — any value in 0.1-0.9 scored
    #: within 1.4 pp. Onset is now the FIRST of `onset_confirm_rows` consecutive rows
    #: above this threshold.
    onset_threshold: float
    onset_confirm_rows: int
    wake_confirm_rows: int
    candidate_timeout_hours: float
    nap_max_duration_min: int
    allow_redetection: bool

    def validate(self) -> None:
        if not 0.0 < self.onset_threshold < 1.0:
            raise ConfigError(
                f"onset_threshold must be in (0, 1), got {self.onset_threshold}")
        if self.onset_confirm_rows < 1:
            raise ConfigError(
                f"onset_confirm_rows must be >= 1, got {self.onset_confirm_rows}")
        if self.allow_redetection:
            raise ConfigError("§4 forbids re-running Layer 1 once ASLEEP")


@dataclass(frozen=True, slots=True)
class HMMConfig:
    transition_matrix_version: str
    transition_matrix: np.ndarray
    initial_distribution: np.ndarray
    laplace_alpha: float
    missing_row_policy: str
    prior_correction: bool = False
    class_prior: np.ndarray | None = None

    def validate(self) -> None:
        k = Stage.n_classes()
        if self.transition_matrix.shape != (k, k):
            raise ConfigError(f"transition matrix must be {k}x{k}")
        rowsums = self.transition_matrix.sum(axis=1)
        if not np.allclose(rowsums, 1.0, atol=1e-6):
            raise ConfigError(f"transition matrix rows must sum to 1, got {rowsums}")
        if not np.isclose(self.initial_distribution.sum(), 1.0, atol=1e-6):
            raise ConfigError("initial_distribution must sum to 1")
        if (self.transition_matrix < 0).any():
            raise ConfigError("transition matrix has negative entries")


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    cold_start_nights: int
    within_night_reference: str
    cycle_phase_subtraction: bool


@dataclass(frozen=True, slots=True)
class Targets:
    """§9 evaluation targets. Consumed by `eval.targets` as assertions."""

    values: dict[str, float]

    def get(self, metric: str) -> float | None:
        return self.values.get(metric)

    def check(self, metric: str, observed: float) -> bool | None:
        """None when no target is defined for the metric."""
        target = self.get(metric)
        if target is None:
            return None
        # Error-style metrics are "lower is better"; everything else is "higher".
        if metric.endswith("_min") or metric.endswith("_err"):
            return observed <= target
        return observed >= target


@dataclass(frozen=True, slots=True)
class DataConfig:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    sample_rate_hz: int
    excluded_subjects: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    scheme: str
    n_folds: int | None
    report_per_subject: bool
    stratify_by: tuple[str, ...]

    def validate(self) -> None:
        if self.scheme != "loso":
            raise ConfigError(
                "§8 mandates subject-level LOSO; random row splits leak subject "
                "physiology and produce fake accuracy"
            )


@dataclass(frozen=True, slots=True)
class Config:
    pipeline_version: str
    aggregation: AggregationConfig
    state_machine: StateMachineConfig
    hmm: HMMConfig
    baselines: BaselineConfig
    evaluation: EvaluationConfig
    data: DataConfig
    mvp_targets: Targets
    production_targets: Targets
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def validate(self) -> None:
        self.aggregation.validate()
        self.state_machine.validate()
        self.hmm.validate()
        self.evaluation.validate()


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    try:
        cfg = Config(
            pipeline_version=raw["pipeline_version"],
            aggregation=AggregationConfig(**raw["aggregation"]),
            state_machine=StateMachineConfig(**raw["state_machine"]),
            hmm=HMMConfig(
                transition_matrix_version=raw["hmm"]["transition_matrix_version"],
                transition_matrix=np.asarray(raw["hmm"]["transition_matrix"], dtype=np.float64),
                initial_distribution=np.asarray(
                    raw["hmm"]["initial_distribution"], dtype=np.float64
                ),
                laplace_alpha=float(raw["hmm"]["laplace_alpha"]),
                missing_row_policy=raw["hmm"]["missing_row_policy"],
                prior_correction=bool(raw["hmm"].get("prior_correction", False)),
                class_prior=(
                    np.asarray(raw["hmm"]["class_prior"], dtype=np.float64)
                    if raw["hmm"].get("class_prior")
                    else None
                ),
            ),
            baselines=BaselineConfig(**raw["baselines"]),
            evaluation=EvaluationConfig(
                scheme=raw["evaluation"]["scheme"],
                n_folds=raw["evaluation"]["n_folds"],
                report_per_subject=bool(raw["evaluation"]["report_per_subject"]),
                stratify_by=tuple(raw["evaluation"]["stratify_by"]),
            ),
            data=DataConfig(
                raw_dir=Path(raw["data"]["raw_dir"]),
                interim_dir=Path(raw["data"]["interim_dir"]),
                processed_dir=Path(raw["data"]["processed_dir"]),
                sample_rate_hz=int(raw["data"]["sample_rate_hz"]),
                excluded_subjects=tuple(raw["data"]["excluded_subjects"] or ()),
            ),
            mvp_targets=Targets(dict(raw["targets"]["mvp"])),
            production_targets=Targets(dict(raw["targets"]["production"])),
            raw=raw,
        )
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"malformed config at {path}: {exc}") from exc

    cfg.validate()
    return cfg
