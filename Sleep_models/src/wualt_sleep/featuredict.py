"""Loader and validator for `config/features.yaml`.

The feature dictionary is contractual: it fixes feature ORDER, which the ONNX export
and every fitted model depend on. Loading it here (rather than hardcoding a list)
means a spec change that breaks a saved model fails loudly at load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml

from .errors import FeatureDictionaryError

__all__ = ["FeatureSpec", "FeatureDictionary", "load_feature_dictionary", "DEFAULT_SPEC_PATH"]

DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[2] / "config" / "features.yaml"

MissingPolicy = Literal["nan", "error", "zero"]
Scope = Literal["shared", "layer1"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    scope: Scope
    source: str
    agg: str
    unit: str
    missing: str
    baseline_subtract: bool
    note: str | None = None

    @property
    def missing_fill(self) -> float | None:
        """Value to substitute when the feature cannot be computed.

        `None` means NaN — trees handle it natively and MUST see it rather than an
        imputed value (§1). `"error"` means the feature is always computable and its
        absence is a bug, so it has no fill.
        """
        if self.missing == "nan":
            return float("nan")
        if self.missing == "error":
            return None
        try:
            return float(self.missing)
        except ValueError as exc:
            raise FeatureDictionaryError(
                f"feature {self.name!r}: bad missing policy {self.missing!r}"
            ) from exc

    @property
    def raises_when_missing(self) -> bool:
        return self.missing == "error"


@dataclass(frozen=True)
class FeatureDictionary:
    """An ordered, immutable view of the feature spec."""

    version: int
    row_minutes: int
    epoch_seconds: int
    #: PPG sensor on-time per row, in seconds. This is a property of the DATASET, not
    #: of a loader: every corpus must be built to the same on-time or the cardiac
    #: features are not comparable across them. Owning it here is the fix for
    #: FAIL-STAGE-001, where DREAMT defaulted to 90 s and BIDSleep silently used the
    #: whole 900 s row, making 67% of the training table a 100% duty cycle the ring
    #: will never run. Both loaders read this value; neither has its own default.
    ppg_burst_seconds: float
    specs: tuple[FeatureSpec, ...]

    def __post_init__(self) -> None:
        names = [s.name for s in self.specs]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise FeatureDictionaryError(f"duplicate feature names: {sorted(dupes)}")

        # Invariant: all `shared` features precede all `layer1` features.
        # This is what lets `index_of` return an index valid for BOTH scopes — a
        # shared feature sits at the same position in a Layer 2 row (32 values) and a
        # Layer 1 row (36 values). Interleave them and every `Row.get` silently reads
        # the wrong column for one of the two layers.
        seen_layer1 = False
        for spec in self.specs:
            if spec.scope == "layer1":
                seen_layer1 = True
            elif seen_layer1:
                raise FeatureDictionaryError(
                    f"shared feature {spec.name!r} appears after a layer1 feature; "
                    "all shared features must come first so indices agree across scopes"
                )

    @property
    def epochs_per_row(self) -> int:
        return self.row_minutes * 60 // self.epoch_seconds

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise FeatureDictionaryError(f"unknown feature {name!r}") from exc

    def for_scope(self, scope: Scope) -> tuple[FeatureSpec, ...]:
        """Features visible to a given layer.

        Layer 2 sees only `shared`; Layer 1 sees `shared` + `layer1` (§5).
        """
        if scope == "shared":
            return tuple(s for s in self.specs if s.scope == "shared")
        return self.specs

    def baseline_subtracted(self) -> tuple[FeatureSpec, ...]:
        """Features the per-cycle-phase personal baseline is subtracted from (§6)."""
        return tuple(s for s in self.specs if s.baseline_subtract)


def default_burst_seconds() -> float:
    """The PPG on-time every corpus must be built to. One source, read by both loaders.

    Exists so that neither `DreamtConfig` nor `BidsConfig` carries its own default.
    They disagreed once — 90 s against a silent 900 s — and it cost every headline
    number in the project roughly kappa 0.06. See FAIL-STAGE-001.
    """
    return load_feature_dictionary().ppg_burst_seconds


def validate_burst(burst: float | None, row_seconds: float, who: str) -> None:
    """Reject a burst that cannot describe a real sensing regime.

    `None` is rejected explicitly rather than falling back to the row length. That
    fallback is what made the original defect silent: it turned "unspecified" into
    "100% duty cycle" with no error and no plausible-looking wrong number.
    """
    if burst is None:
        raise FeatureDictionaryError(
            f"{who}: burst_seconds is None. There is no implicit duty cycle — pass an "
            f"explicit value, or leave it unset to take the {default_burst_seconds():g} s "
            "declared in the feature dictionary."
        )
    if not 0 < burst <= row_seconds:
        raise FeatureDictionaryError(
            f"{who}: burst_seconds={burst} must be in (0, {row_seconds}]"
        )


@lru_cache(maxsize=4)
def load_feature_dictionary(path: Path | str = DEFAULT_SPEC_PATH) -> FeatureDictionary:
    path = Path(path)
    if not path.exists():
        raise FeatureDictionaryError(f"feature dictionary not found at {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        specs = tuple(
            FeatureSpec(
                name=f["name"],
                scope=f["scope"],
                source=f["source"],
                agg=f["agg"],
                unit=f["unit"],
                missing=str(f["missing"]),
                baseline_subtract=bool(f["baseline_subtract"]),
                note=f.get("note"),
            )
            for f in raw["features"]
        )
        return FeatureDictionary(
            version=int(raw["version"]),
            row_minutes=int(raw["row_minutes"]),
            epoch_seconds=int(raw["epoch_seconds"]),
            ppg_burst_seconds=float(raw["ppg_burst_seconds"]),
            specs=specs,
        )
    except (KeyError, TypeError) as exc:
        raise FeatureDictionaryError(f"malformed feature dictionary at {path}: {exc}") from exc
