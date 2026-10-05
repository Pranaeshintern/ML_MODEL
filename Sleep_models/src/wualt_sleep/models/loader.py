"""Load a trained checkpoint and run inference.

A `.pt` file on its own is not a model. It is a bag of tensors that is only
meaningful alongside the feature ORDER and the normalisation constants used at
training time — present them differently and the model returns confident nonsense
rather than an error. Both live in the `.meta.json` written next to the weights,
and this module is what keeps them together.
"""

from __future__ import annotations

from collections.abc import Sequence

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..types import Stage
from .architectures import OnsetTCN, build_stage_model
from .normalise import per_night_transform

__all__ = ["LoadedModel", "LoadedTreeModel", "load_stage_model", "load_onset_model"]


@dataclass(frozen=True)
class LoadedTreeModel:
    """A gradient-boosted onset model plus its calibrator.

    Deliberately NOT a `LoadedModel`. The two differ in ways that would be silent if
    the same wrapper served both:

      - No standardisation. The trees were fitted on raw feature values, so the
        artifact carries `standardised: false` and no mu/sd. Applying the TCN's
        constants here would rescale inputs the model never saw scaled.
      - NaN IS THE SIGNAL. The tree learner splits on missingness directly, so NaN
        must reach it intact. The torch path replaces NaN with 0 after standardising,
        which for a tree means "this sensor read exactly zero" rather than "this
        sensor was absent" — a wrong answer with no error.
      - No availability mask. The mask exists so a network can tell absent from zero;
        the trees get that from NaN itself, and appending it would double the width.
    """

    model: object                      # fitted sklearn-compatible estimator
    calibrator: object | None          # IsotonicCalibrator, or None if uncalibrated
    feature_order: list[str]
    meta: dict

    def prepare(self, rows: np.ndarray) -> np.ndarray:
        x = np.asarray(rows, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_order):
            raise ValueError(
                f"expected (T, {len(self.feature_order)}) matching feature_order, "
                f"got {x.shape}")
        return x                       # raw, NaN preserved

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        """(T, F) rows for one night -> (T,) P(asleep), calibrated if a calibrator
        was bundled."""
        p = np.asarray(self.model.predict_proba(self.prepare(rows))[:, 1],
                       dtype=np.float64)
        if self.calibrator is None:
            return p
        return self.calibrator.transform(np.column_stack([1.0 - p, p]))[:, 1]


@dataclass(frozen=True)
class LoadedModel:
    """A model plus everything needed to feed it correctly.

    `expects_mask` is not a style choice — the two layers genuinely differ. Layer 2
    trains on features CONCATENATED with a binary availability mask, because it
    consumes both corpora and has to distinguish "sensor absent" from "value zero".
    Layer 1 trains on DREAMT alone, where every feature is present, so it takes the
    features unaugmented. Feeding either the other's layout is a silent shape error
    at best and wrong numbers at worst, so it is inferred from the weights rather
    than assumed.
    """

    net: torch.nn.Module
    feature_order: list[str]
    mu: np.ndarray
    sd: np.ndarray
    meta: dict
    expects_mask: bool
    per_night: bool = False
    per_night_features: tuple[str, ...] = ()

    def prepare(self, rows: np.ndarray) -> torch.Tensor:
        """Standardise a (T, F) array with the TRAINING constants.

        Missing values become 0 after standardisation. Where the network expects a
        mask, the availability flags are appended so it can tell an absent sensor
        from a genuine zero, rather than the value being imputed.
        """
        x = np.asarray(rows, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_order):
            raise ValueError(
                f"expected (T, {len(self.feature_order)}) matching feature_order, "
                f"got {x.shape}")
        if self.per_night:
            # Must run BEFORE the training constants are applied, and on the WHOLE
            # night at once — it is a within-night statistic. Feeding rows one at a
            # time would silently skip it and the model would see a different input
            # distribution from the one it was fitted on.
            x = per_night_transform(x, self.feature_order, self.per_night_features)
        avail = np.isfinite(x).astype(np.float32)
        z = (np.nan_to_num(x, nan=0.0) - self.mu) / self.sd
        z = np.nan_to_num(z * avail, nan=0.0)
        if self.expects_mask:
            z = np.hstack([z, avail])
        return torch.tensor(z, dtype=torch.float32)


def _load(path: Path, build) -> LoadedModel:
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} missing — the weights alone are unusable without the "
            "feature order and normalisation constants")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    state = torch.load(path, map_location="cpu", weights_only=True)

    net = build(meta, state)
    net.load_state_dict(state)
    net.eval()

    order = meta.get("feature_order") or meta["input_feature_order"]
    # The first LayerNorm's shape is the network's true input width. If it is twice
    # the feature count, the model was trained with the availability mask appended.
    first_norm = next(v for k, v in state.items() if k.endswith(".0.weight") or k == "enc.0.weight")
    n_in = int(first_norm.shape[0])
    if n_in not in (len(order), 2 * len(order)):
        raise ValueError(
            f"checkpoint expects {n_in} inputs but feature_order has {len(order)} "
            "entries — the metadata does not match the weights")

    return LoadedModel(
        net=net,
        feature_order=order,
        mu=np.asarray(meta["mu"], dtype=np.float64),
        sd=np.asarray(meta["sd"], dtype=np.float64),
        meta=meta,
        expects_mask=(n_in == 2 * len(order)),
        per_night=bool(meta.get("per_night", False)),
        per_night_features=tuple(meta.get("per_night_features", ())),
    )


def load_stage_model(path: Path | str = "artifacts/layer2/model.pt") -> LoadedModel:
    """Layer 2. `predict` returns (T, 4) probabilities in `Stage.ordered()` order."""
    def build(meta, state):
        n_in = state["enc.net.1.weight"].shape[1]
        return build_stage_model(meta.get("model", "tcn"), n_in)
    return _load(Path(path), build)


def load_onset_model(
    path: Path | str = "artifacts/layer1/model.pt",
) -> LoadedModel | LoadedTreeModel:
    """Layer 1. Returns whichever artifact is present.

    The estimator is chosen by experiment, so the serving path must not assume one.
    A `.joblib` sibling means a gradient-boosted bundle; otherwise the torch weights
    are loaded. The meta's `model` field is checked against what was actually found,
    because a mismatch there is exactly the failure that silently serves the wrong
    architecture.
    """
    path = Path(path)
    bundle = path.with_suffix(".joblib")
    meta_path = path.with_suffix(".meta.json")

    if bundle.exists():
        import joblib

        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} missing — the bundle alone is unusable without the "
                "feature order")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("standardised", False):
            raise ValueError(
                f"{bundle} is a tree bundle but its meta says standardised=true; "
                "refusing to guess whether to scale the inputs")
        obj = joblib.load(bundle)
        order = meta.get("feature_order") or meta["input_feature_order"]
        model = obj["model"]
        n_in = getattr(model, "n_features_in_", len(order))
        if n_in != len(order):
            raise ValueError(
                f"bundle expects {n_in} inputs but feature_order has {len(order)} "
                "entries — the metadata does not match the model")
        return LoadedTreeModel(model=model, calibrator=obj.get("calibrator"),
                               feature_order=order, meta=meta)

    def build(meta, state):
        n_in = state["enc.1.weight"].shape[1]
        return OnsetTCN(n_in, causal=meta.get("causal", True))
    return _load(path, build)


#: ---- the feature contract -------------------------------------------------------
#: ONE list in, each model's subset selected here. Before this, three lists disagreed:
#: the feature dictionary declares 32 shared features, Layer 1 consumes 30 and Layer 2
#: consumes 33, and four of those (`hr_succ_diff_rms`, `hr_range_iqr`,
#: `hr_autocorr_lag1`, `hr_coverage`) are in NEITHER dictionary -- they were built for
#: BIDSleep's 0.2 Hz heart-rate channel and have no definition for a ring. A row built
#: to the dictionary could not feed either model, and the only way to stream a night was
#: to bypass validation. That is now a named error instead of a workaround.


class FeatureContractError(ValueError):
    """A model needs features the row cannot supply. Names them, never guesses."""


def missing_features(dict_names: Sequence[str], feature_order: Sequence[str]) -> list[str]:
    """Which of a model's inputs the row contract cannot supply."""
    have = set(dict_names)
    return [c for c in feature_order if c not in have]


def select_features(values: np.ndarray, dict_names: Sequence[str],
                    feature_order: Sequence[str]) -> np.ndarray:
    """Dictionary-ordered row values -> one model's input vector.

    The caller builds ONE vector, in feature-dictionary order, and each model takes the
    columns it declared when it was saved. Order is contractual: presenting features in
    a different order returns a confident wrong answer rather than an error, which is
    why this does the selection by NAME and refuses when it cannot.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        v = v[None, :]
    if v.shape[1] != len(dict_names):
        raise FeatureContractError(
            f"row has {v.shape[1]} values but the contract names {len(dict_names)}")
    gap = missing_features(dict_names, feature_order)
    if gap:
        raise FeatureContractError(
            f"the row contract cannot supply {len(gap)} feature(s) this model needs: "
            f"{gap}. Either define them for the delivered sensor set and add them to the "
            f"feature dictionary, or refit the model without them.")
    idx = {c: i for i, c in enumerate(dict_names)}
    out = v[:, [idx[c] for c in feature_order]]
    return out[0] if np.asarray(values).ndim == 1 else out


def contract_report(dict_names: Sequence[str],
                    models: dict[str, Sequence[str]]) -> dict[str, dict]:
    """What each model needs against what the row contract provides."""
    return {name: {"needs": len(order),
                   "missing": missing_features(dict_names, order),
                   "unused_by_this_model": [c for c in dict_names if c not in set(order)]}
            for name, order in models.items()}


def predict_stages(model: LoadedModel, rows: np.ndarray) -> np.ndarray:
    """(T, F) rows for ONE night -> (T, 4) stage probabilities.

    One night at a time: the temporal layers read across the sequence, so batching
    two nights together would let one night's rows inform the other's.
    """
    x = model.prepare(rows).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(model.net(x)[0], dim=-1).numpy()


def predict_onset_proba(model: LoadedModel | LoadedTreeModel,
                        rows: np.ndarray) -> np.ndarray:
    """(T, F) rows for ONE night -> (T,) P(asleep). Feed to the state machine.

    Accepts either artifact type. The tree bundle applies its own calibrator; the
    torch path returns a raw sigmoid, which is uncalibrated.
    """
    if isinstance(model, LoadedTreeModel):
        return model.predict_proba(rows)
    x = model.prepare(rows).unsqueeze(0)
    with torch.no_grad():
        return torch.sigmoid(model.net(x)[0]).numpy()


STAGE_ORDER = [s.label for s in Stage.ordered()]
