"""Model architectures.

These live in the library, not inside the training scripts, for one concrete
reason: a checkpoint is only loadable if the class that produced it can be
constructed independently. Previously every architecture was a nested class inside
a training function, so `layer1_tcn.pt` and `layer2_tcn.pt` could not be loaded
without invoking training.

Module and attribute names here are load-bearing — they determine `state_dict`
keys. Renaming `enc`, `convs`, `norms` or `head` silently invalidates every saved
checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..types import Stage

__all__ = [
    "RowEncoder",
    "StageTCN",
    "StageBiLSTM",
    "StageMLP",
    "StageMultiHeadTCN",
    "OnsetTCN",
    "build_stage_model",
]

K = Stage.n_classes()


class RowEncoder(nn.Module):
    """Per-row feature encoder shared by every staging architecture (§6)."""

    def __init__(self, n_features: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_features), nn.Linear(n_features, 64), nn.GELU(),
            nn.Linear(64, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StageTCN(nn.Module):
    """Layer 2 — 4-class staging over a whole night.

    Non-causal: staging is not real-time and the runtime re-decodes the night
    retrospectively, so the row after this one is legitimately available.

    Dilations 1/2/4 give a receptive field of ~15 rows (3.75 h). Two blocks span
    only ~1.75 h, barely one 90-minute ultradian cycle; three span several, which
    is the scale at which Deep-front-loading and REM-back-loading are visible.
    """

    def __init__(self, n_features: int, hidden: int = 128, drop: float = 0.2,
                 dilations: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        self.enc = RowEncoder(n_features, hidden)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d) for d in dilations])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in dilations])
        self.act, self.drop = nn.GELU(), nn.Dropout(drop)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, K))

    def forward(self, x: torch.Tensor, head: int = 0) -> torch.Tensor:
        z = self.enc(x).transpose(1, 2)
        for conv, norm in zip(self.convs, self.norms):
            z = z + self.drop(self.act(norm(conv(z).transpose(1, 2)).transpose(1, 2)))
        return self.head(z.transpose(1, 2))


class StageBiLSTM(nn.Module):
    """Alternative temporal encoder. Scores within noise of StageTCN and costs ~2x
    the training time — kept because it is the evidence that the gain comes from
    having temporal context at all, not from the specific architecture."""

    def __init__(self, n_features: int, hidden: int = 128, drop: float = 0.2) -> None:
        super().__init__()
        self.enc = RowEncoder(n_features, hidden)
        self.rnn = nn.LSTM(hidden, hidden // 2, num_layers=1, batch_first=True,
                           bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, K))

    def forward(self, x: torch.Tensor, head: int = 0) -> torch.Tensor:
        z, _ = self.rnn(self.enc(x))
        return self.head(self.drop(z))


class StageMLP(nn.Module):
    """Row-wise only, no temporal layer. This is the CONTROL: identical encoder and
    loss to StageTCN, so the gap between them isolates what sequence modelling buys
    (kappa 0.357 vs 0.432). Not a candidate for deployment."""

    def __init__(self, n_features: int, hidden: int = 128, drop: float = 0.2) -> None:
        super().__init__()
        self.enc = RowEncoder(n_features, hidden)
        self.head = nn.Sequential(nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 64),
                                  nn.GELU(), nn.Linear(64, K))

    def forward(self, x: torch.Tensor, head: int = 0) -> torch.Tensor:
        return self.head(self.enc(x))


class StageMultiHeadTCN(nn.Module):
    """Shared trunk, one output head per corpus. Tested and REJECTED — scored below
    plain fine-tuning (kappa 0.403 vs 0.432). Retained so the negative result is not
    re-discovered later."""

    def __init__(self, n_features: int, hidden: int = 128, drop: float = 0.2,
                 n_heads: int = 2) -> None:
        super().__init__()
        self.trunk = StageTCN(n_features, hidden, drop)
        self.trunk.head = nn.Identity()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, K))
            for _ in range(n_heads)])

    def forward(self, x: torch.Tensor, head: int = 0) -> torch.Tensor:
        return self.heads[head](self.trunk(x))


class OnsetTCN(nn.Module):
    """Layer 1 — P(asleep) per row, for onset detection.

    `causal=True` uses left-only padding and is REQUIRED for the live onset event:
    at row t nothing after t exists, because the event fires so the ring can switch
    sampling mode. `causal=False` is valid only for a retrospective re-estimate
    written into the session record at finalisation (§7).
    """

    def __init__(self, n_features: int, hidden: int = 96,
                 dilations: tuple[int, ...] = (1, 2, 4), causal: bool = True) -> None:
        super().__init__()
        self.causal = causal
        self.enc = nn.Sequential(
            nn.LayerNorm(n_features), nn.Linear(n_features, 64), nn.GELU(),
            nn.Linear(64, hidden))
        self.convs = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, 3, dilation=d) for d in dilations])
        self.pads = [(2 * d, 0) if causal else (d, d) for d in dilations]
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in dilations])
        self.act, self.drop = nn.GELU(), nn.Dropout(0.15)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x).transpose(1, 2)
        for conv, norm, pad in zip(self.convs, self.norms, self.pads):
            c = conv(nn.functional.pad(z, pad))
            z = z + self.drop(self.act(norm(c.transpose(1, 2)).transpose(1, 2)))
        return self.head(z.transpose(1, 2)).squeeze(-1)


_STAGE_MODELS = {
    "tcn": StageTCN,
    "lstm": StageBiLSTM,
    "mlp": StageMLP,
    "tcn_mh": StageMultiHeadTCN,
}


def build_stage_model(kind: str, n_features: int) -> nn.Module:
    if kind not in _STAGE_MODELS:
        raise ValueError(f"unknown stage model {kind!r}; have {sorted(_STAGE_MODELS)}")
    return _STAGE_MODELS[kind](n_features)
