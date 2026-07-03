"""Tabular MLP with categorical embeddings.

Optional (requires ``torch``). Use as a second baseline for the action head;
uses learned embeddings for the 169 hand classes and 6 positions, concatenated
with continuous features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - optional dep
    _HAS_TORCH = False


if _HAS_TORCH:

    class PreflopMLP(nn.Module):
        """MLP with embeddings for hand class and position."""

        def __init__(
            self,
            n_continuous: int,
            n_hand_classes: int = 169,
            n_positions: int = 6,
            hand_dim: int = 16,
            pos_dim: int = 4,
            hidden: tuple[int, ...] = (128, 64),
            n_actions: int = 5,
            dropout: float = 0.15,
        ):
            super().__init__()
            self.hand_embed = nn.Embedding(n_hand_classes, hand_dim)
            self.pos_embed = nn.Embedding(n_positions, pos_dim)

            in_dim = n_continuous + hand_dim + pos_dim
            layers: list[nn.Module] = []
            prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            self.trunk = nn.Sequential(*layers)
            self.action_head = nn.Linear(prev, n_actions)
            self.villain_fold_head = nn.Linear(prev, 1)

        def forward(
            self,
            x_cont: "torch.Tensor",
            hand_idx: "torch.Tensor",
            pos_idx: "torch.Tensor",
        ) -> dict[str, "torch.Tensor"]:
            h = self.hand_embed(hand_idx)
            p = self.pos_embed(pos_idx)
            z = torch.cat([x_cont, h, p], dim=-1)
            z = self.trunk(z)
            return {
                "action_logits": self.action_head(z),
                "villain_fold_logit": self.villain_fold_head(z).squeeze(-1),
            }


@dataclass
class TorchMLPConfig:
    hand_dim: int = 16
    pos_dim: int = 4
    hidden: tuple[int, ...] = (128, 64)
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 20


def has_torch() -> bool:
    return _HAS_TORCH


def build_model(n_continuous: int, n_actions: int, cfg: TorchMLPConfig | None = None) -> Any:
    if not _HAS_TORCH:
        raise RuntimeError("torch is not installed; install extras: pip install '.[torch]'")
    cfg = cfg or TorchMLPConfig()
    return PreflopMLP(
        n_continuous=n_continuous,
        hand_dim=cfg.hand_dim,
        pos_dim=cfg.pos_dim,
        hidden=cfg.hidden,
        n_actions=n_actions,
        dropout=cfg.dropout,
    )
