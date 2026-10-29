"""Small, safely loadable model and optimizer checkpoints."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class CheckpointMetadata:
    epoch: int
    best_miou: float
    config: dict[str, Any]


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    best_miou: float,
    config: Mapping[str, Any],
) -> None:
    """Atomically save state dicts and simple experiment metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_miou": float(best_miou),
        "config": dict(config),
    }
    with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    map_location: torch.device | str = "cpu",
) -> CheckpointMetadata:
    """Restore weights and optimizer state with PyTorch's restricted loader."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    required = {"epoch", "model_state_dict", "optimizer_state_dict", "best_miou", "config"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return CheckpointMetadata(
        epoch=int(checkpoint["epoch"]),
        best_miou=float(checkpoint["best_miou"]),
        config=dict(checkpoint["config"]),
    )
