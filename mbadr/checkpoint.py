"""Tensor-only checkpoint storage; inference never enables arbitrary unpickling."""

from __future__ import annotations

import pickle
from pathlib import Path

import torch

FORMAT_VERSION = 2


def load_checkpoint(path: str | Path) -> dict:
    """Read tensors and primitive metadata on CPU, including on PyTorch 2.6+."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "Checkpoint contains legacy or unsupported pickle objects. "
            "For a trusted legacy MB-ADR file only, use "
            "scripts/migrate_checkpoint.py INPUT --out OUTPUT."
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected an MB-ADR checkpoint dictionary")
    required = {"state_dict", "in_dim", "num_joints", "hidden", "layers", "lower", "upper"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    version = checkpoint.get("format_version", 1)
    if not isinstance(version, int) or not 1 <= version <= FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format version")
    for key in ("in_dim", "num_joints", "hidden", "layers"):
        if not isinstance(checkpoint[key], int) or checkpoint[key] <= 0:
            raise ValueError(f"Invalid checkpoint {key}")
    for key in ("lower", "upper"):
        bound = torch.as_tensor(checkpoint[key])
        if bound.shape != (checkpoint["num_joints"],) or not torch.isfinite(bound).all():
            raise ValueError(f"Invalid checkpoint {key} limits")
        checkpoint[key] = bound
    if not torch.all(checkpoint["lower"] <= checkpoint["upper"]):
        raise ValueError("Checkpoint lower limits exceed upper limits")
    return checkpoint
