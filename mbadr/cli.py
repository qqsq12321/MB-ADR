"""Shared command-line helpers for the entry points.

Every entry point resolves the same two things: which device to run on, and
which checkpoint to load (defaulting to ``out/net_<robot>.pt``). Keeping the
resolution here means the demos, evaluation and the L20 hardware entry point
cannot drift apart on defaults or on what "checkpoint not found" says.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import OUT_DIR


def select_device(name: str | None) -> torch.device:
    """``--device`` value, or CUDA when available, else CPU."""
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def resolve_checkpoint(path: Path | str | None, robot: str) -> Path:
    """Explicit checkpoint path, or the bundled ``out/net_<robot>.pt``.

    Raises:
        FileNotFoundError: when the resolved path is not a file.
    """
    checkpoint = Path(path) if path else OUT_DIR / f"net_{robot}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    return checkpoint


__all__ = ["select_device", "resolve_checkpoint"]
