"""Convert a TRUSTED legacy MB-ADR pickle to tensor-only format.

This maintenance command enables pickle loading and must only be used on files
whose origin you trust. Normal training and inference use the restricted format.
The output path must differ from the input; network parameter values are kept.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mbadr.checkpoint import FORMAT_VERSION, load_checkpoint


def portable(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(portable(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported metadata type: {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.out.resolve() or args.out.exists():
        parser.error("choose a new output path; existing files are not overwritten")
    checkpoint = portable(torch.load(args.input, map_location="cpu", weights_only=False))
    checkpoint["format_version"] = FORMAT_VERSION
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    load_checkpoint(args.out)
    print(f"Converted {args.input} -> {args.out}")


if __name__ == "__main__":
    main()
