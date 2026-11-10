"""Amortised retargeting network.

Maps human keypoint-derived features to L20 joint angles in one forward pass,
replacing the per-frame NLopt solve. Trained self-supervised against the same
adaptive loss the optimiser minimises, with gradients obtained from the
analytic Torch FK — no joint labels are used anywhere.

Design choices that matter:

* **Joint limits are a hard constraint.** The head emits values in [-1, 1] via
  ``tanh``, mapped onto ``[lower, upper]`` per joint. The optimiser needs box
  bounds; the network gets them by construction, so it can never emit an
  unreachable pose.
* **Only independent joints are predicted.** Mimic coupling is applied by the
  FK module, so the 5 dependent joints are always exactly consistent with their
  drivers — unlike GeoRT, which predicts all 21 and then overwrites the
  dependent ones post hoc.
* **Inputs are the task representation, not raw landmarks.** The network sees
  the same target vectors the loss compares against, so learned calibration
  (``segment_scaling``, ``pinch_scaling``) is applied outside the network and
  stays editable without retraining.
"""

from __future__ import annotations

import torch
from torch import nn


class RetargetNet(nn.Module):
    """MLP from keypoint targets to independent joint values.

    Args:
        in_dim: feature dimension produced by :meth:`build_features`.
        num_joints: number of independent joints (16 for L20).
        hidden: hidden width.
        layers: number of hidden layers.
        lower/upper: (num_joints,) joint limits in radians.
    """

    def __init__(
        self,
        in_dim: int,
        num_joints: int,
        lower: torch.Tensor,
        upper: torch.Tensor,
        hidden: int = 256,
        layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("lower", lower.clone().float())
        self.register_buffer("upper", upper.clone().float())
        self.register_buffer("centre", ((lower + upper) / 2.0).float())
        self.register_buffer("half_range", ((upper - lower) / 2.0).float())

        blocks: list[nn.Module] = []
        width = in_dim
        for _ in range(layers):
            blocks.append(nn.Linear(width, hidden))
            blocks.append(nn.LayerNorm(hidden))
            blocks.append(nn.SiLU())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            width = hidden
        blocks.append(nn.Linear(width, num_joints))
        self.net = nn.Sequential(*blocks)

        # Start near the middle of each joint's range: an all-zero output then
        # means "mid-range posture", which is a far better initialisation than
        # tanh(0) == 0 (i.e. every joint at its lower limit).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) -> (B, num_joints) joint values in radians."""
        raw = torch.tanh(self.net(features))
        return self.centre + raw * self.half_range

    def to_normalised(self, x: torch.Tensor) -> torch.Tensor:
        """Radians -> [-1, 1], for diagnosing how hard the net is saturating."""
        return (x - self.centre) / (self.half_range + 1e-9)


def build_features(
    targets: dict[str, torch.Tensor],
    alphas: torch.Tensor,
    include_alphas: bool = True,
) -> torch.Tensor:
    """Flatten the loss targets into the network's input vector.

    The features are exactly the quantities the loss evaluates, so the network
    is learning to reproduce the optimiser's solution from the same
    information the optimiser sees.

    Args:
        targets: output of ``AdaptiveLoss.targets``.
        alphas: (B, nf) pinch weights.
        include_alphas: append alpha so the network can switch strategy between
            free posture and pinch, mirroring the loss's blend.

    Returns:
        (B, in_dim) float tensor.
    """
    parts = [
        targets["tip_vectors"].reshape(targets["tip_vectors"].shape[0], -1),
        targets["tip_dirs"].reshape(targets["tip_dirs"].shape[0], -1),
        targets["full_hand"].reshape(targets["full_hand"].shape[0], -1),
    ]
    if include_alphas:
        parts.append(alphas)
    return torch.cat(parts, dim=1)


def feature_dim(loss_fn, include_alphas: bool = True) -> int:
    """Dimension produced by :func:`build_features` for a given loss module."""
    nf = loss_fn.spec.num_fingers
    dim = nf * 3 + nf * 3 + nf * 3 * 3  # tip (nf,3) + dir (nf,3) + full (3nf,3)
    if include_alphas:
        dim += nf
    return dim


__all__ = ["RetargetNet", "build_features", "feature_dim"]
