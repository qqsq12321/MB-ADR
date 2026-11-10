"""The adaptive retargeting loss, in Torch.

Mirrors :class:`mbadr.reference.ReferenceOptimizer` term for term, so a
network trained against it optimises *exactly* the objective the NLopt reference
minimises. The two are deliberately separate implementations: the reference uses
hand-written NumPy Jacobians, this one takes gradients from autograd through
:class:`TorchFK`, and ``scripts/verify_loss.py`` checks they agree.

Loss terms (all Huber on distances unless noted):

    tip position     huber(||robot_tip - wrist - target_tip||)
    tip direction    huber(||normalize(tip - dip) - target_dir||)
    full hand        mean of huber over PIP/DIP/TIP vectors from the wrist
    contact          w_contact * alpha * ||thumb_tip - partner_tip||^2

Per-finger blend::

    L = alpha * (w_pos * L_tip_pos + w_dir * L_tip_dir) + (1 - alpha) * w_full * L_full

where ``alpha`` is the continuous pinch alpha, so posture and pinch objectives
hand over smoothly rather than switching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from .config import (
    FINGER_NAMES,
    HandSpec,
    MP_DIP_INDICES,
    MP_MCP_INDICES,
    MP_ORIGIN_IDX,
    MP_PIP_INDICES,
    MP_TIP_INDICES,
)

M_TO_CM = 100.0

# Mirrors AdaptiveOptimizerAnalytical.PINCH_ALPHA_MAX: alpha is capped below 1
# so a full pinch still keeps some of the full-hand posture objective, which is
# what stops the intermediate joints drifting once the tip is satisfied.
PINCH_ALPHA_MAX = 0.7


def huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Elementwise Huber loss, matching ``huber_loss_np``."""
    a = x.abs()
    return torch.where(a <= delta, 0.5 * x * x, delta * (a - 0.5 * delta))


@dataclass
class AdaptiveParams:
    """Tunable weights, mirroring the ``retarget:`` section of the YAML.

    The defaults are ``AdaptiveOptimizerAnalytical``'s own defaults, not the
    values from any particular config. They matter: a hand whose YAML omits a
    key falls back here, and if these disagree with the reference the two
    implementations silently optimise different objectives. L20's config writes
    every key explicitly, which is exactly why such a mismatch stays invisible
    until a second hand is tried.
    """

    huber_delta: float = 2.0
    huber_delta_dir: float = 0.5
    w_pos: float = 1.0
    w_dir: float = 10.0
    w_full_hand: float = 1.0
    w_contact: float = 0.0
    alpha: float = PINCH_ALPHA_MAX
    pinch_scaling: float = 1.0
    lateral_scaling: float = 1.0
    lateral_axis: int = 1

    @classmethod
    def from_yaml_section(cls, retarget: dict) -> "AdaptiveParams":
        from dataclasses import fields

        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in retarget.items() if k in known}
        return cls(**kwargs)


class AdaptiveLoss(nn.Module):
    """Differentiable adaptive loss over L20 joint configurations.

    Args:
        spec: hand specification (links, offsets).
        fk: the differentiable FK module.
        params: loss weights.
        segment_scaling: (5, 4) per-finger MCP/PIP/DIP/TIP scaling.
        pinch_thresholds: per non-thumb finger ``(d1, d2)`` in cm.
    """

    def __init__(
        self,
        spec: HandSpec,
        fk: nn.Module,
        params: AdaptiveParams,
        segment_scaling: np.ndarray,
        pinch_thresholds: tuple[np.ndarray, np.ndarray],
    ):
        super().__init__()
        self.spec = spec
        self.fk = fk
        self.params = params

        nf = spec.num_fingers
        segment_scaling = np.asarray(segment_scaling, dtype=np.float64)
        if segment_scaling.shape != (nf, 4):
            raise ValueError(
                f"segment_scaling must be ({nf}, 4), got {segment_scaling.shape}"
            )
        self.register_buffer(
            "segment_scaling", torch.from_numpy(segment_scaling).unsqueeze(0)
        )

        d1, d2 = pinch_thresholds
        self.register_buffer("d1", torch.from_numpy(np.asarray(d1, dtype=np.float64)))
        self.register_buffer("d2", torch.from_numpy(np.asarray(d2, dtype=np.float64)))

        # Point set queried from FK, in a fixed order:
        #   [0]      wrist / origin link
        #   [1..nf]  fingertip points (distal + offset)
        #   [1+nf..] link3 (proximal) points
        #   ...      link4 (distal) points
        self.link_order = (
            [spec.origin_link]
            + list(spec.tip_links)
            + list(spec.link3_links)
            + list(spec.link4_links)
        )
        self.offset_order = np.vstack(
            [
                np.zeros((1, 3)),
                spec.tip_offsets,
                spec.link3_offsets,
                spec.link4_offsets,
            ]
        )
        self.register_buffer(
            "offsets_t", torch.from_numpy(self.offset_order).unsqueeze(0)
        )

        self.idx_origin = 0
        self.idx_tip = slice(1, 1 + nf)
        self.idx_link3 = slice(1 + nf, 1 + 2 * nf)
        self.idx_link4 = slice(1 + 2 * nf, 1 + 3 * nf)

        # MediaPipe landmark indices actually used, for a 5-finger hand.
        self.mp_fingers = list(range(nf))

    # ---- target construction ---------------------------------------------

    def pinch_alpha(self, kp: torch.Tensor) -> torch.Tensor:
        """(B, 21, 3) keypoints -> (B, nf) continuous pinch weights."""
        thumb_tip = kp[:, MP_TIP_INDICES[0]]
        tips = torch.stack(
            [kp[:, MP_TIP_INDICES[i]] for i in self.mp_fingers[1:]], dim=1
        )
        distances = (tips - thumb_tip[:, None]).norm(dim=-1) * M_TO_CM
        cap = self.params.alpha
        alphas_nt = torch.clamp(
            (self.d2 - distances) / (self.d2 - self.d1 + 1e-8), 0.0, cap
        )
        alpha_thumb = alphas_nt.max(dim=1, keepdim=True).values
        return torch.cat([alpha_thumb, alphas_nt], dim=1)

    def full_hand_vectors(self, kp: torch.Tensor) -> torch.Tensor:
        """(B, 21, 3) keypoints (meters) -> (B, nf*3, 3) PIP/DIP/TIP targets in cm.

        Mirrors ``BaseOptimizer._compute_full_hand_vectors``: each finger chain
        is grown segment by segment from the wrist, scaling every span
        independently.
        """
        kp_cm = kp * M_TO_CM
        nf = self.spec.num_fingers
        mcp_i = [MP_MCP_INDICES[i] for i in self.mp_fingers]
        pip_i = [MP_PIP_INDICES[i] for i in self.mp_fingers]
        dip_i = [MP_DIP_INDICES[i] for i in self.mp_fingers]
        tip_i = [MP_TIP_INDICES[i] for i in self.mp_fingers]

        wrist = kp_cm[:, MP_ORIGIN_IDX]  # (B, 3)
        mcp = kp_cm[:, mcp_i]  # (B, nf, 3)
        pip = kp_cm[:, pip_i]
        dip = kp_cm[:, dip_i]
        tip = kp_cm[:, tip_i]

        palm = mcp - wrist[:, None]
        if self.params.lateral_scaling != 1.0:
            ax = self.params.lateral_axis
            # Build the scale vector without in-place ops (autograd safe).
            mult = torch.ones(3, dtype=kp.dtype, device=kp.device)
            mult = torch.cat(
                [
                    mult[:ax],
                    mult[ax : ax + 1] * self.params.lateral_scaling,
                    mult[ax + 1 :],
                ]
            )
            palm = palm * mult

        proximal = pip - mcp
        middle = dip - pip
        distal = tip - dip

        s = self.segment_scaling.to(kp.dtype)  # (1, nf, 4)
        mcp_vec = palm * s[:, :, 0:1]
        pip_vec = mcp_vec + proximal * s[:, :, 1:2]
        dip_vec = pip_vec + middle * s[:, :, 2:3]
        tip_vec = dip_vec + distal * s[:, :, 3:4]
        return torch.cat([pip_vec, dip_vec, tip_vec], dim=1)  # (B, 3nf, 3)

    def tip_directions(self, kp: torch.Tensor) -> torch.Tensor:
        """(B, 21, 3) -> (B, nf, 3) normalized DIP->TIP directions."""
        dips = kp[:, [MP_DIP_INDICES[i] for i in self.mp_fingers]]
        tips = kp[:, [MP_TIP_INDICES[i] for i in self.mp_fingers]]
        vec = tips - dips
        return vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)

    def targets(
        self, kp: torch.Tensor, alphas: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """Build every target the loss needs from keypoints."""
        nf = self.spec.num_fingers
        full = self.full_hand_vectors(kp)
        tip_vectors = full[:, 2 * nf :].clone()

        if alphas is not None and self.params.pinch_scaling != 1.0 and nf > 1:
            batch = kp.shape[0]
            partner = alphas[:, 1:].argmax(dim=1) + 1  # (B,)
            alpha = alphas.gather(1, partner[:, None]).squeeze(1)  # (B,)
            beta = torch.clamp(alpha / (self.params.alpha + 1e-8), 0.0, 1.0)  # (B,)

            kp_cm = kp * M_TO_CM
            wrist = kp_cm[:, MP_ORIGIN_IDX]
            tips = kp_cm[:, [MP_TIP_INDICES[i] for i in self.mp_fingers]]
            pinch_vectors = (tips - wrist[:, None]) * self.params.pinch_scaling

            # Select exactly the thumb and the active partner finger with a
            # one-hot mask, which keeps the blend batch-vectorised and avoids
            # in-place writes into the autograd graph.
            finger_ids = torch.arange(nf, device=kp.device)
            mask = (finger_ids[None, :] == 0) | (
                finger_ids[None, :] == partner[:, None]
            )  # (B, nf)
            w = (beta[:, None] * mask).unsqueeze(-1)  # (B, nf, 1)
            tip_vectors = (1.0 - w) * tip_vectors + w * pinch_vectors

        return {
            "tip_vectors": tip_vectors,
            "tip_dirs": self.tip_directions(kp),
            "full_hand": full,
        }

    # ---- the objective ---------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        targets: dict[str, torch.Tensor],
        alphas: torch.Tensor,
        last_q: Optional[torch.Tensor] = None,
        norm_delta: float = 0.0,
    ) -> torch.Tensor:
        """Total loss for a batch of independent joint values.

        Args:
            x: (B, n_independent) independent joints.
            targets: output of :meth:`targets`.
            alphas: (B, nf) pinch weights.
            last_q: optional (B, n_independent) previous pose for temporal
                regularisation.
            norm_delta: weight of the temporal term.

        Returns:
            (B,) per-sample loss.
        """
        nf = self.spec.num_fingers
        q = self.fk.expand(x)
        pos = self.fk.points(q, self.link_order, self.offset_order) * M_TO_CM  # (B,N,3)

        wrist = pos[:, self.idx_origin]  # (B, 3)
        tip_pos = pos[:, self.idx_tip]
        link3_pos = pos[:, self.idx_link3]
        link4_pos = pos[:, self.idx_link4]

        p = self.params

        # --- tip position --------------------------------------------------
        robot_tip_vec = tip_pos - wrist[:, None]
        diff = robot_tip_vec - targets["tip_vectors"]
        dist = diff.norm(dim=-1)  # (B, nf)
        loss_tip_pos = huber(dist, p.huber_delta)

        # --- tip direction -------------------------------------------------
        robot_dir = tip_pos - link4_pos
        robot_dir = robot_dir / (robot_dir.norm(dim=-1, keepdim=True) + 1e-8)
        diff_dir = robot_dir - targets["tip_dirs"]
        dist_dir = diff_dir.norm(dim=-1)
        loss_tip_dir = huber(dist_dir, p.huber_delta_dir)

        # --- full hand posture ---------------------------------------------
        full = targets["full_hand"]
        robot_pip = link3_pos - wrist[:, None]
        robot_dip = link4_pos - wrist[:, None]
        robot_tip_full = tip_pos - wrist[:, None]

        loss_pip = huber((robot_pip - full[:, :nf]).norm(dim=-1), p.huber_delta)
        loss_dip = huber(
            (robot_dip - full[:, nf : 2 * nf]).norm(dim=-1), p.huber_delta
        )
        loss_tip_full = huber(
            (robot_tip_full - full[:, 2 * nf :]).norm(dim=-1), p.huber_delta
        )
        loss_full = (loss_pip + loss_dip + loss_tip_full) / 3.0

        # --- per-finger blend ----------------------------------------------
        pinch_term = p.w_pos * loss_tip_pos + p.w_dir * loss_tip_dir
        total = (alphas * pinch_term + (1.0 - alphas) * p.w_full_hand * loss_full).sum(
            dim=1
        )

        # --- contact -------------------------------------------------------
        if p.w_contact > 0.0 and nf > 1:
            partner = alphas[:, 1:].argmax(dim=1) + 1
            alpha = alphas.gather(1, partner[:, None]).squeeze(1)
            idx = partner[:, None, None].expand(-1, 1, 3)
            partner_tip = tip_pos.gather(1, idx).squeeze(1)
            contact_diff = tip_pos[:, 0] - partner_tip
            total = total + p.w_contact * alpha * contact_diff.pow(2).sum(dim=1)

        # --- temporal regularisation ---------------------------------------
        if last_q is not None and norm_delta > 0.0:
            total = total + norm_delta * (x - last_q).pow(2).sum(dim=1)

        return total


__all__ = [
    "AdaptiveLoss",
    "AdaptiveParams",
    "PINCH_ALPHA_MAX",
    "huber",
    "M_TO_CM",
]
