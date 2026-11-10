"""Batched, differentiable forward kinematics in Torch.

Same kinematic model as :mod:`mbadr.fk_numpy`, vectorised over a batch of
poses so training can run on GPU. Mimic expansion is a differentiable linear
map, which is the key structural difference from GeoRT: the network only ever
predicts the 16 independent DOF, and the 5 mimic-coupled joints are *derived*,
so the fingertip position the loss sees is the fingertip position the hardware
will actually reach.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .chain import KinematicChain
from .fk_numpy import topological_joint_order


def _rodrigues(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrices for batched ``angle`` about a fixed unit ``axis``.

    Args:
        axis: (3,) unit axis.
        angle: (B,) rotation angles.

    Returns:
        (B, 3, 3) rotation matrices.
    """
    k = axis / axis.norm()
    K = torch.zeros(3, 3, dtype=angle.dtype, device=angle.device)
    K[0, 1], K[0, 2] = -k[2], k[1]
    K[1, 0], K[1, 2] = k[2], -k[0]
    K[2, 0], K[2, 1] = -k[1], k[0]

    sin = torch.sin(angle)[:, None, None]
    cos = torch.cos(angle)[:, None, None]
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
    return eye + sin * K + (1.0 - cos) * (K @ K)


class TorchFK(nn.Module):
    """Differentiable FK for a :class:`KinematicChain`, in meters.

    All structure is stored as buffers so the module moves with ``.to(device)``.
    """

    def __init__(self, chain: KinematicChain):
        super().__init__()
        self.chain = chain
        self.order = topological_joint_order(chain)

        njoints = len(chain.joints)
        parent_idx = np.full(njoints, -1, dtype=np.int64)
        origin = np.zeros((njoints, 4, 4), dtype=np.float64)
        axis = np.zeros((njoints, 3), dtype=np.float64)
        q_slot = np.full(njoints, -1, dtype=np.int64)
        is_prismatic = np.zeros(njoints, dtype=bool)

        slot = 0
        for ji, joint in enumerate(chain.joints):
            origin[ji] = joint.origin_T
            axis[ji] = joint.axis
            if joint.is_actuated:
                q_slot[ji] = slot
                slot += 1
            is_prismatic[ji] = joint.type == "prismatic"
            parent_idx[ji] = chain.link_parent_joint.get(joint.parent, -1)
        assert slot == chain.nq

        self.register_buffer("parent_idx", torch.from_numpy(parent_idx))
        self.register_buffer("origin_T", torch.from_numpy(origin))
        self.register_buffer("axis", torch.from_numpy(axis))
        self.register_buffer("q_slot", torch.from_numpy(q_slot))
        self.register_buffer(
            "is_prismatic", torch.from_numpy(is_prismatic)
        )
        self.register_buffer(
            "order_t", torch.tensor(self.order, dtype=torch.long)
        )

        # Mimic expansion in Torch, mirroring KinematicChain.expand_to_full.
        indep_slots, mimic_slots, src_slots, mult, offset = chain.mimic_spec()
        self.n_independent = chain.num_independent
        self.register_buffer("indep_slots", torch.from_numpy(indep_slots))
        self.register_buffer("mimic_slots", torch.from_numpy(mimic_slots))
        self.register_buffer("mimic_src_slots", torch.from_numpy(src_slots))
        self.register_buffer("mimic_mult", torch.from_numpy(mult))
        self.register_buffer("mimic_offset", torch.from_numpy(offset))

        lim = chain.joint_limits()
        self.register_buffer("joint_lower", torch.from_numpy(lim[:, 0]))
        self.register_buffer("joint_upper", torch.from_numpy(lim[:, 1]))
        ilim = chain.independent_limits()
        self.register_buffer("indep_lower", torch.from_numpy(ilim[:, 0]))
        self.register_buffer("indep_upper", torch.from_numpy(ilim[:, 1]))

        # Per-joint switch resolving revolute vs prismatic without branching on
        # python data inside the graph.
        self.register_buffer(
            "origin_translation", torch.from_numpy(origin[:, :3, 3].copy())
        )
        self.register_buffer(
            "origin_rotation", torch.from_numpy(origin[:, :3, :3].copy())
        )

    # ---- joint vector handling -------------------------------------------

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, n_independent) independent joints to (B, nq) actuated joints.

        Clipping matches ``BaseOptimizer.expand_to_full_qpos``; see
        ``KinematicChain.expand_to_full`` for why L20 needs it.
        """
        dtype = x.dtype
        batch = x.shape[0]
        q = x.new_zeros((batch, self.chain.nq))
        q = q.index_copy(1, self.indep_slots, x)
        if self.mimic_slots.numel():
            derived = (
                x.index_select(1, self.mimic_src_slots)
                * self.mimic_mult.to(dtype)
                + self.mimic_offset.to(dtype)
            )
            q = q.index_copy(1, self.mimic_slots, derived)
        return torch.clamp(q, self.joint_lower.to(dtype), self.joint_upper.to(dtype))

    def to_independent(self, q: torch.Tensor) -> torch.Tensor:
        """Extract independent joints from a full qpos vector."""
        return q.index_select(1, self.indep_slots)

    # ---- kinematics -------------------------------------------------------

    def link_transforms(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map link name -> (B, 4, 4) world transforms."""
        if q.dim() != 2 or q.shape[1] != self.chain.nq:
            raise ValueError(
                f"Expected qpos shape (B, {self.chain.nq}), got {tuple(q.shape)}"
            )
        batch = q.shape[0]
        dtype, device = q.dtype, q.device

        transforms: dict[str, torch.Tensor] = {
            self.chain.root_link: torch.eye(4, dtype=dtype, device=device)
            .expand(batch, 4, 4)
            .contiguous()
        }

        for ji in self.order:
            joint = self.chain.joints[ji]
            slot = int(self.q_slot[ji])

            # Cast to the incoming dtype so the module works whether the
            # caller feeds float32 (training) or float64 (validation).
            R0 = self.origin_rotation[ji].to(dtype)  # (3, 3)
            p0 = self.origin_translation[ji].to(dtype)  # (3,)

            if slot < 0:
                rot = R0.expand(batch, 3, 3)
                trans = p0.expand(batch, 3)
            else:
                value = q[:, slot]
                if bool(self.is_prismatic[ji]):
                    rot = R0.expand(batch, 3, 3)
                    trans = p0 + (R0 @ self.axis[ji]) * value[:, None]
                else:
                    # URDF: T_parent_child = T_origin @ R_joint(axis, q), so
                    # the joint rotation acts in the joint frame, on the right.
                    rot = R0 @ _rodrigues(self.axis[ji].to(dtype), value)
                    trans = p0.expand(batch, 3)

            local = torch.zeros(batch, 4, 4, dtype=dtype, device=device)
            local[:, :3, :3] = rot
            local[:, :3, 3] = trans
            local[:, 3, 3] = 1.0

            transforms[joint.child] = transforms[joint.parent] @ local

        return transforms

    def points(
        self,
        q: torch.Tensor,
        link_names: list[str],
        local_offsets: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """World positions of link-local offsets, shape (B, N, 3)."""
        transforms = self.link_transforms(q)
        n = len(link_names)
        if local_offsets is None:
            offsets = q.new_zeros((n, 3))
        else:
            offsets = torch.as_tensor(
                local_offsets, dtype=q.dtype, device=q.device
            ).reshape(n, 3)

        stacked = torch.stack([transforms[name] for name in link_names], dim=1)
        rot = stacked[:, :, :3, :3]
        trans = stacked[:, :, :3, 3]
        return trans + torch.einsum("bnij,nj->bni", rot, offsets)


__all__ = ["TorchFK"]
