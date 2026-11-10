"""NumPy reference forward kinematics.

This exists purely as the correctness oracle: it is validated against Pinocchio
(see ``scripts/verify_fk.py``) and then the Torch implementation in
``fk_torch.py`` is validated against *it*. Keeping the two implementations
structurally identical makes the comparison meaningful.
"""

from __future__ import annotations

import numpy as np

from .chain import KinematicChain


def topological_joint_order(chain: KinematicChain) -> list[int]:
    """Joint indices ordered so every parent link is resolved before its child."""
    resolved = {chain.root_link}
    remaining = list(range(len(chain.joints)))
    order: list[int] = []
    while remaining:
        progressed = False
        still: list[int] = []
        for ji in remaining:
            if chain.joints[ji].parent in resolved:
                order.append(ji)
                resolved.add(chain.joints[ji].child)
                progressed = True
            else:
                still.append(ji)
        if not progressed:
            raise ValueError("URDF tree is disconnected or cyclic")
        remaining = still
    return order


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about a unit ``axis``."""
    kx, ky, kz = axis
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


class NumpyFK:
    """Forward kinematics for a :class:`KinematicChain`, in meters."""

    def __init__(self, chain: KinematicChain):
        self.chain = chain
        self.order = topological_joint_order(chain)
        # Actuated joints occupy slots in the nq-length qpos vector; fixed
        # joints have no slot.
        self.q_slot: dict[int, int] = {}
        slot = 0
        for ji, joint in enumerate(chain.joints):
            if joint.is_actuated:
                self.q_slot[ji] = slot
                slot += 1
        assert slot == chain.nq

    def link_transforms(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Map link name -> (4, 4) world transform."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.chain.nq,):
            raise ValueError(f"Expected qpos shape ({self.chain.nq},), got {q.shape}")

        transforms = {self.chain.root_link: np.eye(4, dtype=np.float64)}
        for ji in self.order:
            joint = self.chain.joints[ji]
            local = joint.origin_T.copy()
            if joint.is_actuated:
                value = q[self.q_slot[ji]]
                if joint.type == "prismatic":
                    local[:3, 3] = local[:3, 3] + local[:3, :3] @ (joint.axis * value)
                else:  # revolute / continuous
                    local[:3, :3] = local[:3, :3] @ axis_angle_to_matrix(
                        joint.axis, value
                    )
            transforms[joint.child] = transforms[joint.parent] @ local
        return transforms

    def points(
        self,
        q: np.ndarray,
        link_names: list[str],
        local_offsets: np.ndarray | None = None,
    ) -> np.ndarray:
        """World positions of ``local_offsets`` expressed in each link's frame.

        ``p = T.translation + T.rotation @ offset``, in metres.
        """
        transforms = self.link_transforms(q)
        n = len(link_names)
        if local_offsets is None:
            offsets = np.zeros((n, 3), dtype=np.float64)
        else:
            offsets = np.asarray(local_offsets, dtype=np.float64).reshape(n, 3)

        out = np.zeros((n, 3), dtype=np.float64)
        for i, name in enumerate(link_names):
            if name not in transforms:
                raise ValueError(f"Unknown link '{name}'")
            T = transforms[name]
            out[i] = T[:3, 3] + T[:3, :3] @ offsets[i]
        return out

    def points_jacobian(
        self,
        q: np.ndarray,
        link_names: list[str],
        local_offsets: np.ndarray | None = None,
    ) -> np.ndarray:
        """Position Jacobian of the task points, (num_points, 3, nq).

        For a point rigidly attached to a link, only the joints on that link's
        path to the root contribute:

            revolute:   dp/dq_k = axis_world_k x (p - p_joint_k)
            prismatic:  dp/dq_k = axis_world_k

        This is an independent implementation of the same quantity Torch's
        autograd produces, which is what makes it useful as a cross-check.
        """
        q = np.asarray(q, dtype=np.float64)
        transforms = self.link_transforms(q)
        n = len(link_names)
        offsets = (
            np.zeros((n, 3), dtype=np.float64)
            if local_offsets is None
            else np.asarray(local_offsets, dtype=np.float64).reshape(n, 3)
        )

        jac = np.zeros((n, 3, self.chain.nq), dtype=np.float64)
        for i, name in enumerate(link_names):
            if name not in transforms:
                raise ValueError(f"Unknown link '{name}'")
            T = transforms[name]
            point = T[:3, 3] + T[:3, :3] @ offsets[i]

            for ji in self.chain.chain_to(name):
                joint = self.chain.joints[ji]
                if not joint.is_actuated:
                    continue
                slot = self.q_slot[ji]
                parent_T = transforms[joint.parent]
                # Joint axis in world coordinates, and the joint origin.
                axis_world = parent_T[:3, :3] @ joint.origin_T[:3, :3] @ joint.axis
                joint_origin = parent_T[:3, 3] + parent_T[:3, :3] @ joint.origin_T[:3, 3]
                if joint.type == "prismatic":
                    jac[i, :, slot] = axis_world
                else:
                    jac[i, :, slot] = np.cross(axis_world, point - joint_origin)
        return jac


__all__ = ["NumpyFK", "topological_joint_order", "axis_angle_to_matrix"]
