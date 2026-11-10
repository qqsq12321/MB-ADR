"""Framework-agnostic URDF kinematic chain parsing.

Parses a URDF into flat arrays describing the kinematic tree, so that both the
NumPy reference FK (for validation against Pinocchio) and the differentiable
Torch FK (for training) can be built from the exact same data.

The joint ordering produced here is the URDF document order. Pinocchio uses its
own ordering, so callers that exchange ``qpos`` vectors with it must remap via
:meth:`KinematicChain.permutation_to` (or the named helpers
:meth:`~KinematicChain.to_target_order` and
:meth:`~KinematicChain.reorder_from`).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis rpy convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


@dataclass
class Joint:
    name: str
    type: str  # 'revolute' | 'continuous' | 'prismatic' | 'fixed'
    parent: str
    child: str
    origin_T: np.ndarray  # (4, 4) fixed parent->joint transform
    axis: np.ndarray  # (3,) unit axis in joint frame; zeros for fixed
    lower: float
    upper: float
    # Populated when this joint carries a <mimic> tag.
    mimic_source: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0

    @property
    def is_actuated(self) -> bool:
        return self.type != "fixed"

    @property
    def is_mimic(self) -> bool:
        return self.mimic_source is not None


@dataclass
class KinematicChain:
    """Flat description of a URDF kinematic tree.

    Attributes:
        joints: all joints in URDF document order.
        joint_names: names of actuated joints (``nq`` of them), URDF order.
        independent_names: actuated joints that are not mimic-driven.
        link_parent_joint: link name -> index into ``joints`` of the joint whose
            child it is. The root link is absent.
    """

    joints: list[Joint]
    root_link: str
    urdf_path: str = ""
    link_parent_joint: dict[str, int] = field(default_factory=dict)

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_urdf(cls, urdf_path: str | Path) -> "KinematicChain":
        urdf_path = Path(urdf_path)
        root = ET.parse(urdf_path).getroot()

        joints: list[Joint] = []
        for elem in root.iter("joint"):
            # Skip <joint> elements nested inside <transmission>/<gazebo>,
            # which lack parent/child links.
            parent_elem = elem.find("parent")
            child_elem = elem.find("child")
            if parent_elem is None or child_elem is None:
                continue

            origin = elem.find("origin")
            if origin is not None:
                xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
            else:
                xyz = np.zeros(3, dtype=np.float64)
                rpy = np.zeros(3, dtype=np.float64)
            origin_T = np.eye(4, dtype=np.float64)
            origin_T[:3, :3] = rpy_to_matrix(*rpy)
            origin_T[:3, 3] = xyz

            jtype = elem.get("type", "fixed")
            axis_elem = elem.find("axis")
            if jtype == "fixed" or axis_elem is None:
                axis = np.zeros(3, dtype=np.float64)
            else:
                axis = np.fromstring(axis_elem.get("xyz", "1 0 0"), sep=" ", dtype=np.float64)
                norm = np.linalg.norm(axis)
                if norm < 1e-12:
                    raise ValueError(f"Joint '{elem.get('name')}' has a degenerate axis")
                axis = axis / norm

            limit_elem = elem.find("limit")
            if limit_elem is not None:
                lower = float(limit_elem.get("lower", "0.0"))
                upper = float(limit_elem.get("upper", "0.0"))
            elif jtype == "continuous":
                lower, upper = -np.pi, np.pi
            else:
                lower, upper = 0.0, 0.0

            mimic_elem = elem.find("mimic")
            if mimic_elem is not None:
                mimic_source = mimic_elem.get("joint")
                mimic_multiplier = float(mimic_elem.get("multiplier", "1.0"))
                mimic_offset = float(mimic_elem.get("offset", "0.0"))
            else:
                mimic_source, mimic_multiplier, mimic_offset = None, 1.0, 0.0

            joints.append(
                Joint(
                    name=elem.get("name", f"joint_{len(joints)}"),
                    type=jtype,
                    parent=parent_elem.get("link", ""),
                    child=child_elem.get("link", ""),
                    origin_T=origin_T,
                    axis=axis,
                    lower=lower,
                    upper=upper,
                    mimic_source=mimic_source,
                    mimic_multiplier=mimic_multiplier,
                    mimic_offset=mimic_offset,
                )
            )

        children = {j.child for j in joints}
        parents = {j.parent for j in joints}
        roots = parents - children
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one root link, found {sorted(roots)}")

        chain = cls(
            joints=joints,
            root_link=roots.pop(),
            urdf_path=str(urdf_path),
            link_parent_joint={j.child: i for i, j in enumerate(joints)},
        )
        chain._validate()
        return chain

    def _validate(self) -> None:
        names = [j.name for j in self.joints]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate joint names in URDF")
        actuated = set(self.joint_names)
        for j in self.joints:
            if j.is_mimic and j.mimic_source not in actuated:
                raise ValueError(
                    f"Joint '{j.name}' mimics unknown joint '{j.mimic_source}'"
                )
        # A mimic joint driven by another mimic joint would need an ordered
        # resolve pass; reject it rather than silently mis-expanding.
        mimic_names = {j.name for j in self.joints if j.is_mimic}
        for j in self.joints:
            if j.is_mimic and j.mimic_source in mimic_names:
                raise ValueError(
                    f"Chained mimic not supported: '{j.name}' -> '{j.mimic_source}'"
                )
        for name in self.link_names:
            self.chain_to(name)  # raises if the path to the root is broken

    # ---- joint bookkeeping ------------------------------------------------

    @property
    def joint_names(self) -> list[str]:
        """Actuated joint names (length ``nq``), URDF document order."""
        return [j.name for j in self.joints if j.is_actuated]

    @property
    def nq(self) -> int:
        return len(self.joint_names)

    @property
    def independent_names(self) -> list[str]:
        return [j.name for j in self.joints if j.is_actuated and not j.is_mimic]

    @property
    def num_independent(self) -> int:
        return len(self.independent_names)

    @property
    def link_names(self) -> list[str]:
        return [self.root_link] + [j.child for j in self.joints]

    def joint_limits(self) -> np.ndarray:
        """(nq, 2) lower/upper limits in actuated-joint order."""
        return np.array(
            [[j.lower, j.upper] for j in self.joints if j.is_actuated],
            dtype=np.float64,
        )

    def independent_limits(self) -> np.ndarray:
        """(num_independent, 2) limits for the independent joints only."""
        return np.array(
            [
                [j.lower, j.upper]
                for j in self.joints
                if j.is_actuated and not j.is_mimic
            ],
            dtype=np.float64,
        )

    def mimic_spec(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Linear map from independent joints to the full actuated vector.

        Returns ``(indep_slots, mimic_slots, mimic_src_slots, mult, offset)``
        packed as four arrays, where slots index into the ``nq`` actuated
        vector and ``src_slots`` indexes into the independent vector:

            q[indep_slots]  = x
            q[mimic_slots]  = x[mimic_src_slots] * mult + offset
        """
        qidx = {name: i for i, name in enumerate(self.joint_names)}
        xidx = {name: i for i, name in enumerate(self.independent_names)}

        indep_slots = np.array(
            [qidx[n] for n in self.independent_names], dtype=np.int64
        )
        mimic_joints = [j for j in self.joints if j.is_actuated and j.is_mimic]
        mimic_slots = np.array([qidx[j.name] for j in mimic_joints], dtype=np.int64)
        mimic_src_slots = np.array(
            [xidx[j.mimic_source] for j in mimic_joints], dtype=np.int64
        )
        mult = np.array([j.mimic_multiplier for j in mimic_joints], dtype=np.float64)
        offset = np.array([j.mimic_offset for j in mimic_joints], dtype=np.float64)
        return indep_slots, mimic_slots, mimic_src_slots, mult, offset

    def expand_to_full(self, x: np.ndarray, clip: bool = True) -> np.ndarray:
        """Expand independent joint values to the full actuated vector.

        Args:
            x: (num_independent,) independent joint values.
            clip: clamp to per-joint limits, matching
                ``BaseOptimizer.expand_to_full_qpos``. This is not optional in
                practice for L20: ``index_pip`` tops out at 1.75 rad while its
                mimic-driven ``index_dip`` tops out at 1.55, and
                1.75 * 0.89 = 1.5575 overshoots the distal limit, so the raw
                chain is unreachable by roughly 0.008 rad. The real hand stops,
                hence the clamp.
        """
        x = np.asarray(x, dtype=np.float64)
        indep_slots, mimic_slots, src_slots, mult, offset = self.mimic_spec()
        q = np.zeros(self.nq, dtype=np.float64)
        q[indep_slots] = x
        if mimic_slots.size:
            q[mimic_slots] = x[src_slots] * mult + offset
        if clip:
            lim = self.joint_limits()
            q = np.clip(q, lim[:, 0], lim[:, 1])
        return q

    # ---- topology ---------------------------------------------------------

    def chain_to(self, link_name: str) -> list[int]:
        """Indices into ``self.joints`` from the root down to ``link_name``."""
        if link_name == self.root_link:
            return []
        path: list[int] = []
        current = link_name
        seen: set[str] = set()
        while current != self.root_link:
            if current in seen:
                raise ValueError(f"Cycle in URDF tree at link '{current}'")
            seen.add(current)
            if current not in self.link_parent_joint:
                raise ValueError(f"Link '{link_name}' is not connected to the root")
            ji = self.link_parent_joint[current]
            path.append(ji)
            current = self.joints[ji].parent
        path.reverse()
        return path

    def permutation_to(self, target_names: list[str]) -> np.ndarray:
        """Index array ``p`` such that ``q_urdf[p] == q_target``.

        Use this to convert a qpos vector in this chain's URDF order into
        another tool's joint order (e.g. Pinocchio's ``dof_joint_names``).
        Note the direction: this maps *URDF -> target*. Going the other way
        needs :meth:`reorder_from`; using this array in both directions is a
        silent, plausible-looking bug, so prefer the two named helpers.
        """
        own = {name: i for i, name in enumerate(self.joint_names)}
        missing = [n for n in target_names if n not in own]
        if missing:
            raise ValueError(f"Joints absent from URDF: {missing}")
        if len(target_names) != self.nq:
            raise ValueError(
                f"Expected {self.nq} target names, got {len(target_names)}"
            )
        return np.array([own[n] for n in target_names], dtype=np.int64)

    def to_target_order(self, q_urdf: np.ndarray, target_names: list[str]) -> np.ndarray:
        """Convert a URDF-order qpos into ``target_names`` order."""
        return np.asarray(q_urdf)[self.permutation_to(target_names)]

    def reorder_from(self, q_target: np.ndarray, target_names: list[str]) -> np.ndarray:
        """Convert a qpos in ``target_names`` order into this chain's URDF order."""
        perm = self.permutation_to(target_names)
        q_urdf = np.empty_like(np.asarray(q_target), dtype=np.float64)
        q_urdf[perm] = q_target
        return q_urdf

    def independent_from_full(self, q_full: np.ndarray) -> np.ndarray:
        """Select the independent joints from a full URDF-order qpos."""
        return np.asarray(q_full)[self.mimic_spec()[0]]


__all__ = ["Joint", "KinematicChain", "rpy_to_matrix"]
