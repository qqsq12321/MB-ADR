"""Map between URDF joint order and a MuJoCo model's qpos layout.

ElementTree is not an option here: MuJoCo supports ``<include file="..."/>``
and ElementTree does not expand it, so a perfectly normal single-file-looking
MJCF (e.g. the Shadow hand's ``scene_right.xml``) parses to zero joints. The
model MuJoCo itself builds is the only reliable source of joint names.

Only hinge and slide joints are considered. Fixed joints contribute no DOF, and
multi-DOF joints are not supported by ``RobotWrapper`` either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MjcfMapping:
    """Resolved correspondence between a URDF joint vector and MJCF qpos."""

    mjcf_names: list[str]
    src: np.ndarray  # src[i] indexes the URDF-order vector; -1 = not driven
    nq: int
    defaults: np.ndarray  # (nq,) model default qpos, used for undriven joints
    unmapped: list[str]

    def to_qpos(self, q_urdf: np.ndarray) -> np.ndarray:
        """URDF-order (nq_urdf,) -> MuJoCo qpos (model.nq,).

        Joints with no URDF counterpart (a wrist built into the scene file, for
        example) keep the model's default value rather than being zeroed.
        """
        q = self.defaults.copy()
        urdf = np.asarray(q_urdf, dtype=np.float64)
        for slot, index in enumerate(self.src):
            if index >= 0:
                q[slot] = urdf[index]
        return q


def build_mapping(mjcf, urdf_joint_names: list[str]) -> MjcfMapping:
    """Resolve which URDF joint feeds each MuJoCo joint, by name.

    Args:
        mjcf: path to an MJCF file, or an already compiled ``MjModel`` (e.g.
            the scene from :func:`mbadr.scene.build_scene`).
        urdf_joint_names: joint names in this project's URDF order.

    MuJoCo models routinely contain DOF the URDF does not: the Shadow hand's
    scene file adds two wrist joints around the fingers. Those are reported in
    ``unmapped`` and left at their default pose.
    """
    import mujoco

    if isinstance(mjcf, mujoco.MjModel):
        model = mjcf
    else:
        model = mujoco.MjModel.from_xml_path(str(mjcf))

    names: list[str] = []
    slots: list[int] = []
    for j in range(model.njnt):
        if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not name:
            continue
        names.append(name)
        # qpos address; equals the joint index for hinge/slide-only models but
        # read from the model so multi-dof joints cannot silently misalign.
        slots.append(int(model.jnt_qposadr[j]))

    urdf_index = {n: i for i, n in enumerate(urdf_joint_names)}
    src = np.array([urdf_index.get(n, -1) for n in names], dtype=np.int64)
    unmapped = [n for n, i in zip(names, src) if i < 0]

    # ``qpos0`` is the model's reference configuration, used for joints this
    # URDF does not drive (e.g. a wrist built into the Shadow scene file).
    defaults = np.array(model.qpos0, dtype=np.float64).copy()
    return MjcfMapping(
        mjcf_names=names, src=src, nq=model.nq, defaults=defaults, unmapped=unmapped
    )


__all__ = ["MjcfMapping", "build_mapping"]
