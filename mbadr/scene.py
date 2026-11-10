"""A uniform MuJoCo viewing scene for every hand.

The bundled MJCFs disagree on everything that affects how a demo looks: some
ship a skybox, floor and lights in their own colours, some ship none at all, and
each hand's base frame points its fingers and palm in a different direction.

:func:`build_scene` loads a hand's MJCF without modifying it on disk and
returns a model in which

* the hand's own floor, lights, skybox and ``<visual>`` settings are replaced by
  one shared background, and
* the whole hand is wrapped in a fixed ``mbadr_mount`` body whose rotation puts
  the fingers along +Z and the palm facing -Y, i.e. towards the default camera,
  lifted clear of the floor and centred on the origin.

The orientation is derived from the model's own geometry rather than a
per-hand table, so a new hand gets the same treatment automatically:

* *up* is the direction from the hand's base to the centroid of its leaf
  (fingertip) bodies at the neutral pose;
* *palm normal* is the direction the fingertips move when every finger joint is
  driven towards the larger end of its range — fingers curl towards the palm.

Wrapping adds no joints, so qpos layout and joint order are unchanged and
:mod:`mbadr.mjcf` mappings stay valid.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MOUNT_BODY = "mbadr_mount"

# Default free-camera view: straight at the palm, no tilt.
CAMERA_AZIMUTH = 90.0
CAMERA_ELEVATION = 0.0

_BACKGROUND = """
<mujoco>
  <visual>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="4096"/>
    <global azimuth="{az}" elevation="{el}"/>
  </visual>
  <asset>
    <texture name="mbadr_sky" type="skybox" builtin="gradient"
             rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture name="mbadr_grid" type="2d" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="mbadr_floor" texture="mbadr_grid" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light name="mbadr_key" pos="0 -0.6 1.2" dir="0 0.4 -1" directional="true"/>
    <light name="mbadr_fill" pos="0 0.6 0.8" dir="0 -0.5 -1" diffuse="0.3 0.3 0.3"
           directional="true" castshadow="false"/>
    <geom name="mbadr_floor" type="plane" size="0 0 0.05" material="mbadr_floor"
          contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""


@dataclass
class Scene:
    """Compiled scene plus the camera framing for it."""

    model: object  # mujoco.MjModel
    lookat: np.ndarray
    distance: float
    azimuth: float = CAMERA_AZIMUTH
    elevation: float = CAMERA_ELEVATION

    def apply_camera(self, cam) -> None:
        """Point a viewer/renderer ``MjvCamera`` at the hand."""
        cam.lookat[:] = self.lookat
        cam.distance = self.distance
        cam.azimuth = self.azimuth
        cam.elevation = self.elevation


def _flatten(mjcf_path: Path) -> ET.Element:
    """Expand includes and make every asset path absolute."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    spec.compile()
    root = ET.fromstring(spec.to_xml())

    base = Path(mjcf_path).resolve().parent
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    for key in ("meshdir", "texturedir"):
        rel = compiler.get(key) or ""
        compiler.set(key, str((base / rel).resolve()) + "/")
    compiler.attrib.pop("assetdir", None)
    return root


def _strip_background(root: ET.Element) -> None:
    """Remove the hand file's own floor, lights, skybox and visual settings."""
    for visual in root.findall("visual"):
        root.remove(visual)

    removed_textures: set[str] = set()
    for asset in root.findall("asset"):
        for tex in list(asset.findall("texture")):
            if tex.get("type") == "skybox":
                asset.remove(tex)
    worldbody = root.find("worldbody")
    floor_materials: set[str] = set()
    for child in list(worldbody):
        if child.tag == "light":
            worldbody.remove(child)
        elif child.tag == "geom" and child.get("type") == "plane":
            if child.get("material"):
                floor_materials.add(child.get("material"))
            worldbody.remove(child)

    # Drop the floor's material and its texture once nothing else uses them.
    for asset in root.findall("asset"):
        for mat in list(asset.findall("material")):
            if mat.get("name") in floor_materials:
                if mat.get("texture"):
                    removed_textures.add(mat.get("texture"))
                asset.remove(mat)
    still_used = {m.get("texture") for m in root.iter("material")}
    for asset in root.findall("asset"):
        for tex in list(asset.findall("texture")):
            if tex.get("name") in removed_textures and tex.get("name") not in still_used:
                asset.remove(tex)


def _wrap_in_mount(root: ET.Element, quat=None, pos=None) -> None:
    """Move every remaining worldbody element into one fixed mount body."""
    worldbody = root.find("worldbody")
    mount = ET.Element("body", name=MOUNT_BODY)
    if quat is not None:
        mount.set("quat", " ".join(f"{v:.9f}" for v in quat))
    if pos is not None:
        mount.set("pos", " ".join(f"{v:.9f}" for v in pos))
    for child in list(worldbody):
        worldbody.remove(child)
        mount.append(child)
    worldbody.append(mount)


def _add_background(root: ET.Element) -> None:
    bg = ET.fromstring(
        _BACKGROUND.format(az=CAMERA_AZIMUTH, el=CAMERA_ELEVATION)
    )
    root.insert(0, bg.find("visual"))
    root.append(bg.find("asset"))
    worldbody = root.find("worldbody")
    for i, child in enumerate(bg.find("worldbody")):
        worldbody.insert(i, child)


def _compile(root: ET.Element):
    import mujoco

    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _hand_geometry(
    model,
    driven_joints: set[str] | None,
    link1_names: list[str] | None,
    link4_names: list[str] | None,
):
    """(base, up, side, palm normal) of the hand, in the model's world frame.

    ``up`` and ``side`` come straight from the knuckle geometry at rest — the
    average proximal->distal direction of the non-thumb fingers, and the line
    from the index knuckle to the pinky knuckle — which is far steadier than
    deriving an axis from whole-hand centroids (the thumb and palm thickness
    skew those). Flexing the fingers is used only to pick which side of that
    plane is the palm, not to measure any angle.
    """
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mount = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MOUNT_BODY)
    children = {b: [] for b in range(model.nbody)}
    for b in range(1, model.nbody):
        children[int(model.body_parentid[b])].append(b)
    leaves = [b for b in range(model.nbody) if b not in (0, mount) and not children[b]]
    top = [b for b in children[mount]] if mount >= 0 else children[0]
    base = data.xpos[top].mean(axis=0)

    name2id = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    link1 = [name2id(n) for n in (link1_names or [])]
    link4 = [name2id(n) for n in (link4_names or [])]
    have_fingers = len(link1) == len(link4) and len(link1) >= 2 and all(b >= 0 for b in link1 + link4)

    if have_fingers:
        # Finger 0 is the thumb; everything else sets the knuckle line.
        non_thumb1 = link1[1:]
        non_thumb4 = link4[1:]
        up = (data.xpos[non_thumb4] - data.xpos[non_thumb1]).mean(axis=0)
        up /= np.linalg.norm(up)
        side_raw = data.xpos[link1[-1]] - data.xpos[link1[1]]  # index -> pinky
        flex_target = link4
    else:
        # Fallback for a hand with no usable link1/link4 names: whole-hand
        # centroid, least accurate but better than failing outright.
        tips_rest = data.xpos[leaves].copy()
        up = tips_rest.mean(axis=0) - base
        up /= np.linalg.norm(up)
        side_raw = None
        flex_target = leaves

    if side_raw is not None:
        side = side_raw - np.dot(side_raw, up) * up
        side /= np.linalg.norm(side)
    else:
        side = None

    # Curl every finger joint towards the larger end of its range, to read off
    # which way the palm faces.
    rest = data.xpos[flex_target].copy()
    q = model.qpos0.copy()
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if driven_joints is not None and name not in driven_joints:
            continue  # e.g. Shadow's wrist joints, not part of the hand
        lo, hi = model.jnt_range[j]
        q[model.jnt_qposadr[j]] = 0.6 * (hi if abs(hi) >= abs(lo) else lo)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    shift = (data.xpos[flex_target] - rest).mean(axis=0)

    if side is None:
        normal = shift - np.dot(shift, up) * up
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            raise ValueError("could not determine the palm direction from flexion")
        return base, up, None, normal / norm

    # The third axis is fixed by up/side; flexion only supplies its sign, so a
    # noisy curl measurement can never introduce roll. Fingers curl towards the
    # palm, so the palm normal should point the same way as the curl shift —
    # flip when it doesn't, not when it does.
    normal = np.cross(side, up)
    if np.dot(shift, normal) < 0:
        normal = -normal
        side = -side
    return base, up, side, normal


def _hand_bodies(model, driven_joints: set[str] | None) -> set[int]:
    """Bodies that make up the hand itself, for framing the camera.

    Scene files sometimes carry an arm segment above the hand (Shadow has a
    forearm and two wrist joints the retargeter does not drive). Anything that
    is a strict ancestor of the deepest body carrying an undriven joint is arm,
    not hand, and is excluded so the camera frames the fingers.
    """
    import mujoco

    if driven_joints is None:
        return set(range(1, model.nbody))

    def depth(b: int) -> int:
        d = 0
        while b != 0:
            b = int(model.body_parentid[b])
            d += 1
        return d

    undriven_bodies = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name not in driven_joints:
            undriven_bodies.append(int(model.jnt_bodyid[j]))
    if not undriven_bodies:
        return set(range(1, model.nbody))

    deepest = max(undriven_bodies, key=depth)
    excluded = set()
    b = int(model.body_parentid[deepest])
    while b != 0:
        excluded.add(b)
        b = int(model.body_parentid[b])
    return {b for b in range(1, model.nbody) if b not in excluded}


def _rotation_to_quat(R: np.ndarray) -> np.ndarray:
    import mujoco

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.reshape(-1))
    return quat


def build_scene(
    mjcf_path,
    driven_joints: list[str] | None = None,
    link1_names: list[str] | None = None,
    link4_names: list[str] | None = None,
) -> Scene:
    """Compile ``mjcf_path`` into the shared scene, upright and palm-forward.

    Args:
        mjcf_path: the hand's MJCF (left untouched on disk).
        driven_joints: joint names the retargeter drives; other joints (wrists
            built into a scene file) are ignored when estimating the palm side.
        link1_names, link4_names: proximal/distal body names per finger, thumb
            first (``HandSpec.link1_links`` / ``link4_links``). These pin the
            up and side axes to the knuckle line, which is far more stable than
            measuring from the whole hand; omit only if a hand has no such
            names, which falls back to a cruder whole-hand estimate.
    """
    import mujoco

    driven = set(driven_joints) if driven_joints is not None else None

    # Pass 1: measure the hand in its native frame.
    root = _flatten(Path(mjcf_path))
    _strip_background(root)
    _wrap_in_mount(root)
    native = _compile(root)
    base, up, side, normal = _hand_geometry(native, driven, link1_names, link4_names)

    # Rotation taking the native frame to: up -> +Z, palm normal -> -Y, with the
    # knuckle line -> X when available so roll is pinned by real geometry
    # rather than left over from a cross product of two noisy vectors.
    z = up
    if side is not None:
        x = side
        y = np.cross(z, x)
    else:
        y = -normal
        x = np.cross(y, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
    R = np.stack([x, y, z])  # rows: native axes expressed in the new frame
    quat = _rotation_to_quat(R)

    # Pass 2: rotate, then lift above the floor and centre on the origin.
    root = _flatten(Path(mjcf_path))
    _strip_background(root)
    _wrap_in_mount(root, quat=quat)
    rotated = _compile(root)
    data = mujoco.MjData(rotated)
    mujoco.mj_forward(rotated, data)
    geoms = [g for g in range(rotated.ngeom) if rotated.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
    low = data.geom_xpos[geoms] - rotated.geom_rbound[geoms, None]
    high = data.geom_xpos[geoms] + rotated.geom_rbound[geoms, None]

    # Everything (arm included) must clear the floor, but the camera frames
    # only the hand itself.
    hand = _hand_bodies(rotated, driven)
    hand_geoms = [i for i, g in enumerate(geoms) if int(rotated.geom_bodyid[g]) in hand]
    h_low = low[hand_geoms].min(axis=0)
    h_high = high[hand_geoms].max(axis=0)
    centre = 0.5 * (h_low + h_high)
    lift = 0.02 - low[:, 2].min()
    offset = np.array([-centre[0], -centre[1], lift])

    root = _flatten(Path(mjcf_path))
    _strip_background(root)
    _wrap_in_mount(root, quat=quat, pos=offset)
    _add_background(root)
    model = _compile(root)

    height = h_high[2] - h_low[2]
    lookat = np.array([0.0, 0.0, centre[2] + lift])
    distance = max(0.35, 1.6 * height)
    return Scene(model=model, lookat=lookat, distance=distance)


__all__ = ["Scene", "build_scene", "MOUNT_BODY", "CAMERA_AZIMUTH", "CAMERA_ELEVATION"]
