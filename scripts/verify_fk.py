"""Validate the NumPy FK against Pinocchio, an independent third-party oracle.

Everything downstream rests on this: if FK disagrees, every loss value and
gradient is wrong. Pinocchio is used only as an outside reference — it is a
library dependency, not part of this project's own models.

    python scripts/verify_fk.py --robot all
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mbadr.chain import KinematicChain
from mbadr.config import (
    ROBOT_CONFIGS,
    HandSpec,
    default_config_path,
    resolve_robot_name,
)
from mbadr.fk_numpy import NumpyFK

TOL_POS = 1e-9  # metres; both sides are float64 doing the same products

DEFAULT_HANDS = ["l20", "shadow", "wuji", "sharpa", "gaia", "inspire"]


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"  ok: {msg}")


class PinocchioFK:
    """Minimal Pinocchio wrapper, used purely as a cross-check oracle."""

    def __init__(self, urdf_path):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        if self.model.nv != self.model.nq:
            raise NotImplementedError("multi-DOF joints are not supported")
        self.joint_names = [
            self.model.names[i]
            for i in range(1, self.model.njoints)
            if self.model.joints[i].nq > 0
        ]

    def limits(self) -> np.ndarray:
        return np.stack(
            [self.model.lowerPositionLimit, self.model.upperPositionLimit], axis=1
        )

    def frame_id(self, name: str) -> int:
        return self.model.getFrameId(name, self.pin.BODY)

    def points(self, q, frame_ids, offsets) -> np.ndarray:
        self.pin.forwardKinematics(self.model, self.data, np.asarray(q))
        self.pin.updateFramePlacements(self.model, self.data)
        out = np.zeros((len(frame_ids), 3))
        for i, fid in enumerate(frame_ids):
            T = self.data.oMf[fid]
            out[i] = T.translation + T.rotation @ np.asarray(offsets)[i]
        return out

    def position_jacobian(self, q, frame_ids, offsets) -> np.ndarray:
        self.pin.computeJointJacobians(self.model, self.data, np.asarray(q))
        self.pin.updateFramePlacements(self.model, self.data)
        jac = np.zeros((len(frame_ids), 3, self.model.nq))
        for i, fid in enumerate(frame_ids):
            J = self.pin.getFrameJacobian(
                self.model, self.data, fid, self.pin.LOCAL
            )
            R = self.data.oMf[fid].rotation
            off = np.asarray(offsets)[i]
            if np.allclose(off, 0.0):
                jac[i] = R @ J[:3, :]
            else:
                skew = np.array(
                    [
                        [0.0, -off[2], off[1]],
                        [off[2], 0.0, -off[0]],
                        [-off[1], off[0], 0.0],
                    ]
                )
                jac[i] = R @ (J[:3, :] - skew @ J[3:, :])
        return jac


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="l20",
                        help="hand name or alias; 'all' checks every adapted hand")
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    args = parser.parse_args()

    names = DEFAULT_HANDS if args.robot == "all" else [args.robot]
    failures = 0
    for name in names:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        try:
            check_hand(name, args.hand)
        except SystemExit:
            failures += 1
            print(f"  {name} FAILED")
    if failures:
        print(f"\n{failures}/{len(names)} hands failed")
        raise SystemExit(1)
    print(f"\nAll {len(names)} hand(s) passed.")


def check_hand(name: str, hand_side: str) -> None:
    from mbadr.config import _mirror_name

    robot_type = resolve_robot_name(name)
    spec = HandSpec.from_name(name, hand_side)
    print(f"URDF: {spec.urdf_path}")
    if not spec.urdf_path.exists():
        _fail(f"URDF missing: {spec.urdf_path}")

    chain = KinematicChain.from_urdf(spec.urdf_path)
    fk = NumpyFK(chain)
    pin_fk = PinocchioFK(spec.urdf_path)

    # --- 1. spec matches the bundled robot table --------------------------
    print(f"\n[1] HandSpec vs ROBOT_CONFIGS['{robot_type}']")
    ref = ROBOT_CONFIGS[robot_type]

    def mirror(seq):
        return [_mirror_name(n, robot_type, hand_side) for n in seq]

    checks = [
        ("origin_link", spec.origin_link, mirror([ref["origin_link"]])[0]),
        ("tip_links", list(spec.tip_links), mirror(ref["tip_links"])),
        ("link1_names", list(spec.link1_links), mirror(ref["link1_names"])),
        ("link3_names", list(spec.link3_links), mirror(ref["link3_names"])),
        ("link4_names", list(spec.link4_links), mirror(ref["link4_names"])),
        ("num_fingers", spec.num_fingers, ref["num_fingers"]),
    ]
    for label, mine, theirs in checks:
        if mine != theirs:
            _fail(f"{label}: {mine!r} != {theirs!r}")
    if ref.get("tip_offsets") is not None:
        if not np.allclose(spec.tip_offsets, np.asarray(ref["tip_offsets"])):
            _fail("tip_offsets differ")
    _ok("all link names, offsets and num_fingers match")

    # --- 2. joint bookkeeping ---------------------------------------------
    print("\n[2] joint structure")
    pin_names = pin_fk.joint_names
    if chain.nq != pin_fk.model.nq:
        _fail(f"nq mismatch: chain {chain.nq} vs pinocchio {pin_fk.model.nq}")
    if sorted(chain.joint_names) != sorted(pin_names):
        _fail("actuated joint name sets differ")
    perm = chain.permutation_to(pin_names)  # q_urdf[perm] == q_pin
    _ok(f"nq={chain.nq}, independent={chain.num_independent}")
    _ok(f"mimic joints: {[j.name for j in chain.joints if j.is_mimic]}")

    lim_urdf = chain.joint_limits()
    if not np.allclose(lim_urdf[perm], pin_fk.limits(), atol=1e-12):
        _fail("joint limits differ from Pinocchio")
    _ok("joint limits match Pinocchio")

    # --- 3. mimic expansion vs the bundled reference optimiser ------------
    print("\n[3] mimic expansion vs ReferenceOptimizer.expand")
    from mbadr.reference import AdaptiveConfig, ReferenceOptimizer

    config = yaml.safe_load(open(default_config_path(robot_type, "pico4")))
    cfg = AdaptiveConfig.from_retarget_section(config.get("retarget", {}))
    ref_opt = ReferenceOptimizer(spec, cfg)

    if sorted(ref_opt.chain.independent_names) != sorted(chain.independent_names):
        _fail("independent joint sets differ between spec and reference")

    rng = np.random.default_rng(0)
    ilim = chain.independent_limits()
    for _ in range(200):
        x = rng.uniform(ilim[:, 0], ilim[:, 1])
        if not np.allclose(chain.expand_to_full(x), ref_opt.expand(x), atol=1e-12):
            _fail("mimic expansion differs from the reference optimiser")
    _ok("mimic expansion identical over 200 samples (incl. limit clipping)")

    # --- 4. FK positions vs Pinocchio -------------------------------------
    print("\n[4] FK points vs Pinocchio")
    query_links = (
        [spec.origin_link]
        + list(spec.tip_links)
        + list(spec.link3_links)
        + list(spec.link4_links)
        + list(spec.link1_links)
    )
    query_offsets = np.vstack(
        [
            np.zeros((1, 3)),
            spec.tip_offsets,
            spec.link3_offsets,
            spec.link4_offsets,
            np.zeros((len(spec.link1_links), 3)),
        ]
    )
    frame_ids = [pin_fk.frame_id(n) for n in query_links]

    lim = chain.joint_limits()
    worst = 0.0
    for trial in range(300):
        q_pin = (
            np.zeros(chain.nq)
            if trial == 0
            else rng.uniform(lim[perm][:, 0], lim[perm][:, 1])
        )
        q_urdf = np.zeros(chain.nq)
        q_urdf[perm] = q_pin

        mine = fk.points(q_urdf, query_links, query_offsets)
        theirs = pin_fk.points(q_pin, frame_ids, query_offsets)
        worst = max(worst, float(np.abs(mine - theirs).max()))
    if worst > TOL_POS:
        _fail(f"max FK position error {worst:.3e} m exceeds {TOL_POS:.0e}")
    _ok(f"300 random poses, {len(query_links)} points; max error {worst:.3e} m")

    # --- 5. analytic Jacobian vs Pinocchio + finite differences ----------
    print("\n[5] analytic Jacobian vs Pinocchio and central differences")
    q_pin = rng.uniform(lim[perm][:, 0], lim[perm][:, 1])
    q_urdf = np.zeros(chain.nq)
    q_urdf[perm] = q_pin

    jac_mine = fk.points_jacobian(q_urdf, query_links, query_offsets)
    jac_pin = pin_fk.position_jacobian(q_pin, frame_ids, query_offsets)
    # Pinocchio columns are in its joint order; remap to URDF order.
    inv = np.argsort(perm)
    jac_pin_urdf = jac_pin[:, :, inv]
    worst_vs_pin = float(np.abs(jac_mine - jac_pin_urdf).max())
    if worst_vs_pin > 1e-6:
        _fail(f"Jacobian vs Pinocchio mismatch {worst_vs_pin:.3e}")

    eps = 1e-7
    worst_vs_fd = 0.0
    for k in range(chain.nq):
        qp, qm = q_urdf.copy(), q_urdf.copy()
        qp[k] += eps
        qm[k] -= eps
        num = (
            fk.points(qp, query_links, query_offsets)
            - fk.points(qm, query_links, query_offsets)
        ) / (2 * eps)
        worst_vs_fd = max(worst_vs_fd, float(np.abs(num - jac_mine[:, :, k]).max()))
    if worst_vs_fd > 1e-6:
        _fail(f"Jacobian vs finite differences mismatch {worst_vs_fd:.3e}")
    _ok(f"vs Pinocchio {worst_vs_pin:.3e}, vs finite differences {worst_vs_fd:.3e}")

    print("\nAll FK checks passed.")


if __name__ == "__main__":
    main()
