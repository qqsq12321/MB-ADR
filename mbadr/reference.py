"""Self-contained reference retargeter: NumPy gradients + NLopt SLSQP.

This is the project's own "ground truth" solver. It exists so the learned
network can be compared against a real optimiser without depending on any other
repository, and so the Torch implementation has an independent numerical
counterpart to be checked against.

Relationship to the rest of the package:

  * :class:`mbadr.fk_numpy.NumpyFK` supplies positions and analytic
    Jacobians (hand-derived, not autograd).
  * :class:`mbadr.loss_torch.AdaptiveLoss` supplies the *same* objective in
    Torch. Two independent expressions of one objective is the point: if they
    agree to 1e-8, both are almost certainly right.
  * This module optimises with NLopt over the independent joints, applying the
    mimic chain rule by hand.

The objective is the adaptive one: a per-finger blend between a fingertip
position/direction term and a full-hand posture term, plus an optional contact
term and temporal regularisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .chain import KinematicChain
from .config import (
    FINGER_NAMES,
    HandSpec,
    MP_DIP_INDICES,
    MP_MCP_INDICES,
    MP_ORIGIN_IDX,
    MP_PIP_INDICES,
    MP_TIP_INDICES,
    default_config_path,
)
from .fk_numpy import NumpyFK

M_TO_CM = 100.0


def huber(x: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(x)
    return np.where(a <= delta, 0.5 * x * x, delta * (a - 0.5 * delta))


def huber_grad(x: np.ndarray, delta: float) -> np.ndarray:
    return np.where(np.abs(x) <= delta, x, delta * np.sign(x))


@dataclass
class AdaptiveConfig:
    """Loss weights, read from the ``retarget:`` section of a YAML."""

    huber_delta: float = 2.0
    huber_delta_dir: float = 0.5
    w_pos: float = 1.0
    w_dir: float = 10.0
    w_full_hand: float = 1.0
    w_contact: float = 0.0
    alpha: float = 0.7
    pinch_scaling: float = 1.0
    lateral_scaling: float = 1.0
    lateral_axis: int = 1
    norm_delta: float = 0.04
    segment_scaling: dict = field(default_factory=dict)
    pinch_thresholds: dict = field(default_factory=dict)
    mediapipe_rotation: dict = field(default_factory=dict)

    @classmethod
    def from_retarget_section(cls, retarget: dict) -> "AdaptiveConfig":
        from dataclasses import fields as dc_fields

        known = {f.name for f in dc_fields(cls)}
        kwargs = {k: v for k, v in retarget.items() if k in known}
        return cls(**kwargs)

    def segment_scaling_array(self, num_fingers: int) -> np.ndarray:
        """(num_fingers, 4) MCP/PIP/DIP/TIP scaling; 3-value entries imply MCP=1."""
        out = np.ones((num_fingers, 4), dtype=np.float64)
        names = FINGER_NAMES[:num_fingers]
        for i, name in enumerate(names):
            raw = self.segment_scaling.get(name)
            if raw is None:
                continue
            values = np.asarray(raw, dtype=np.float64)
            if values.size == 4:
                out[i] = values
            elif values.size == 3:
                out[i] = np.array([1.0, *values])
            else:
                raise ValueError(
                    f"segment_scaling[{name!r}] needs 3 or 4 values, got {values.size}"
                )
        return out

    def pinch_threshold_arrays(
        self, num_fingers: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """(d1, d2) per non-thumb finger, in cm."""
        names = FINGER_NAMES[1:num_fingers]
        d1, d2 = [], []
        for name in names:
            entry = self.pinch_thresholds.get(name)
            if entry is None:
                raise ValueError(f"pinch_thresholds missing for {name!r}")
            d1.append(float(entry["d1"]))
            d2.append(float(entry["d2"]))
        return np.array(d1), np.array(d2)


class ReferenceOptimizer:
    """NLopt SLSQP solve of the adaptive objective, in NumPy.

    Args:
        spec: hand specification.
        cfg: loss weights and per-hand scaling.
        maxeval: NLopt evaluation budget.
        ftol_abs: NLopt absolute tolerance on the objective.
    """

    def __init__(
        self,
        spec: HandSpec,
        cfg: AdaptiveConfig,
        maxeval: int = 50,
        ftol_abs: float = 1e-4,
    ):
        self.spec = spec
        self.cfg = cfg
        self.chain = KinematicChain.from_urdf(spec.urdf_path)
        self.fk = NumpyFK(self.chain)
        self.maxeval = maxeval
        self.ftol_abs = ftol_abs

        nf = spec.num_fingers
        self.num_fingers = nf
        self.segment_scaling = cfg.segment_scaling_array(nf)
        self.d1, self.d2 = cfg.pinch_threshold_arrays(nf)

        self.link_order = (
            [spec.origin_link]
            + list(spec.tip_links)
            + list(spec.link3_links)
            + list(spec.link4_links)
        )
        self.offsets = np.vstack(
            [
                np.zeros((1, 3)),
                spec.tip_offsets,
                spec.link3_offsets,
                spec.link4_offsets,
            ]
        )
        self.idx_tip = slice(1, 1 + nf)
        self.idx_link3 = slice(1 + nf, 1 + 2 * nf)
        self.idx_link4 = slice(1 + 2 * nf, 1 + 3 * nf)

        self.indep_slots, self.mimic_slots, self.mimic_src, self.mimic_mult, self.mimic_off = (
            self.chain.mimic_spec()
        )
        self.limits = self.chain.joint_limits()
        self.indep_limits = self.chain.independent_limits()
        self.last_qpos: np.ndarray | None = None

    # ---- keypoint-derived targets ---------------------------------------

    def pinch_alpha(self, kp: np.ndarray) -> np.ndarray:
        """(nf,) continuous pinch weights; thumb entry is the largest partner."""
        thumb = kp[MP_TIP_INDICES[0]]
        tips = np.array([kp[MP_TIP_INDICES[i]] for i in range(1, self.num_fingers)])
        dist = np.linalg.norm(tips - thumb, axis=1) * M_TO_CM
        nt = np.clip(
            (self.d2 - dist) / (self.d2 - self.d1 + 1e-8), 0.0, self.cfg.alpha
        )
        return np.concatenate([[nt.max()], nt])

    def full_hand_vectors(self, kp: np.ndarray) -> np.ndarray:
        """(3*nf, 3) PIP/DIP/TIP targets in cm, grown from the wrist."""
        nf = self.num_fingers
        kp_cm = kp * M_TO_CM
        mcp_idx = [MP_MCP_INDICES[i] for i in range(nf)]
        pip_idx = [MP_PIP_INDICES[i] for i in range(nf)]
        dip_idx = [MP_DIP_INDICES[i] for i in range(nf)]
        tip_idx = [MP_TIP_INDICES[i] for i in range(nf)]

        wrist = kp_cm[MP_ORIGIN_IDX]
        palm = kp_cm[mcp_idx] - wrist
        if self.cfg.lateral_scaling != 1.0:
            palm = palm.copy()
            palm[:, self.cfg.lateral_axis] *= self.cfg.lateral_scaling
        proximal = kp_cm[pip_idx] - kp_cm[mcp_idx]
        middle = kp_cm[dip_idx] - kp_cm[pip_idx]
        distal = kp_cm[tip_idx] - kp_cm[dip_idx]

        s = self.segment_scaling
        mcp_vec = palm * s[:, 0:1]
        pip_vec = mcp_vec + proximal * s[:, 1:2]
        dip_vec = pip_vec + middle * s[:, 2:3]
        tip_vec = dip_vec + distal * s[:, 3:4]
        return np.vstack([pip_vec, dip_vec, tip_vec])

    def tip_directions(self, kp: np.ndarray) -> np.ndarray:
        dips = kp[[MP_DIP_INDICES[i] for i in range(self.num_fingers)]]
        tips = kp[[MP_TIP_INDICES[i] for i in range(self.num_fingers)]]
        vec = tips - dips
        return vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)

    def targets(self, kp: np.ndarray) -> dict[str, np.ndarray]:
        """All targets for one keypoint frame."""
        nf = self.num_fingers
        alphas = self.pinch_alpha(kp)
        full = self.full_hand_vectors(kp)
        tip_vectors = full[2 * nf :].copy()

        if self.cfg.pinch_scaling != 1.0 and nf > 1:
            partner = int(np.argmax(alphas[1:]) + 1)
            alpha = float(alphas[partner])
            if alpha > 0.0:
                beta = min(alpha / (self.cfg.alpha + 1e-8), 1.0)
                kp_cm = kp * M_TO_CM
                wrist = kp_cm[MP_ORIGIN_IDX]
                tips = kp_cm[[MP_TIP_INDICES[i] for i in range(nf)]]
                pinch_vec = (tips - wrist) * self.cfg.pinch_scaling
                for i in (0, partner):
                    tip_vectors[i] = (1.0 - beta) * tip_vectors[i] + beta * pinch_vec[i]

        return {
            "alphas": alphas,
            "tip_vectors": tip_vectors,
            "tip_dirs": self.tip_directions(kp),
            "full_hand": full,
        }

    # ---- objective -------------------------------------------------------

    def expand(self, x: np.ndarray) -> np.ndarray:
        """Independent joints -> full qpos, clipped to joint limits."""
        return self.chain.expand_to_full(x)

    def loss_and_grad(
        self,
        x: np.ndarray,
        targets: dict[str, np.ndarray],
        last_x: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """Objective value and gradient with respect to independent joints."""
        cfg = self.cfg
        nf = self.num_fingers
        q = self.expand(x)

        pos = self.fk.points(q, self.link_order, self.offsets) * M_TO_CM
        jac = self.fk.points_jacobian(q, self.link_order, self.offsets) * M_TO_CM

        wrist = pos[0]
        tips = pos[self.idx_tip]
        l3 = pos[self.idx_link3]
        l4 = pos[self.idx_link4]

        alphas = targets["alphas"]
        full = targets["full_hand"]

        # Per-point gradient accumulator in full-qpos space; the mimic chain
        # rule is applied once at the end.
        g_full = np.zeros(self.chain.nq, dtype=np.float64)
        total = 0.0

        # --- tip position ---
        tip_vec = tips - wrist
        diff = tip_vec - targets["tip_vectors"]
        dist = np.linalg.norm(diff, axis=1)
        hg = huber_grad(dist, cfg.huber_delta)
        dirv = diff / (dist[:, None] + 1e-8)
        loss_tip = huber(dist, cfg.huber_delta)
        for i in range(nf):
            coef = alphas[i] * cfg.w_pos * hg[i]
            g = coef * (dirv[i] @ (jac[1 + i] - jac[0]))
            total += alphas[i] * cfg.w_pos * loss_tip[i]
            g_full += g

        # --- tip direction ---
        raw_dir = tips - l4
        norm = np.linalg.norm(raw_dir, axis=1, keepdims=True)
        rdir = raw_dir / (norm + 1e-8)
        ddir = rdir - targets["tip_dirs"]
        dist_dir = np.linalg.norm(ddir, axis=1)
        hgd = huber_grad(dist_dir, cfg.huber_delta_dir)
        loss_dir = huber(dist_dir, cfg.huber_delta_dir)
        for i in range(nf):
            u = rdir[i]
            n = float(norm[i, 0])
            proj = (np.eye(3) - np.outer(u, u)) / (n + 1e-8)
            coef = alphas[i] * cfg.w_dir * hgd[i] / (dist_dir[i] + 1e-8)
            J_diff = jac[1 + i] - jac[1 + 2 * nf + i]
            g_full += coef * (ddir[i] @ proj @ J_diff)
            total += alphas[i] * cfg.w_dir * loss_dir[i]

        # --- full-hand posture ---
        robot_pip = l3 - wrist
        robot_dip = l4 - wrist
        robot_tip_full = tips - wrist
        for i in range(nf):
            coef_common = (1.0 - alphas[i]) * cfg.w_full_hand / 3.0
            for robot_pt, target_pt, j_pt in (
                (robot_pip[i], full[i], jac[1 + nf + i]),
                (robot_dip[i], full[nf + i], jac[1 + 2 * nf + i]),
                (robot_tip_full[i], full[2 * nf + i], jac[1 + i]),
            ):
                d = robot_pt - target_pt
                dn = float(np.linalg.norm(d))
                total += coef_common * float(
                    huber(np.array([dn]), cfg.huber_delta)[0]
                )
                g_full += (
                    coef_common
                    * float(huber_grad(np.array([dn]), cfg.huber_delta)[0])
                    * (d / (dn + 1e-8))
                    @ (j_pt - jac[0])
                )

        # --- contact ---
        if cfg.w_contact > 0.0 and nf > 1:
            partner = int(np.argmax(alphas[1:]) + 1)
            alpha = float(alphas[partner])
            if alpha > 0.0:
                cd = tips[0] - tips[partner]
                total += cfg.w_contact * alpha * float(cd @ cd)
                g_full += (
                    2.0 * cfg.w_contact * alpha * (cd @ (jac[1] - jac[1 + partner]))
                )

        # --- temporal ---
        if last_x is not None and cfg.norm_delta > 0.0:
            dq = q - self.expand(last_x)
            total += cfg.norm_delta * float(dq @ dq)
            g_full += 2.0 * cfg.norm_delta * dq

        return total, self.map_gradient(g_full, q)

    def map_gradient(self, g_full: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Full-qpos gradient -> independent-joint gradient (mimic chain rule).

        A mimic joint moves only because its driver does, so its gradient is
        folded into the driver's with the multiplier. When a mimic joint sits
        exactly on its limit the clamp makes it locally constant, so it must not
        contribute — otherwise the reported gradient disagrees with finite
        differences at saturated poses.
        """
        g_indep = g_full[self.indep_slots].copy()
        eps = 1e-9
        for slot, src, mult in zip(self.mimic_slots, self.mimic_src, self.mimic_mult):
            at_limit = (
                abs(q[slot] - self.limits[slot, 0]) < eps
                or abs(q[slot] - self.limits[slot, 1]) < eps
            )
            if not at_limit:
                g_indep[src] += g_full[slot] * mult
        return g_indep

    # ---- solving ---------------------------------------------------------

    def solve(
        self, kp: np.ndarray, warm_start: np.ndarray | None = None
    ) -> np.ndarray:
        """Solve one frame; returns the full qpos in URDF order.

        Args:
            kp: (21, 3) keypoints in the retargeting frame.
            warm_start: optional full qpos to initialise from.
        """
        import nlopt

        kp = np.asarray(kp, dtype=np.float64)
        targets = self.targets(kp)

        if warm_start is not None:
            x0 = np.asarray(warm_start, dtype=np.float64)[self.indep_slots]
        elif self.last_qpos is not None:
            x0 = self.last_qpos[self.indep_slots]
        else:
            x0 = np.zeros(self.chain.num_independent)
        x0 = np.clip(x0, self.indep_limits[:, 0], self.indep_limits[:, 1])

        opt = nlopt.opt(nlopt.LD_SLSQP, self.chain.num_independent)
        opt.set_lower_bounds(self.indep_limits[:, 0])
        opt.set_upper_bounds(self.indep_limits[:, 1])
        opt.set_maxeval(self.maxeval)
        opt.set_ftol_abs(self.ftol_abs)

        def objective(values, grad_out):
            value, gradient = self.loss_and_grad(np.asarray(values), targets)
            if grad_out.size:
                grad_out[:] = gradient
            return float(value)

        opt.set_min_objective(objective)
        try:
            x_opt = np.asarray(opt.optimize(x0.tolist()), dtype=np.float64)
        except RuntimeError:
            x_opt = x0

        q = self.expand(x_opt)
        self.last_qpos = q
        return q

    def cost(self, q_full: np.ndarray, kp: np.ndarray) -> float:
        """Objective value at a full qpos (for reporting / comparison)."""
        targets = self.targets(np.asarray(kp, dtype=np.float64))
        x = np.asarray(q_full, dtype=np.float64)[self.indep_slots]
        return self.loss_and_grad(x, targets)[0]

    def reset(self) -> None:
        self.last_qpos = None


def load_reference(
    robot: str = "l20", hand: str = "right", input_source: str = "pico4"
) -> tuple[ReferenceOptimizer, dict]:
    """Build the reference optimiser from the bundled YAML for a hand."""
    import yaml

    path = default_config_path(robot, input_source)
    config = yaml.safe_load(path.read_text())
    retarget = config.get("retarget", {})
    cfg = AdaptiveConfig.from_retarget_section(retarget)
    spec = HandSpec.from_name(robot, hand)
    return ReferenceOptimizer(spec, cfg), config


def rotation_for(config: dict, hand: str) -> dict:
    """Pick the ``mediapipe_rotation`` entry for one hand side."""
    rot = config.get("retarget", {}).get("mediapipe_rotation", {})
    if isinstance(rot, dict) and hand in rot:
        return rot[hand]
    return rot


__all__ = [
    "AdaptiveConfig",
    "ReferenceOptimizer",
    "huber",
    "huber_grad",
    "load_reference",
    "rotation_for",
    "M_TO_CM",
]
