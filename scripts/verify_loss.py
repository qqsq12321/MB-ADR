"""Validate the Torch adaptive loss against the NumPy reference optimiser.

Two independently written expressions of one objective must agree:

  1. pinch alpha weights
  2. full-hand and tip target vectors
  3. total loss value
  4. the gradient (autograd through Torch FK vs hand-written NumPy Jacobians)

Sharing code between them would defeat the check, so they are separate
implementations of the same formulas.

    python scripts/verify_loss.py --robot all
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mbadr.chain import KinematicChain
from mbadr.config import (
    DEFAULT_CONFIG_PATH,
    HandSpec,
    default_config_path,
)

TOL = 1e-6

DEFAULT_HANDS = ["l20", "shadow", "wuji", "sharpa", "gaia", "inspire"]


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"  ok: {msg}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="l20",
                        help="hand name or alias; 'all' checks every adapted hand")
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--frames", type=int, default=400)
    args = parser.parse_args()

    names = DEFAULT_HANDS if args.robot == "all" else [args.robot]
    failures = 0
    for name in names:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        try:
            check_hand(name, args.hand, args.frames)
        except SystemExit:
            failures += 1
            print(f"  {name} FAILED")
    if failures:
        print(f"\n{failures}/{len(names)} hands failed")
        raise SystemExit(1)
    print(f"\nAll {len(names)} hand(s) passed.")


def check_hand(robot: str, hand: str, n_frames: int) -> None:
    import torch

    from mbadr.fk_torch import TorchFK
    from mbadr.loss_torch import AdaptiveLoss, AdaptiveParams
    from mbadr.reference import AdaptiveConfig, ReferenceOptimizer

    spec = HandSpec.from_name(robot, hand)
    chain = KinematicChain.from_urdf(spec.urdf_path)

    config_path = default_config_path(robot, "pico4")
    cfg = yaml.safe_load(open(config_path))
    retarget_cfg = cfg["retarget"]

    # Two independent expressions of the same objective: AdaptiveLoss in Torch
    # (differentiable, used for training) and ReferenceOptimizer in NumPy
    # (analytic gradients, used as the solver). Agreeing numerically is the
    # check; sharing code would defeat it, so they are written separately.
    ref_cfg = AdaptiveConfig.from_retarget_section(retarget_cfg)
    reference = ReferenceOptimizer(spec, ref_cfg)

    fk = TorchFK(chain).to(dtype=torch.float64)
    params = AdaptiveParams.from_yaml_section(retarget_cfg)
    seg_full = ref_cfg.segment_scaling_array(spec.num_fingers)
    d1, d2 = ref_cfg.pinch_threshold_arrays(spec.num_fingers)

    loss_fn = AdaptiveLoss(spec, fk, params, seg_full, (d1, d2)).to(torch.float64)

    if abs(reference.cfg.alpha - params.alpha) > 1e-12:
        _fail(f"alpha cap differs: reference {reference.cfg.alpha} vs ours {params.alpha}")
    _ok(f"alpha cap matches ({reference.cfg.alpha})")

    # --- load real keypoints ----------------------------------------------
    from mbadr.data import load_keypoint_array, preprocess_keypoints
    from mbadr.reference import rotation_for

    rotation = rotation_for(cfg, hand)
    data_npy = Path(__file__).resolve().parent.parent / "data" / "qsq_pico4_merged_v1.npy"
    keypoints = load_keypoint_array(
        data_npy, hand, rotation, normalize_scale=True, verbose=False
    )[:n_frames]
    print(f"\nusing {len(keypoints)} frames from {data_npy.name}")

    def to_optimizer_frame(raw):
        """Re-derive from the pre-normalised frame, which is what the
        reference solver's alpha/targets are defined on."""
        return preprocess_keypoints(raw, hand, rotation, normalize_scale=False)

    # --- 1. pinch alpha ----------------------------------------------------
    print("\n[1] pinch alpha vs ReferenceOptimizer.pinch_alpha")
    worst = 0.0
    for raw in keypoints:
        kp = to_optimizer_frame(np.asarray(raw, dtype=np.float64))
        kp_t = torch.from_numpy(kp)[None]
        a_t = loss_fn.pinch_alpha(kp_t)[0].detach().numpy()
        a_ref = reference.pinch_alpha(kp)
        worst = max(worst, float(np.abs(a_t - a_ref).max()))
    if worst > TOL:
        _fail(f"alpha mismatch {worst:.3e}")
    _ok(f"max deviation {worst:.3e} over {len(keypoints)} frames")

    # --- 2. full-hand and tip targets -------------------------------------
    print("\n[2] target vectors vs ReferenceOptimizer.targets")
    worst_full = 0.0
    worst_tip = 0.0
    worst_dir = 0.0
    for raw in keypoints:
        kp = to_optimizer_frame(np.asarray(raw, dtype=np.float64))
        kp_t = torch.from_numpy(kp)[None]
        alphas = loss_fn.pinch_alpha(kp_t)
        tgt = loss_fn.targets(kp_t, alphas)

        ref_targets = reference.targets(kp)

        worst_full = max(
            worst_full,
            float(np.abs(tgt["full_hand"][0].detach().numpy() - ref_targets["full_hand"]).max()),
        )
        worst_tip = max(
            worst_tip,
            float(np.abs(tgt["tip_vectors"][0].detach().numpy() - ref_targets["tip_vectors"]).max()),
        )
        worst_dir = max(
            worst_dir,
            float(np.abs(tgt["tip_dirs"][0].detach().numpy() - ref_targets["tip_dirs"]).max()),
        )
    for label, value in (
        ("full-hand", worst_full),
        ("tip vectors", worst_tip),
        ("tip dirs", worst_dir),
    ):
        if value > TOL:
            _fail(f"{label} mismatch {value:.3e}")
        _ok(f"{label:12s} max deviation {value:.3e}")

    # --- 3. total loss ----------------------------------------------------
    print("\n[3] total loss vs ReferenceOptimizer.cost")
    rng = np.random.default_rng(3)
    ilim = chain.independent_limits()
    worst_loss = 0.0
    checked = 0
    for raw in keypoints[::4]:
        kp = to_optimizer_frame(np.asarray(raw, dtype=np.float64))
        kp_t = torch.from_numpy(kp)[None]
        alphas = loss_fn.pinch_alpha(kp_t)
        tgt = loss_fn.targets(kp_t, alphas)

        for _ in range(2):
            x = rng.uniform(ilim[:, 0], ilim[:, 1])
            q_full = chain.expand_to_full(x)

            mine = float(loss_fn(torch.from_numpy(x[None]), tgt, alphas)[0])
            ref = reference.cost(q_full, kp)  # same URDF joint order
            if not np.isfinite(ref):
                continue
            worst_loss = max(worst_loss, abs(mine - ref))
            checked += 1
    if worst_loss > 1e-3:
        _fail(f"loss mismatch {worst_loss:.3e} over {checked} (kp, q) pairs")
    _ok(f"max |torch - reference| = {worst_loss:.3e} over {checked} pairs")

    # --- 4. gradient vs the reference analytic gradient -------------------
    print("\n[4] autograd gradient vs reference analytic gradient")
    raw = keypoints[0]
    kp = to_optimizer_frame(np.asarray(raw, dtype=np.float64))
    kp_t = torch.from_numpy(kp)[None]
    alphas = loss_fn.pinch_alpha(kp_t)
    tgt = loss_fn.targets(kp_t, alphas)

    x = rng.uniform(ilim[:, 0], ilim[:, 1])
    x_t = torch.from_numpy(x[None]).requires_grad_(True)
    loss_fn(x_t, tgt, alphas).sum().backward()
    grad_mine = x_t.grad[0].detach().numpy()

    # Both sides work in independent-joint space over the same URDF order, so
    # the gradients are directly comparable without any reordering.
    _, grad_ref = reference.loss_and_grad(x, reference.targets(kp))

    worst_grad = float(np.abs(grad_mine - grad_ref).max())
    rel = worst_grad / (np.abs(grad_ref).max() + 1e-12)
    if rel > 1e-5:
        _fail(f"gradient mismatch {worst_grad:.3e} (relative {rel:.3e})")
    _ok(f"max |autograd - reference| = {worst_grad:.3e} (relative {rel:.3e})")

    print("\nAll loss checks passed.")


if __name__ == "__main__":
    main()
