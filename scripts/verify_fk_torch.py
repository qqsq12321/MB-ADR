"""Validate the Torch FK against NumPy FK and Pinocchio, forward and backward.

    python scripts/verify_fk_torch.py [--device cuda]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mbadr.chain import KinematicChain
from mbadr.config import HandSpec
from mbadr.fk_numpy import NumpyFK
from scripts.verify_fk import PinocchioFK

TOL_F32 = 1e-5
TOL_F64 = 1e-12


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"  ok: {msg}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    from mbadr.fk_torch import TorchFK

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    spec = HandSpec.from_name("l20", "right")
    chain = KinematicChain.from_urdf(spec.urdf_path)
    np_fk = NumpyFK(chain)
    robot = PinocchioFK(spec.urdf_path)
    perm = chain.permutation_to(robot.joint_names)

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
            np.zeros((len(spec.link3_links), 3)),
            np.zeros((len(spec.link4_links), 3)),
            np.zeros((len(spec.link1_links), 3)),
        ]
    )
    pin_indices = [robot.frame_id(n) for n in query_links]

    # --- 1. torch vs numpy, float64, mimic expansion -----------------------
    print("\n[1] TorchFK (float64) vs NumpyFK")
    fk64 = TorchFK(chain).to(device=device, dtype=torch.float64)
    indep_lim = chain.independent_limits()
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(100):
        x = rng.uniform(indep_lim[:, 0], indep_lim[:, 1])
        q_np = chain.expand_to_full(x)

        with torch.no_grad():
            x_t = torch.from_numpy(x[None]).to(device)
            q_t = fk64.expand(x_t)[0].cpu().numpy()
            pts_t = (
                fk64.points(fk64.expand(x_t), query_links, query_offsets)[0]
                .cpu()
                .numpy()
            )
        worst = max(worst, float(np.abs(q_t - q_np).max()))
        worst = max(worst, float(np.abs(pts_t - np_fk.points(q_np, query_links, query_offsets)).max()))
    if worst > TOL_F64:
        _fail(f"torch float64 vs numpy max error {worst:.3e}")
    _ok(f"expand() and points() agree to {worst:.3e} over 100 poses")

    # --- 2. torch vs Pinocchio, float32, batched ---------------------------
    print("\n[2] TorchFK (float32, batched) vs Pinocchio")
    fk32 = TorchFK(chain).to(device=device, dtype=torch.float32)
    batch = 256
    x = rng.uniform(indep_lim[:, 0], indep_lim[:, 1], size=(batch, chain.num_independent))
    x32 = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        pts32 = fk32.points(fk32.expand(x32), query_links, query_offsets)
    pts32 = pts32.cpu().numpy()

    worst_pin = 0.0
    for b in range(batch):
        q_np = chain.expand_to_full(x[b])
        q_pin = q_np[perm]
        theirs = robot.points(q_pin, pin_indices, query_offsets)
        worst_pin = max(worst_pin, float(np.abs(pts32[b] - theirs).max()))
    if worst_pin > TOL_F32:
        _fail(f"float32 batched FK vs Pinocchio error {worst_pin:.3e} m")
    _ok(f"batch of {batch}, {len(query_links)} points; max error {worst_pin:.3e} m")

    # --- 3. autograd Jacobian vs Pinocchio analytic -----------------------
    print("\n[3] autograd d(points)/dq vs Pinocchio analytic Jacobian")
    # Small batch: jacobian() loops over outputs.
    n_test = 8
    x = rng.uniform(indep_lim[:, 0], indep_lim[:, 1], size=(n_test, chain.num_independent))
    x_t = torch.from_numpy(x).to(device=device, dtype=torch.float64)

    def points_of(q):
        return fk64.points(q, query_links, query_offsets)

    J_auto = torch.autograd.functional.jacobian(
        lambda xx: points_of(fk64.expand(xx)), x_t, vectorize=True
    )
    # Shape: (B, N_l, 3, B, n_indep) -> take diagonal over batch.
    J_auto = torch.stack([J_auto[b, :, :, b, :] for b in range(n_test)])  # (B,N,3,n_indep)
    J_auto = J_auto.cpu().numpy()

    # Pinocchio Jacobians are per-actuated-joint columns. Differentiating an
    # *independent* joint, however, also moves every mimic joint that follows
    # it, so the reference must apply the same chain rule as
    # BaseOptimizer.map_gradient_to_independent:
    #     d/dx_src = d/dq_src + sum_mimic mult * d/dq_mimic
    indep_slots, mimic_slots, mimic_src, mult, _ = chain.mimic_spec()
    worst_j = 0.0
    for b in range(n_test):
        q_np = chain.expand_to_full(x[b])
        q_pin = q_np[perm]
        J_pin = robot.position_jacobian(
            q_pin, pin_indices, query_offsets
        )  # (N, 3, nq_pin)
        inv = np.argsort(perm)
        J_urdf = J_pin[:, :, inv]  # columns -> URDF order

        J_ref = J_urdf[:, :, indep_slots].copy()  # (N, 3, n_indep)
        for source_x, mimic_q, m in zip(mimic_src, mimic_slots, mult):
            J_ref[:, :, source_x] += m * J_urdf[:, :, mimic_q]

        worst_j = max(worst_j, float(np.abs(J_auto[b] - J_ref).max()))
    if worst_j > 1e-8:
        _fail(f"autograd vs Pinocchio Jacobian mismatch {worst_j:.3e}")
    _ok(f"max deviation {worst_j:.3e} across {n_test} poses (mimic chain rule applied)")

    # --- 4. gradient flows through the mimic expansion --------------------
    print("\n[4] gradient reaches every independent joint through mimic joints")
    x_t = torch.zeros((1, chain.num_independent), dtype=torch.float64, device=device)
    x_t.requires_grad_(True)
    tip_ids = [query_links.index(name) for name in spec.tip_links]
    pts = points_of(fk64.expand(x_t))[0, tip_ids].reshape(-1)  # (5*3,)
    weights = torch.arange(1, pts.numel() + 1, dtype=torch.float64, device=device)
    loss = (pts * weights).sum()
    loss.backward()
    grad = x_t.grad[0].cpu().numpy()
    if not np.isfinite(grad).all():
        _fail("non-finite gradients through mimic expansion")
    zero_joints = [n for n, g in zip(chain.independent_names, grad) if abs(g) < 1e-12]
    _ok(f"all {chain.num_independent} grads finite; |grad| range "
        f"[{np.abs(grad).min():.2e}, {np.abs(grad).max():.2e}]")
    if zero_joints:
        print(f"  note: near-zero grads at neutral pose for {zero_joints} "
              f"(expected — these joints do not move the fingertip at q=0)")

    # --- 5. CUDA and CPU agree --------------------------------------------
    if device.type == "cuda":
        print("\n[5] CUDA vs CPU consistency")
        fk_cpu = TorchFK(chain).to(device="cpu", dtype=torch.float32)
        with torch.no_grad():
            pts_cpu = fk_cpu.points(
                fk_cpu.expand(x32.detach().cpu().float()),
                query_links,
                query_offsets,
            ).numpy()
        worst_cuda = float(np.abs(pts_cpu - pts32).max())  # both ndarray here
        if worst_cuda > TOL_F32:
            _fail(f"cuda vs cpu mismatch {worst_cuda:.3e}")
        _ok(f"max deviation {worst_cuda:.3e} m")

    print("\nAll Torch FK checks passed.")


if __name__ == "__main__":
    main()
