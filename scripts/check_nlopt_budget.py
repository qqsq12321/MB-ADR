"""Is the network's loss advantage real, or is NLopt just under-converged?

Every hand shows a loss ratio at or below 1.0, which would be surprising if the
reference optimiser were reaching its optimum: the network cannot be better than
the objective's minimum. The reference is deployed with ``maxeval=50`` and a cold
start from the neutral pose, so the likely explanation is that it stops early.

This re-solves the same frames with a much larger evaluation budget and a tighter
tolerance, under identical targets, and reports how far NLopt moves. If the large
budget closes the gap, the honest framing is "the network matches the reference's
deployed quality", not "the network is better than the optimiser".

    python scripts/check_nlopt_budget.py --robot gaia --frames 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mbadr.config import OUT_DIR, default_config_path
from mbadr.data import load_keypoint_array, split_by_time
from mbadr.evaluate import Evaluator, stratified_indices
from mbadr.reference import load_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="gaia")
    parser.add_argument("--data-npy", default="data/qsq_pico4_merged_v1.npy")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--hand", default="right")
    parser.add_argument("--maxeval", type=int, default=2000)
    parser.add_argument("--ftol", type=float, default=1e-9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = OUT_DIR / f"net_{args.robot}.pt"
    ev = Evaluator(
        default_config_path(args.robot, "pico4"), args.hand, ckpt, device, args.robot
    )

    all_kp = load_keypoint_array(
        Path(args.data_npy), args.hand, ev.rotation, verbose=False
    )
    _, val_kp = split_by_time(all_kp, 0.15)
    kp_t = torch.from_numpy(val_kp).float().to(device)
    alphas_all = ev.loss_fn.pinch_alpha(kp_t)
    pick = stratified_indices(alphas_all, args.frames, np.random.default_rng(0))
    kp_np = val_kp[pick]
    kp = torch.from_numpy(kp_np).float().to(device)
    alphas, targets = ev.targets_for(kp)

    ref, _ = load_reference(args.robot, args.hand, "pico4")
    print(f"robot={args.robot}  frames={len(kp_np)}")
    print(f"default budget: maxeval={ref.maxeval} ftol={ref.ftol_abs}")

    def solve_all() -> np.ndarray:
        """Solve every frame; the reference works in this project's URDF order."""
        x = np.zeros((len(kp_np), ev.chain.num_independent))
        ref.reset()
        for i, frame in enumerate(kp_np):
            q = ref.solve(frame)
            x[i] = ev.chain.independent_from_full(q)
        return x

    x_default = solve_all()

    ref.maxeval = args.maxeval
    ref.ftol_abs = args.ftol
    x_big = solve_all()
    ref.maxeval = 50
    ref.ftol_abs = 1e-4

    with torch.no_grad():
        x_net = ev.net_solve(kp)
        l_net = ev.loss_fn(x_net, targets, alphas)
        l_def = ev.loss_fn(torch.from_numpy(x_default).float().to(device), targets, alphas)
        l_big = ev.loss_fn(torch.from_numpy(x_big).float().to(device), targets, alphas)

    net_m = float(l_net.mean())
    def_m = float(l_def.mean())
    big_m = float(l_big.mean())
    print(f"\nnet            {net_m:8.4f}")
    print(f"nlopt @50      {def_m:8.4f}   (ratio net/nlopt {net_m / def_m:.3f})")
    print(f"nlopt @{args.maxeval:<5d}   {big_m:8.4f}   (ratio net/nlopt {net_m / big_m:.3f})")
    print(f"\nbudget improvement: {def_m - big_m:.4f} "
          f"({100 * (1 - big_m / def_m):.1f}% lower loss)")
    gap = net_m - big_m
    print(f"net minus fully-converged nlopt: {gap:+.4f} "
          f"({100 * gap / big_m:+.1f}%)")
    if gap <= 0:
        print("-> the network is still at or below the converged optimum; check targets align")
    elif gap / big_m < 0.05:
        print("-> network matches a well-converged NLopt to within 5%")
    else:
        print("-> network is worse than a well-converged NLopt by more than 5%")


if __name__ == "__main__":
    main()
