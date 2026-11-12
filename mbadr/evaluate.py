"""Compare the network against the NumPy NLopt reference on identical keypoints.

The headline numbers are the ones that matter in use:

* **fingertip position error** (cm) between the robot's achieved fingertip and
  the target vector the loss asks for — this is what "precision" means for a
  retargeter, and what drove the decision to build this;
* **pinch gap** (cm) between thumb tip and the pinching fingertip, measured on
  the robot, compared with the human's own gap;
* **loss ratio** to the NLopt optimum on identical keypoints, which says how
  much of the optimiser's quality the network captures;
* **latency** per frame.

All comparisons run on the same keypoints with the same targets, so the only
difference is who produced the joint angles.

    python -m mbadr.evaluate --checkpoint out/net.pt
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .chain import KinematicChain
from .checkpoint import load_checkpoint
from .config import (
    DEFAULT_CONFIG_PATH,
    OUT_DIR,
    HandSpec,
    default_config_path,
)
from .data import RECORDING_ROOT, build_dataset
from .model import RetargetNet, build_features
from .reference import load_reference
from .train import load_loss_spec


def stratified_indices(
    alphas: torch.Tensor, limit: int, rng: np.random.Generator
) -> np.ndarray:
    """Pick evaluation frames that cover both free-space and pinch poses.

    Taking the first N frames is not representative: pinch occurs in roughly a
    third of frames but in runs, so a contiguous slice can contain none of it —
    which silently leaves the metric that matters most unmeasured. Here half the
    budget goes to pinch frames (stratified across which finger is pinching) and
    half to everything else.
    """
    active = alphas[:, 1:].max(dim=1).values.cpu().numpy() > 0.05
    pinch_idx = np.where(active)[0]
    other_idx = np.where(~active)[0]

    half = limit // 2
    chosen: list[np.ndarray] = []
    if len(pinch_idx):
        chosen.append(
            rng.choice(pinch_idx, size=min(half, len(pinch_idx)), replace=False)
        )
    if len(other_idx):
        chosen.append(
            rng.choice(other_idx, size=min(limit - sum(len(c) for c in chosen), len(other_idx)), replace=False)
        )
    if not chosen:
        return np.arange(min(limit, len(alphas)))
    return np.sort(np.concatenate(chosen))


class Evaluator:
    def __init__(
        self,
        config_path: Path,
        hand: str,
        checkpoint: Path,
        device,
        robot: str = "l20",
    ):
        self.cfg, self.chain, self.fk, self.loss_fn, self.params, self.rotation, self.norm_delta = (
            load_loss_spec(config_path, hand, robot)
        )
        self.loss_fn = self.loss_fn.to(device=device, dtype=torch.float32)
        self.fk = self.loss_fn.fk
        self.device = device
        self.hand = hand
        self.robot = robot
        # The reference solver works in this project's URDF joint order, so no
        # joint-order conversion is needed when exchanging qpos vectors.
        self.ref, _ = load_reference(robot, hand, "pico4")

        ckpt = load_checkpoint(checkpoint)
        self.net = RetargetNet(
            in_dim=ckpt["in_dim"],
            num_joints=ckpt["num_joints"],
            lower=ckpt["lower"],
            upper=ckpt["upper"],
            hidden=ckpt["hidden"],
            layers=ckpt["layers"],
        ).to(device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.ckpt = ckpt

    # ---- helpers ---------------------------------------------------------

    def targets_for(self, kp: torch.Tensor):
        alphas = self.loss_fn.pinch_alpha(kp)
        targets = self.loss_fn.targets(kp, alphas)
        return alphas, targets

    @torch.no_grad()
    def net_solve(self, kp: torch.Tensor) -> torch.Tensor:
        alphas, targets = self.targets_for(kp)
        return self.net(build_features(targets, alphas))

    # ---- metrics ---------------------------------------------------------

    def fingertip_error_cm(self, x: torch.Tensor, targets: dict, alphas) -> np.ndarray:
        """Per-frame mean ||achieved tip vector - target tip vector|| in cm."""
        nf = self.loss_fn.spec.num_fingers
        q = self.fk.expand(x)
        pos = self.fk.points(q, self.loss_fn.link_order, self.loss_fn.offset_order) * 100.0
        wrist = pos[:, 0]
        tips = pos[:, 1 : 1 + nf]
        achieved = tips - wrist[:, None]
        err = (achieved - targets["tip_vectors"]).norm(dim=-1)  # (B, nf)
        return err.mean(dim=1).cpu().numpy()

    def pinch_gap_cm(self, x: torch.Tensor, alphas) -> np.ndarray:
        """Robot thumb-tip to partner-tip distance for frames with an active pinch."""
        nf = self.loss_fn.spec.num_fingers
        q = self.fk.expand(x)
        pos = self.fk.points(q, self.loss_fn.link_order, self.loss_fn.offset_order) * 100.0
        tips = pos[:, 1 : 1 + nf]
        partner = alphas[:, 1:].argmax(dim=1) + 1
        partner_tip = tips.gather(
            1, partner[:, None, None].expand(-1, 1, 3)
        ).squeeze(1)
        gap = (tips[:, 0] - partner_tip).norm(dim=1)
        return gap.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="l20",
                        help="hand name or alias: l20, shadow, wuji, sharpa, gaia, inspire, ...")
    parser.add_argument("--config", default=None,
                        help="adaptive YAML; defaults to the pico4 config for --robot")
    parser.add_argument("--hand", default="right")
    parser.add_argument("--checkpoint", default=str(OUT_DIR / "net.pt"))
    parser.add_argument("--recordings", nargs="*", default=None,
                        help="pkl recording names; ignored when --data-npy is given")
    parser.add_argument("--data-npy", default=None,
                        help="bare (N, 21, 3) keypoint array, e.g. the pico4 set")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="tail fraction used as validation with --data-npy")
    parser.add_argument("--limit", type=int, default=400, help="frames per recording")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(OUT_DIR / "eval.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = (
        Path(args.config) if args.config else default_config_path(args.robot, "pico4")
    )
    evaluator = Evaluator(
        config_path, args.hand, Path(args.checkpoint), device, args.robot
    )

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint} (validated at step {evaluator.ckpt.get('step')})")

    reference, _ = load_reference(args.robot, args.hand, "pico4")

    # Build the evaluation frame set. With --data-npy the validation tail of the
    # same Pico capture the model was trained on is used, so the comparison is
    # held out but shares the training distribution.
    if args.data_npy:
        from .data import load_keypoint_array, split_by_time

        all_kp_np = load_keypoint_array(
            Path(args.data_npy), args.hand, evaluator.rotation, verbose=False
        )
        _, val_kp_np = split_by_time(all_kp_np, args.val_fraction)
        sources = {Path(args.data_npy).stem + "_val": val_kp_np}
    else:
        names = args.recordings or ["pico4_check.pkl"]
        sources = {}
        for rec_name in names:
            data = build_dataset(
                {Path(rec_name).stem: RECORDING_ROOT / rec_name},
                args.hand,
                rotation_xyz=evaluator.rotation,
                verbose=False,
            )
            sources[rec_name] = data.keypoints

    results = {}
    for rec_name, keys in sources.items():
        # Select frames by pinch coverage rather than by position in the file.
        all_kp = torch.from_numpy(keys).float().to(device)
        all_alphas = evaluator.loss_fn.pinch_alpha(all_kp)
        pick = stratified_indices(
            all_alphas, args.limit, np.random.default_rng(args.seed)
        )
        kp_np = keys[pick]
        kp = torch.from_numpy(kp_np).float().to(device)
        alphas, targets = evaluator.targets_for(kp)

        # --- network ---
        t0 = time.perf_counter()
        x_net = evaluator.net_solve(kp)
        t_net = (time.perf_counter() - t0) / len(kp_np) * 1000

        # --- NLopt reference, one frame at a time ---
        x_ref = np.zeros((len(kp_np), evaluator.chain.num_independent))
        t_ref_total = 0.0
        reference.reset()
        for i, frame in enumerate(kp_np):
            t1 = time.perf_counter()
            q_ref = reference.solve(frame)  # URDF order, same as this project
            t_ref_total += time.perf_counter() - t1
            x_ref[i] = evaluator.chain.independent_from_full(q_ref)
        t_ref = t_ref_total / len(kp_np) * 1000

        x_ref_t = torch.from_numpy(x_ref).float().to(device)

        loss_net = evaluator.loss_fn(x_net, targets, alphas)
        loss_ref = evaluator.loss_fn(x_ref_t, targets, alphas)

        tip_err_net = evaluator.fingertip_error_cm(x_net, targets, alphas)
        tip_err_ref = evaluator.fingertip_error_cm(x_ref_t, targets, alphas)

        gap_net = evaluator.pinch_gap_cm(x_net, alphas)
        gap_ref = evaluator.pinch_gap_cm(x_ref_t, alphas)

        # Human's own pinch gap, for scale.
        nf = evaluator.loss_fn.spec.num_fingers
        thumb = kp_np[:, 4]
        partner_kp = alphas[:, 1:].argmax(dim=1).cpu().numpy() + 1
        mp_tip = [4, 8, 12, 16, 20]
        human_gap = np.stack(
            [
                np.linalg.norm(thumb[i] - kp_np[i, mp_tip[partner_kp[i]]]) * 100
                for i in range(len(kp_np))
            ]
        )

        active = alphas[:, 1:].max(dim=1).values.cpu().numpy() > 0.05
        pinch_active = active & (human_gap < 4.0)

        def summarise(a: np.ndarray) -> dict:
            return {
                "mean": float(np.mean(a)),
                "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)),
            }

        results[rec_name] = {
            "frames": int(len(kp_np)),
            "pinch_active_frames": int(pinch_active.sum()),
            "loss": {"net": float(loss_net.mean()), "nlopt": float(loss_ref.mean())},
            "loss_ratio": float(loss_net.mean() / max(float(loss_ref.mean()), 1e-9)),
            "fingertip_err_cm": {
                "net": summarise(tip_err_net),
                "nlopt": summarise(tip_err_ref),
            },
            "pinch_gap_cm": {
                "net_all": summarise(gap_net),
                "nlopt_all": summarise(gap_ref),
                "net_pinch": summarise(gap_net[pinch_active]) if pinch_active.any() else None,
                "nlopt_pinch": summarise(gap_ref[pinch_active]) if pinch_active.any() else None,
                "human_pinch": summarise(human_gap[pinch_active]) if pinch_active.any() else None,
            },
            "latency_ms": {"net": t_net, "nlopt": t_ref},
        }

        r = results[rec_name]
        print(f"\n=== {rec_name} ({r['frames']} frames, {r['pinch_active_frames']} pinch) ===")
        print(f"  loss        net {r['loss']['net']:8.4f}   nlopt {r['loss']['nlopt']:8.4f}   "
              f"ratio {r['loss_ratio']:.3f}")
        print(f"  tip error   net {r['fingertip_err_cm']['net']['mean']:.4f} cm  "
              f"nlopt {r['fingertip_err_cm']['nlopt']['mean']:.4f} cm")
        if r["pinch_gap_cm"]["net_pinch"]:
            print(f"  pinch gap   net {r['pinch_gap_cm']['net_pinch']['mean']:.4f} cm  "
                  f"nlopt {r['pinch_gap_cm']['nlopt_pinch']['mean']:.4f} cm  "
                  f"human {r['pinch_gap_cm']['human_pinch']['mean']:.4f} cm")
        print(f"  latency     net {t_net*1000:.1f} us   nlopt {t_ref*1000:.1f} us")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
