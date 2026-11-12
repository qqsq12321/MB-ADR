"""Replay a recording through the trained network in the L20 MuJoCo model.

Shows three hands side by side, or one at a time with --mode:

  * the retargeted robot (network output), and
  * optionally the optimizer's solution on the same frames, for comparison.

    python scripts/demo_mujoco.py --checkpoint out/net_all.pt \
        --recording pico4_check.pkl --mode net

Controls: space to pause, left/right to step while paused, Esc to quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mbadr.config import (
    HandSpec,
    default_config_path,
)
from mbadr.data import RECORDING_ROOT, build_dataset
from mbadr.evaluate import Evaluator
from mbadr.mjcf import build_mapping
from mbadr.model import build_features
from mbadr.scene import build_scene


class NetRetargeter:
    """Loads a checkpoint and turns keypoints into a full joint vector."""

    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        robot: str = "l20",
        hand: str = "right",
        *,
        config_path: Path | None = None,
    ):
        config_path = Path(config_path) if config_path is not None else default_config_path(robot)
        self.ev = Evaluator(config_path, hand, checkpoint, device, robot)
        self.device = device
        self.robot = robot
        self.hand = hand
        self.chain = self.ev.chain
        self.loss_fn = self.ev.loss_fn

    @torch.no_grad()
    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        """(21, 3) keypoints -> (nq,) joint angles in URDF order."""
        kp = torch.from_numpy(np.asarray(keypoints, dtype=np.float32))[None].to(self.device)
        alphas, targets = self.ev.targets_for(kp)
        x = self.ev.net(build_features(targets, alphas))
        q = self.loss_fn.fk.expand(x)[0]
        return q.detach().cpu().numpy().astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="l20",
                        help="hand name or alias: l20, shadow, wuji, sharpa, gaia, inspire, ...")
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--checkpoint", default=None,
                        help="defaults to out/net_<robot>.pt")
    parser.add_argument("--config", type=Path, default=None,
                        help="adaptive YAML; defaults to the Pico config for --robot")
    parser.add_argument("--recording", default="pico4_check.pkl")
    parser.add_argument("--limit", type=int, default=0, help="0 = whole recording")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--mode", choices=["net", "nlopt"], default="net")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import mujoco
    import mujoco.viewer

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint = Path(
        args.checkpoint or (Path("out") / f"net_{args.robot}.pt")
    )
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    # ---- data ------------------------------------------------------------
    net = NetRetargeter(checkpoint, device, args.robot, args.hand, config_path=args.config)
    ckpt = net.ev.ckpt
    val = ckpt.get("val_loss")
    print(f"checkpoint step {ckpt.get('step')}, val {val:.4f}" if val else
          f"checkpoint step {ckpt.get('step')}")

    data = build_dataset(
        {Path(args.recording).stem: RECORDING_ROOT / args.recording},
        args.hand,
        rotation_xyz=net.ev.rotation,
        verbose=False,
    )
    kp_all = data.keypoints
    if args.limit:
        kp_all = kp_all[: args.limit]
    print(f"frames: {len(kp_all)}  calibration: {net.ev.rotation}")

    reference = net.ev.ref if args.mode == "nlopt" else None

    # ---- model -----------------------------------------------------------
    # Shared background, hand upright with the palm facing the camera.
    spec = HandSpec.from_name(args.robot, args.hand)
    scene = build_scene(
        spec.mjcf_path, net.chain.joint_names, spec.link1_links, spec.link4_links
    )
    model = scene.model
    mapping = build_mapping(model, net.chain.joint_names)
    print(f"MJCF: nq={model.nq} nu={model.nu} joints={len(mapping.mjcf_names)}")
    print(f"name-based mapping resolved {len(mapping.src)}/{model.nq} DOF")

    data_mj = mujoco.MjData(model)

    def solve(frame: np.ndarray) -> np.ndarray:
        if args.mode == "nlopt":
            # The bundled reference solver already works in URDF joint order.
            return reference.solve(frame)
        return net(frame)

    print("\nlaunching viewer (Esc to quit, space to pause)...")
    period = 1.0 / args.fps
    with mujoco.viewer.launch_passive(model, data_mj) as viewer:
        scene.apply_camera(viewer.cam)
        i = 0
        paused = False
        last = time.perf_counter()
        while viewer.is_running():
            step_start = time.perf_counter()

            if not paused:
                q = solve(kp_all[i])
                data_mj.qpos[:] = mapping.to_qpos(q)
                data_mj.qvel[:] = 0.0
                mujoco.mj_forward(model, data_mj)
                i += 1
                if i >= len(kp_all):
                    if args.loop:
                        i = 0
                    else:
                        i = len(kp_all) - 1
                        paused = True

            viewer.sync()
            elapsed = time.perf_counter() - step_start
            if elapsed < period:
                time.sleep(period - elapsed)

    print("viewer closed")


if __name__ == "__main__":
    main()
