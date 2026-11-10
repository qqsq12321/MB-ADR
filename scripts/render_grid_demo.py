"""Render all six hands retargeting the same keypoint window, offscreen.

Produces one MP4 per hand plus a 3x2 grid MP4 combining them, using the shared
scene from :mod:`mbadr.scene` so background and orientation are already
consistent. Intended for building the README's demo GIF; not part of the
verification suite.

    MUJOCO_GL=egl python scripts/render_grid_demo.py --start 300 --end 750
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mbadr.config import HandSpec, default_config_path
from mbadr.data import RECORDING_ROOT, build_dataset
from mbadr.evaluate import Evaluator
from mbadr.mjcf import build_mapping
from mbadr.model import build_features
from mbadr.scene import build_scene

HANDS = ["l20", "shadow", "wuji", "sharpa", "gaia", "inspire"]


def render_hand(robot: str, kp_all: np.ndarray, device, fps: int, size: int, out_dir: Path) -> Path:
    import imageio.v2 as imageio
    import mujoco

    ev = Evaluator(
        default_config_path(robot, "pico4"), "right", Path("out") / f"net_{robot}.pt",
        device, robot,
    )
    spec = HandSpec.from_name(robot, "right")
    scene = build_scene(spec.mjcf_path, ev.chain.joint_names, spec.link1_links, spec.link4_links)
    mapping = build_mapping(scene.model, ev.chain.joint_names)
    model = scene.model
    data = mujoco.MjData(model)

    kp_t = torch.from_numpy(kp_all).float().to(device)
    with torch.no_grad():
        alphas, targets = ev.targets_for(kp_t)
        x = ev.net(build_features(targets, alphas))
        q_all = ev.loss_fn.fk.expand(x).cpu().numpy().astype(np.float64)

    out_path = out_dir / f"{robot}.mp4"
    with mujoco.Renderer(model, size, size) as renderer:
        cam = mujoco.MjvCamera()
        scene.apply_camera(cam)
        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
        for q in q_all:
            data.qpos[:] = mapping.to_qpos(q)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
        writer.close()
    print(f"{robot}: wrote {out_path} ({len(q_all)} frames)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", default="pico4_check.pkl")
    parser.add_argument("--start", type=int, default=300)
    parser.add_argument("--end", type=int, default=750)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--size", type=int, default=360, help="per-tile render size")
    parser.add_argument("--out-dir", default="out/demo_video")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # All hands share the same calibration rotation by convention in this
    # project's pico4 configs' hand-independent axes, but each hand's YAML can
    # define its own, so build the dataset once per hand against its rotation.
    for robot in HANDS:
        rot_ev = Evaluator(
            default_config_path(robot, "pico4"), "right", Path("out") / f"net_{robot}.pt",
            device, robot,
        )
        ds = build_dataset(
            {"p": RECORDING_ROOT / args.recording}, "right",
            rotation_xyz=rot_ev.rotation, verbose=False,
        )
        kp = ds.keypoints[args.start:args.end]
        render_hand(robot, kp, device, args.fps, args.size, out_dir)


if __name__ == "__main__":
    main()
