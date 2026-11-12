"""Live Pico 4 hand tracking -> trained network -> simulated hand in MuJoCo.

Runs closed-loop on the *input* side: whatever the headset reports each frame is
preprocessed exactly as during training, pushed through the network, and applied
to the simulated hand. No physics integration is involved — qpos is set
directly, so what you see is the retargeter's output, not a simulated response.

Prerequisites:
    * the Pico 4 relay daemon running (default), or use --mode direct. The
      driver is bundled in ``mbadr/pico4.py``.

    python scripts/demo_pico4.py --robot l20 --hand right

Keyboard: Esc quits. The viewer's own UI still works.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mbadr.config import HandSpec
from mbadr.live import KeypointRetargeter
from mbadr.mjcf import build_mapping
from mbadr.pico4 import Pico4
from mbadr.scene import build_scene


class LiveRetargeter(KeypointRetargeter):
    """Keypoints -> joint vector, in the MJCF joint order."""

    def __init__(
        self,
        checkpoint: Path,
        hand: str,
        device: torch.device,
        robot: str = "l20",
        *,
        config_path: Path | None = None,
    ):
        super().__init__(checkpoint, hand, device, robot, config_path=config_path)

        spec = HandSpec.from_name(robot, hand)
        self.mjcf_path = spec.mjcf_path
        # Shared background, hand upright with the palm facing the camera.
        self.scene = build_scene(
            spec.mjcf_path, self.chain.joint_names, spec.link1_links, spec.link4_links
        )
        self.mapping = build_mapping(self.scene.model, self.chain.joint_names)
        self.mjcf_names = self.mapping.mjcf_names

    @torch.no_grad()
    def __call__(self, raw_keypoints: np.ndarray) -> np.ndarray | None:
        """Raw device keypoints (21, 3) -> MJCF-order joint vector, or None."""
        q = super().__call__(raw_keypoints)
        return None if q is None else self.mapping.to_qpos(q)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="l20",
                        help="hand name or alias: l20, shadow, wuji, sharpa, gaia, inspire, ...")
    parser.add_argument("--checkpoint", default=None,
                        help="defaults to out/net_<robot>.pt")
    parser.add_argument("--config", type=Path, default=None,
                        help="adaptive YAML; defaults to the Pico config for --robot")
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--mode", default="relay", choices=["relay", "direct"])
    parser.add_argument("--relay-host", default="127.0.0.1")
    parser.add_argument("--relay-port", type=int, default=63902)
    parser.add_argument("--port", type=int, default=63901)
    parser.add_argument("--broadcast-port", type=int, default=29888)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smooth", type=float, default=0.3,
                        help="EMA alpha for joint output, 0 = off (default: 0.3)")
    parser.add_argument("--stats-every", type=float, default=2.0,
                        help="seconds between throughput prints, 0 = silent")
    args = parser.parse_args()

    import mujoco
    import mujoco.viewer
    from mbadr.pico4 import Pico4

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint = Path(args.checkpoint or (Path("out") / f"net_{args.robot}.pt"))
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    rt = LiveRetargeter(checkpoint, args.hand, device, args.robot, config_path=args.config)
    ckpt = rt.ev.ckpt
    val = ckpt.get("val_loss")
    print(
        f"checkpoint step {ckpt.get('step')}, val {val:.4f}" if val
        else f"checkpoint step {ckpt.get('step')}"
    )
    print(f"robot={args.robot}  hand={args.hand}  config={rt.config_path}  calibration={rt.ev.rotation}")

    model = rt.scene.model
    data = mujoco.MjData(model)
    print(f"MJCF nq={model.nq} nu={model.nu}, mapped {len(rt.mapping.src)} joints")

    print(f"connecting to Pico 4 ({args.mode})...")
    dev = Pico4(
        mode=args.mode,
        relay_host=args.relay_host,
        relay_port=args.relay_port,
        port=args.port,
        broadcast_port=args.broadcast_port,
    )

    key = f"{args.hand}_fingers"
    held = np.zeros(len(rt.mjcf_names))
    ema = None
    frames = 0
    applied = 0
    t_last = time.perf_counter()
    t_report = t_last
    waiting_printed = False

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            rt.scene.apply_camera(viewer.cam)
            while viewer.is_running():
                fingers = dev.get_fingers_data()
                q = rt(fingers[key])

                if q is None:
                    if not waiting_printed:
                        print("waiting for hand tracking data...")
                        waiting_printed = True
                else:
                    waiting_printed = False
                    if args.smooth > 0:
                        ema = q if ema is None else ema + args.smooth * (q - ema)
                        q = ema
                    held = q
                    data.qpos[:] = held
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    applied += 1

                viewer.sync()
                frames += 1
                now = time.perf_counter()
                if args.stats_every > 0 and now - t_report >= args.stats_every:
                    fps = (applied / (now - t_last)) if now > t_last else 0.0
                    print(f"applied {applied} frames, {fps:5.1f} fps")
                    t_last = now
                    applied = 0
                    t_report = now
    except KeyboardInterrupt:
        pass
    finally:
        dev.stop()
        print(f"done, {frames} iterations")


if __name__ == "__main__":
    main()
