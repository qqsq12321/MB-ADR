"""Pico 4 -> MB-ADR checkpoint -> Linker L20 RS485, without a GUI.

    python scripts/teleop_l20.py --hand right --dry-run
    python scripts/teleop_l20.py --hand right --l20-port /dev/ttyUSB0

Ctrl+C closes the connection. By default no extra pose is sent on exit.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from mbadr.config import OUT_DIR
from mbadr.l20 import L20Mapper, L20Output
from mbadr.live import KeypointRetargeter
from mbadr.model import feature_dim
from mbadr.pico4 import Pico4


def uint8(value: str) -> int:
    result = int(value)
    if not 0 <= result <= 255:
        raise argparse.ArgumentTypeError("must be between 0 and 255")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=OUT_DIR / "net_l20.pt")
    parser.add_argument("--config", type=Path, default=None,
                        help="adaptive YAML; defaults to the bundled Pico L20 config")
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smooth", type=float, default=0.3,
                        help="EMA alpha per fresh input sample; 0 = off (default: 0.3)")
    parser.add_argument("--mode", choices=("relay", "direct"), default="relay")
    parser.add_argument("--relay-host", default="127.0.0.1")
    parser.add_argument("--relay-port", type=int, default=63902)
    parser.add_argument("--port", type=int, default=63901, help="Pico direct TCP port")
    parser.add_argument("--broadcast-port", type=int, default=29888)
    parser.add_argument("--l20-port", default="/dev/ttyUSB0")
    parser.add_argument("--l20-baudrate", type=int, default=460800)
    parser.add_argument("--l20-slave-id", type=lambda v: int(v, 0), default=None,
                        help="default: 0x29 right / 0x2A left")
    parser.add_argument("--l20-speed", type=uint8, default=200)
    parser.add_argument("--l20-current-limit", type=uint8, default=200)
    parser.add_argument("--l20-clear-faults", action="store_true")
    parser.add_argument("--l20-open-on-exit", action="store_true")
    parser.add_argument("--l20-print-registers", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print computed registers; never open the serial port")
    parser.add_argument("--hz", type=float, default=60.0,
                        help="maximum inference/command rate (default: 60)")
    parser.add_argument("--input-timeout", type=float, default=0.5,
                        help="stop updating targets after this many seconds without tracking")
    parser.add_argument("--stats-every", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=0.0,
                        help="exit after N seconds; 0 = run until Ctrl+C")
    return parser


def validate_checkpoint(rt: KeypointRetargeter, hand: str) -> None:
    ckpt = rt.ev.ckpt
    if ckpt.get("hand", hand) != hand:
        raise ValueError(f"Checkpoint was trained for {ckpt['hand']}; requested {hand}")
    limits = rt.chain.independent_limits()
    if ckpt["num_joints"] != len(limits) or ckpt["in_dim"] != feature_dim(rt.loss_fn):
        raise ValueError("Checkpoint dimensions do not match L20")
    if not (np.allclose(ckpt["lower"], limits[:, 0], atol=1e-6) and
            np.allclose(ckpt["upper"], limits[:, 1], atol=1e-6)):
        raise ValueError("Checkpoint joint limits do not match L20")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 0 <= args.smooth <= 1:
        parser.error("--smooth must be between 0 and 1")
    for name in ("hz", "input_timeout", "stats_every", "duration"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name in ("hz", "input_timeout") and value == 0):
            parser.error(f"invalid --{name.replace('_', '-')}")
    slave_id = args.l20_slave_id
    if slave_id is None:
        slave_id = 0x29 if args.hand == "right" else 0x2A
    if not 1 <= slave_id <= 247 or args.l20_baudrate <= 0:
        parser.error("invalid --l20-slave-id or --l20-baudrate")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rt = KeypointRetargeter(args.checkpoint, args.hand, device, "l20",
                           smooth=args.smooth, config_path=args.config)
    validate_checkpoint(rt, args.hand)
    L20Mapper(rt.chain.joint_names)  # Validate every motor mapping before opening serial.
    print(f"checkpoint step {rt.ev.ckpt.get('step')}, hand={args.hand}, device={device}, smooth={args.smooth}")
    print(f"config={rt.config_path}; calibration={rt.ev.rotation}")
    print(f"Pico={args.mode}; L20={args.l20_port} @ {args.l20_baudrate}, slave_id={slave_id:#x}")
    print("DRY RUN: serial output disabled" if args.dry_run else "L20 hardware output enabled")

    output = None
    dev = None
    try:
        output = L20Output(
            rt.chain.joint_names, port=args.l20_port, baudrate=args.l20_baudrate,
            slave_id=slave_id, speed=args.l20_speed, current_limit=args.l20_current_limit,
            clear_faults=args.l20_clear_faults, open_on_exit=args.l20_open_on_exit,
            print_registers=args.l20_print_registers, dry_run=args.dry_run,
        )
        dev = Pico4(mode=args.mode, relay_host=args.relay_host, relay_port=args.relay_port,
                    port=args.port, broadcast_port=args.broadcast_port)
        started = last_report = time.monotonic()
        last_sample = 0.0
        processed = 0
        last_send_count = 0
        waiting = False
        period = 1.0 / args.hz
        while args.duration == 0 or time.monotonic() - started < args.duration:
            tick = time.monotonic()
            raw, timestamp = dev.get_hand_sample(args.hand)
            if raw is None or tick - timestamp > args.input_timeout:
                if not waiting:
                    print("waiting for fresh hand tracking; target updates paused")
                    waiting = True
            elif timestamp != last_sample:
                last_sample = timestamp
                q = rt(raw)
                # Inference itself can overrun the input timeout.
                if q is not None and time.monotonic() - timestamp <= args.input_timeout:
                    output.send(q)
                    processed += 1
                    waiting = False
            now = time.monotonic()
            if args.stats_every and now - last_report >= args.stats_every:
                elapsed = now - last_report
                sent = output.send_count - last_send_count
                print(f"input {processed / elapsed:.1f} fps; commands {sent / elapsed:.1f} Hz")
                processed = 0
                last_send_count = output.send_count
                last_report = now
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if dev is not None:
                dev.stop()
        finally:
            if output is not None:
                output.close()
        print("stopped")


if __name__ == "__main__":
    main()
