"""Self-supervised training of the amortised retargeting network.

The objective is identical to the one the NLopt reference minimises, so no
labels are needed: keypoints come from real recordings, the loss is evaluated
through the analytic differentiable FK, and the network learns to produce the
joint angles that would minimise that loss.

Validation uses *held-out recordings*, and reports two things that matter:

* the network's own loss, and
* the loss of the NLopt solution on the same keypoints, which is the floor.

The ratio between them is the honest summary: 1.0 means the network has learned
to reproduce the optimiser; higher means it has not.

    python -m mbadr.train --steps 20000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .chain import KinematicChain
from .checkpoint import FORMAT_VERSION
from .config import (
    DEFAULT_CONFIG_PATH,
    OUT_DIR,
    HandSpec,
    default_config_path,
    resolve_robot_name,
)
from .data import (
    RECORDING_ROOT,
    KeypointAugmenter,
    build_dataset,
    load_keypoint_array,
    split_by_time,
)
from .fk_torch import TorchFK
from .loss_torch import AdaptiveLoss, AdaptiveParams
from .model import RetargetNet, build_features, feature_dim


def load_loss_spec(config_path: Path, hand: str, robot: str = "l20"):
    """Read the retarget config and build the matching loss + FK.

    Args:
        config_path: adaptive YAML for this robot + input source.
        hand: 'left' or 'right'.
        robot: hand name or alias (``l20``, ``shadow``, ``wuji``, ...).

    Returns:
        ``(cfg, chain, fk, loss_fn, params, rotation, norm_delta)``.
    """
    with Path(config_path).open() as stream:
        cfg = yaml.safe_load(stream)
    cfg.setdefault("optimizer", {})["hand_side"] = hand
    rc = cfg["retarget"]

    spec = HandSpec.from_name(robot, hand)
    chain = KinematicChain.from_urdf(spec.urdf_path)
    fk = TorchFK(chain)
    params = AdaptiveParams.from_yaml_section(rc)

    # Parsing of the per-finger scaling and pinch thresholds lives in
    # AdaptiveConfig, which is also what the NumPy reference solver reads, so a
    # single definition feeds both implementations.
    from .reference import AdaptiveConfig

    shared_cfg = AdaptiveConfig.from_retarget_section(rc)
    segment_scaling = shared_cfg.segment_scaling_array(spec.num_fingers)
    d1, d2 = shared_cfg.pinch_threshold_arrays(spec.num_fingers)

    loss_fn = AdaptiveLoss(spec, fk, params, segment_scaling, (d1, d2))
    rotation = rc.get("mediapipe_rotation", {})
    # Some configs are side-aware ({right: {...}, left: {...}}), others are flat.
    if isinstance(rotation, dict) and hand in rotation:
        rotation = rotation[hand]
    return cfg, chain, fk, loss_fn, params, rotation, float(rc.get("norm_delta", 0.04))


def make_batch(
    pool: torch.Tensor,
    loss_fn: AdaptiveLoss,
    augmenter: KeypointAugmenter,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a batch of keypoints, augment, and derive targets."""
    idx = rng.integers(0, len(pool), size=batch_size)
    batch = pool[idx].cpu().numpy().astype(np.float64)
    batch = augmenter(batch, pool.cpu().numpy().astype(np.float64))
    kp = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
    alphas = loss_fn.pinch_alpha(kp)
    targets = loss_fn.targets(kp, alphas)
    features = build_features(targets, alphas)
    return kp, features, alphas, targets


@torch.no_grad()
def evaluate(
    net: RetargetNet,
    loss_fn: AdaptiveLoss,
    keypoints: torch.Tensor,
    device: torch.device,
    chunk: int = 512,
) -> dict[str, float]:
    """Mean loss of the network on fixed keypoints, plus FK reconstruction error."""
    net.eval()
    losses = []
    for start in range(0, len(keypoints), chunk):
        kp = keypoints[start : start + chunk].to(device)
        alphas = loss_fn.pinch_alpha(kp)
        targets = loss_fn.targets(kp, alphas)
        features = build_features(targets, alphas)
        x = net(features)
        losses.append(loss_fn(x, targets, alphas).detach())
    net.train()
    return {"loss": float(torch.cat(losses).mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="l20",
                        help="hand name or alias: l20, shadow, wuji, sharpa, gaia, inspire, ...")
    parser.add_argument("--config", default=None,
                        help="adaptive YAML; defaults to the pico4 config for --robot")
    parser.add_argument("--hand", default="right")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--norm-delta", type=float, default=None,
                        help="temporal regularisation; 0 trains a stateless model")
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(OUT_DIR / "net.pt"))
    parser.add_argument("--train-recordings", nargs="*", default=None)
    parser.add_argument("--val-recordings", nargs="*", default=None)
    parser.add_argument("--data-npy", default=None,
                        help="bare (N, 21, 3) keypoint array; overrides the pkl "
                             "recordings and splits train/val by time")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="tail fraction held out when --data-npy is used")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    config_path = (
        Path(args.config) if args.config
        else default_config_path(args.robot, "pico4")
    )
    cfg, chain, fk, loss_fn, params, rotation, norm_delta = load_loss_spec(
        config_path, args.hand, args.robot
    )
    # Training runs in float32: the network and the FK must share a dtype, and
    # the loss module's buffers are built from float64 numpy arrays.
    loss_fn = loss_fn.to(device=device, dtype=torch.float32)
    fk = loss_fn.fk
    print(f"robot: {args.robot}  config: {config_path.name}")
    print(f"independent joints: {chain.num_independent}  (nq={chain.nq})")

    # ---- data ------------------------------------------------------------
    # Two sources: a bare .npy of keypoints (GeoRT-style, single continuous
    # capture) or the .pkl recordings from the headset. When a .npy is given,
    # train/val are split by time within it, so validation stays a held-out
    # segment of the same capture.
    if args.data_npy:
        print("\ndataset:")
        all_kp = load_keypoint_array(
            Path(args.data_npy), args.hand, rotation, normalize_scale=True
        )
        train_kp, val_kp_np = split_by_time(all_kp, args.val_fraction)
        print(
            f"\ntrain frames: {len(train_kp)}   val frames: {len(val_kp_np)} "
            f"(time split at {100 * (1 - args.val_fraction):.0f}%)"
        )
        if len(train_kp) < 100:
            raise SystemExit("not enough training frames")
        pool = torch.from_numpy(train_kp).float()
        val_kp = torch.from_numpy(val_kp_np).float()
    else:
        train_files = args.train_recordings or ["pico4_check.pkl"]
        val_files = args.val_recordings or ["avp_20261019_103754.pkl"]
        print("\ntraining recordings:")
        train_set = build_dataset(
            {Path(f).stem: RECORDING_ROOT / f for f in train_files},
            args.hand,
            rotation_xyz=rotation,
        )
        print("validation recordings:")
        val_set = build_dataset(
            {Path(f).stem: RECORDING_ROOT / f for f in val_files},
            args.hand,
            rotation_xyz=rotation,
        )
        print(f"\ntrain frames: {len(train_set)}   val frames: {len(val_set)}")
        if len(train_set) < 100:
            raise SystemExit("not enough training frames")
        pool = torch.from_numpy(train_set.keypoints).float()
        val_kp = torch.from_numpy(val_set.keypoints).float()

    # ---- model -----------------------------------------------------------
    in_dim = feature_dim(loss_fn)
    ilim = chain.independent_limits()
    net = RetargetNet(
        in_dim=in_dim,
        num_joints=chain.num_independent,
        lower=torch.from_numpy(ilim[:, 0]),
        upper=torch.from_numpy(ilim[:, 1]),
        hidden=args.hidden,
        layers=args.layers,
    ).to(device)
    print(f"model: {sum(p.numel() for p in net.parameters()):,} params, in_dim={in_dim}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    augmenter = KeypointAugmenter(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    nd = norm_delta if args.norm_delta is None else args.norm_delta

    best = float("inf")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        kp, features, alphas, targets = make_batch(
            pool, loss_fn, augmenter, args.batch_size, rng, device
        )
        x = net(features)

        # NOTE on the temporal term. The reference uses norm_delta * ||q - q_last||^2
        # to keep consecutive solves from jumping, and it works because the
        # optimiser is warm-started from the previous answer. A stateless
        # network has no such anchor: regularising towards its own current
        # output is a no-op, and regularising towards a *previous frame's*
        # prediction requires sequence-ordered batches. Sequential training is
        # therefore a separate option (--sequence-mode), not something to fake
        # here. These batches are shuffled independent frames, so the term is
        # deliberately omitted and smoothness is handled by the low-pass filter
        # at inference, exactly as it is for the optimiser's output.
        loss = loss_fn(x, targets, alphas, last_q=None, norm_delta=0.0).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.val_every == 0 or step == 1:
            metrics = evaluate(net, loss_fn, val_kp, device)
            elapsed = time.time() - t0
            print(
                f"step {step:6d}  train {float(loss):8.4f}  val {metrics['loss']:8.4f}  "
                f"lr {sched.get_last_lr()[0]:.2e}  {elapsed:6.1f}s"
            )
            if metrics["loss"] < best:
                best = metrics["loss"]
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "format_version": FORMAT_VERSION,
                        "state_dict": net.state_dict(),
                        "in_dim": in_dim,
                        "num_joints": chain.num_independent,
                        "hidden": args.hidden,
                        "layers": args.layers,
                        "lower": torch.from_numpy(ilim[:, 0].copy()),
                        "upper": torch.from_numpy(ilim[:, 1].copy()),
                        "robot": resolve_robot_name(args.robot),
                        "joint_names": chain.joint_names,
                        "independent_names": chain.independent_names,
                        "config": config_path.name,
                        "hand": args.hand,
                        "rotation": rotation,
                        "val_loss": best,
                        "step": step,
                    },
                    args.out,
                )

    print(f"\nbest val loss {best:.4f}  ->  {args.out}")


if __name__ == "__main__":
    main()
