"""Keypoint dataset construction for self-supervised retargeting training.

No joint labels are needed: the loss is computed from the analytic FK model, so
the only thing required is a representative distribution of human keypoints.
That makes dataset *coverage* the binding constraint, and the recordings at hand
are small — ``pico4_check.pkl`` is 40 s at ~64 Hz with 71 duplicate frames.

Three things this module does about that:

1. **Deduplication.** The AVP recordings repeat 65-76 % of frames verbatim
   (the headset holds its last sample when tracking stalls). Those duplicates
   are dropped rather than up-weighting whatever the user was holding still.
2. **Augmentation in keypoint space.** Novel poses are synthesised by
   interpolating between real frames, by recombining fingers across frames, and
   by small geometric jitter. Every augmented sample stays a valid 21-landmark
   configuration, so derived quantities (pinch alpha, targets) remain
   self-consistent.
3. **Split by recording.** Validation uses held-out recordings, not held-out
   frames, so reported numbers measure generalisation across capture sessions.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PROCESSED_DIR, PROJECT_ROOT
from .mediapipe import apply_mediapipe_transformations

# Raw ``.pkl`` recordings, when present, live under the project's data/ dir.
RECORDING_ROOT = PROCESSED_DIR / "recordings"


def _rotation_matrix(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """Extrinsic XYZ rotation, matching ``Retargeter._apply_rotation``."""
    from scipy.spatial.transform import Rotation

    if x_deg == 0 and y_deg == 0 and z_deg == 0:
        return np.eye(3)
    return Rotation.from_euler("xyz", [x_deg, y_deg, z_deg], degrees=True).as_matrix()


@dataclass
class Recordings:
    """Loaded, cleaned keypoints for one hand side."""

    keypoints: np.ndarray  # (N, 21, 3) float64, meters, retargeter input frame
    source: np.ndarray  # (N,) str labels
    frame_index: np.ndarray  # (N,) original index within its recording

    def __len__(self) -> int:
        return len(self.keypoints)


def preprocess_keypoints(
    kp: np.ndarray,
    hand: str = "right",
    rotation_xyz: dict | None = None,
    normalize_scale: bool = True,
) -> np.ndarray:
    """Put one raw 21-landmark frame into the frame the network expects.

    This is the single preprocessing definition, shared by recording loaders and
    live device input. Keeping it in one place matters: if live input and
    training input diverge even slightly, the network silently sees a
    distribution it never learned.

    Args:
        kp: (21, 3) raw device output, wrist-centred, meters.
        hand: 'left' or 'right'.
        rotation_xyz: ``mediapipe_rotation`` for the active hand, degrees.
        normalize_scale: rescale so wrist->middle-MCP is :data:`REFERENCE_PALM`.
            VR inputs are metric but drift with tracking quality, and the
            config's ``segment_scaling`` was calibrated against a normalised
            palm.
    """
    kp = np.asarray(kp, dtype=np.float64)
    out = apply_mediapipe_transformations(kp, hand)

    if rotation_xyz:
        rot = _rotation_matrix(
            rotation_xyz.get("x", 0.0),
            rotation_xyz.get("y", 0.0),
            rotation_xyz.get("z", 0.0),
        )
        if not np.allclose(rot, np.eye(3)):
            out = out @ rot.T

    if normalize_scale:
        palm = np.linalg.norm(out[9])
        if palm > 1e-6:
            out = out * (REFERENCE_PALM / palm)
    return out


def load_recording(
    path: Path,
    hand: str = "right",
    rotation_xyz: dict | None = None,
    normalize_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one ``.pkl`` recording into retargeter-input keypoints.

    Returns ``(keypoints, keep_mask)`` where keypoints is (N, 21, 3) in meters,
    already put through :func:`preprocess_keypoints` — i.e. exactly what the
    optimizer and the network consume.
    """
    key = "right_fingers" if hand == "right" else "left_fingers"
    with path.open("rb") as fh:
        frames = pickle.load(fh)

    raw = np.stack([np.asarray(f[key], dtype=np.float64) for f in frames])
    finite = np.isfinite(raw).all(axis=(1, 2))
    nonzero = np.any(raw != 0, axis=(1, 2))
    keep = finite & nonzero

    out = np.zeros_like(raw)
    for i in np.where(keep)[0]:
        out[i] = preprocess_keypoints(
            raw[i], hand, rotation_xyz, normalize_scale
        )
    return out, keep


# Matches landmark_utils.REFERENCE_WRIST_TO_MIDDLE_MCP.
REFERENCE_PALM = 0.092


def load_keypoint_array(
    path: Path,
    hand: str = "right",
    rotation_xyz: dict | None = None,
    normalize_scale: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Load a bare ``.npy`` array of keypoints, (N, 21, 3), in meters.

    The GeoRT-style recordings omit per-frame metadata and hold a single hand
    for the whole sequence. Format is wrist-centred MediaPipe landmarks, same
    as a ``.pkl`` frame after unpacking, so the same preprocessing applies.
    """
    raw = np.load(path, allow_pickle=True)
    if raw.ndim != 3 or raw.shape[1:] != (21, 3):
        raise ValueError(
            f"{Path(path).name}: expected (N, 21, 3), got {raw.shape}"
        )

    finite = np.isfinite(raw).all(axis=(1, 2))
    nonzero = np.any(raw != 0, axis=(1, 2))
    keep = finite & nonzero
    valid = np.where(keep)[0]

    out = np.stack(
        [
            preprocess_keypoints(raw[i], hand, rotation_xyz, normalize_scale)
            for i in valid
        ]
    )
    dedup = deduplicate(out)
    result = out[dedup]

    if verbose:
        print(
            f"  {Path(path).stem:24s} frames={len(raw):6d} valid={len(valid):6d} "
            f"unique={len(result):6d} "
            f"({100 * (1 - len(result) / max(len(valid), 1)):.0f}% dups)"
        )
    return result


def split_by_time(
    keypoints: np.ndarray, val_fraction: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Split a single continuous recording into train/val by position in time.

    Splitting a continuous capture this way keeps validation frames out of
    training while preserving the motion statistics of both halves, which a
    random frame split would destroy (neighbouring frames are near-duplicates).
    """
    n = len(keypoints)
    if n < 20:
        raise ValueError(f"too few frames to split: {n}")
    cut = int(round(n * (1.0 - val_fraction)))
    return keypoints[:cut], keypoints[cut:]


def deduplicate(keypoints: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Indices to keep, dropping frames identical to their predecessor."""
    if len(keypoints) <= 1:
        return np.arange(len(keypoints))
    diff = np.abs(np.diff(keypoints, axis=0)).reshape(len(keypoints) - 1, -1).max(axis=1)
    keep = np.ones(len(keypoints), dtype=bool)
    keep[1:] = diff > tol
    return np.where(keep)[0]


def build_dataset(
    recordings: dict[str, Path],
    hand: str,
    rotation_xyz: dict | None = None,
    verbose: bool = True,
) -> Recordings:
    """Load and deduplicate a set of recordings."""
    all_kp, all_src, all_idx = [], [], []
    for name, path in recordings.items():
        kp, keep = load_recording(path, hand=hand, rotation_xyz=rotation_xyz)
        valid = np.where(keep)[0]
        sub = kp[valid]
        dedup = deduplicate(sub)
        kept = sub[dedup]
        if verbose:
            print(
                f"  {name:24s} frames={len(keep):6d} valid={len(valid):6d} "
                f"unique={len(kept):6d} ({100 * (1 - len(kept) / max(len(valid), 1)):.0f}% dups)"
            )
        all_kp.append(kept)
        all_src.extend([name] * len(kept))
        all_idx.append(valid[dedup])

    return Recordings(
        keypoints=np.concatenate(all_kp, axis=0),
        source=np.array(all_src),
        frame_index=np.concatenate(all_idx, axis=0),
    )


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


class KeypointAugmenter:
    """Synthesises plausible keypoint configurations from real frames.

    All operations act on the 21-landmark array as a whole, so scalar-derived
    quantities computed downstream (pinch alpha, target vectors) stay
    consistent with whatever pose is produced.
    """

    def __init__(
        self,
        interpolate_prob: float = 0.35,
        mix_prob: float = 0.35,
        jitter_std: float = 0.0015,
        noise_std: float = 0.0004,
        scale_range: tuple[float, float] = (0.95, 1.05),
        donor_pool: int = 512,
        seed: int = 0,
    ):
        self.interpolate_prob = interpolate_prob
        self.mix_prob = mix_prob
        self.jitter_std = jitter_std
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.donor_pool = donor_pool
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch: np.ndarray, pool: np.ndarray) -> np.ndarray:
        """Augment ``batch`` (B, 21, 3) using extra frames from ``pool``.

        Fully vectorised: the earlier per-row Python loops made a 512-sample
        batch take seconds, which dominated training time by an order of
        magnitude over the forward/backward pass.
        """
        out = np.array(batch, dtype=np.float64, copy=True)
        n = len(out)
        if n == 0:
            return out

        flat_pool = pool.reshape(len(pool), -1)
        flat_out = out.reshape(n, -1)

        # 1. Interpolate towards a *similar* real frame.
        #
        #    Mixing two dissimilar poses collapses bone lengths (interpolating a
        #    curled finger with an extended one shortens the phalanges, which is
        #    kinematically impossible). Restricting donors to the nearest poses
        #    in pool space keeps every synthesised hand anatomically valid while
        #    still densifying the manifold between captured frames.
        mask = self.rng.random(n) < self.interpolate_prob
        if mask.any() and len(pool) > 1:
            idx = np.where(mask)[0]
            # Search donors among a sampled subset. The nearest neighbour of a
            # frame within a random third of the pool is as good a donor as the
            # global one, and restricting the search keeps the distance matrix
            # small enough to stay cheap: the full matrix is O(batch * pool *
            # features) and dominated the training step.
            cand_n = min(self.donor_pool, len(pool))
            cand = self.rng.choice(len(pool), size=cand_n, replace=False)
            cand_flat = flat_pool[cand]

            query = flat_out[idx]
            # ||a - b||^2 = |a|^2 + |b|^2 - 2 a.b, via one BLAS call.
            d2 = (
                (query * query).sum(axis=1)[:, None]
                + (cand_flat * cand_flat).sum(axis=1)[None, :]
                - 2.0 * query @ cand_flat.T
            )
            k = min(32, cand_n)
            nearest = np.argpartition(d2, k - 1, axis=1)[:, :k]  # (len(idx), k)
            choice = nearest[np.arange(len(idx)), self.rng.integers(0, k, len(idx))]
            pick = cand[choice]

            w = self.rng.uniform(0.15, 0.5, size=(len(idx), 1, 1))
            out[idx] = (1.0 - w) * out[idx] + w * pool[pick]

        # 2. Pinch-depth interpolation. Pinch accuracy is the reason this model
        #    exists, yet over half the captured frames contain no pinch at all,
        #    and uniform interpolation rarely lands in the 0-2 cm band that the
        #    pinch alpha switches on over. Each selected finger is interpolated
        #    towards a donor frame chosen for having a *deeper* thumb-tip gap, so
        #    synthesised poses sweep through pinch depths built from real
        #    geometry.
        mask = self.rng.random(n) < self.mix_prob
        if mask.any() and len(pool):
            idx = np.where(mask)[0]
            thumb_tip_idx = 4
            fingers = self.rng.integers(1, 5, size=len(idx))
            tip_idx = np.array([FINGER_LANDMARK_START[f] + 3 for f in fingers])

            gap_now = np.linalg.norm(
                out[idx, tip_idx] - out[idx, thumb_tip_idx], axis=1
            )
            # Gaps for every finger in every pool frame, computed once per call
            # as a (5, len(pool)) table and then indexed per row.
            pool_gaps_by_finger = np.stack(
                [
                    np.linalg.norm(
                        pool[:, FINGER_LANDMARK_START[f] + 3] - pool[:, thumb_tip_idx],
                        axis=1,
                    )
                    for f in range(5)
                ]
            )  # (5, len(pool))
            pool_gaps = pool_gaps_by_finger[fingers]  # (len(idx), len(pool))

            # Score: prefer donors with a smaller gap than the current frame.
            score = np.where(
                pool_gaps < gap_now[:, None],
                -np.abs(pool_gaps - gap_now[:, None]),
                pool_gaps - gap_now[:, None],
            )
            top_k = min(128, len(pool))
            top = np.argsort(score, axis=1)[:, -top_k:]
            donor = pool[top[np.arange(len(idx)), self.rng.integers(0, top_k, len(idx))]]

            w = self.rng.uniform(0.2, 0.6, size=(len(idx), 1))
            candidate = out[idx].copy()
            starts = np.array([FINGER_LANDMARK_START[f] for f in fingers])
            rows = np.arange(len(idx))
            for offset in range(4):
                col = starts + offset
                blend = w[:, 0][:, None]  # (N, 1) to broadcast over xyz
                candidate[rows, col] = (
                    (1.0 - blend) * out[idx, col] + blend * donor[rows, col]
                )

            # Anatomical guard: phalanx lengths must stay those of a real hand.
            # A blend that shortens a bone is kinematically impossible and would
            # teach the network a pose the robot cannot reach.
            ref_len = phalanx_lengths(out[idx])  # batched
            cand_len = phalanx_lengths(candidate)
            ok = (
                np.abs(cand_len - ref_len) / (ref_len + 1e-9)
            ).max(axis=1) < 0.25
            out[idx[ok]] = candidate[ok]
            self.last_mix_accept_rate = float(ok.mean())

        # 3. Rigid jitter: small random rotation of the whole hand about the
        #    wrist, plus per-landmark noise, so the net does not memorise exact
        #    landmark positions.
        if self.jitter_std > 0:
            axis = self.rng.normal(size=(n, 3))
            axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
            angle = self.rng.normal(scale=self.jitter_std, size=n)
            R = _batch_rotate(axis, angle)  # (n, 3, 3)
            out = np.einsum("nij,nkj->nki", R, out)
        if self.noise_std > 0:
            out += self.rng.normal(scale=self.noise_std, size=out.shape)

        # 4. Uniform scale jitter: hand size differs per user, and the config's
        #    segment_scaling is a single calibrated point in that space.
        lo, hi = self.scale_range
        out *= self.rng.uniform(lo, hi, size=(n, 1, 1))

        return out


# MediaPipe: landmark 0 is the wrist, then 4 landmarks per finger.
#   1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
FINGER_LANDMARK_START = {0: 1, 1: 5, 2: 9, 3: 13, 4: 17}


def _batch_rotate(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rodrigues rotations, vectorised over a batch of (axis, angle) pairs.

    Args:
        axes: (N, 3) unit rotation axes.
        angles: (N,) rotation angles in radians.

    Returns:
        (N, 3, 3) rotation matrices.
    """
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    K = np.zeros((len(axes), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -z, y
    K[:, 1, 0], K[:, 1, 2] = z, -x
    K[:, 2, 0], K[:, 2, 1] = -y, x
    s = np.sin(angles)[:, None, None]
    c = np.cos(angles)[:, None, None]
    return np.eye(3)[None] + s * K + (1.0 - c) * (K @ K)


def phalanx_lengths(kp: np.ndarray) -> np.ndarray:
    """Lengths of MCP->PIP, PIP->DIP, DIP->TIP for the five fingers.

    Accepts a single frame (21, 3) -> (15,), or a batch (N, 21, 3) -> (N, 15).
    """
    kp = np.asarray(kp)
    single = kp.ndim == 2
    if single:
        kp = kp[None]

    starts = np.array(list(FINGER_LANDMARK_START.values()))  # (5,)
    segs = np.stack(
        [
            np.linalg.norm(kp[:, starts + 1] - kp[:, starts], axis=-1),
            np.linalg.norm(kp[:, starts + 2] - kp[:, starts + 1], axis=-1),
            np.linalg.norm(kp[:, starts + 3] - kp[:, starts + 2], axis=-1),
        ],
        axis=2,
    )  # (N, 5, 3)
    out = segs.reshape(len(kp), -1)
    return out[0] if single else out


def _bones_within_tolerance(
    reference: np.ndarray, candidate: np.ndarray, atol: float
) -> bool:
    """True if ``candidate`` keeps every phalanx near its length in ``reference``.

    ``atol`` is a relative tolerance, so the check scales with hand size.
    """
    ref_len = phalanx_lengths(reference)
    cand_len = phalanx_lengths(candidate)
    rel = np.abs(cand_len - ref_len) / (ref_len + 1e-9)
    return bool(rel.max() < atol)


__all__ = [
    "Recordings",
    "KeypointAugmenter",
    "build_dataset",
    "deduplicate",
    "load_recording",
    "REFERENCE_PALM",
    "FINGER_LANDMARK_START",
]
