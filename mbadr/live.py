"""Live keypoint inference, without a MuJoCo model or display connection."""

from pathlib import Path

import numpy as np
import torch

from .config import default_config_path
from .data import preprocess_keypoints
from .evaluate import Evaluator
from .model import build_features


class KeypointRetargeter:
    """Raw (21, 3) landmarks -> full URDF-order joint angles in radians.

    Call once per input sample when smoothing is enabled. The EMA operates on
    independent joints, before mimic expansion, so coupling remains consistent.
    """

    def __init__(
        self,
        checkpoint: Path,
        hand: str,
        device: torch.device,
        robot: str = "l20",
        smooth: float = 0.0,
    ):
        if not 0.0 <= smooth <= 1.0:
            raise ValueError("smooth must be between 0 and 1 (0 disables smoothing)")
        self.ev = Evaluator(default_config_path(robot), hand, checkpoint, device, robot)
        self.device = device
        self.hand = hand
        self.robot = robot
        self.chain = self.ev.chain
        self.loss_fn = self.ev.loss_fn
        self.smooth = smooth
        self._ema = None

    @torch.no_grad()
    def __call__(self, raw_keypoints: np.ndarray) -> np.ndarray | None:
        raw = np.asarray(raw_keypoints, dtype=np.float64)
        if raw.shape != (21, 3) or not np.isfinite(raw).all():
            return None
        # Reject missing/collapsed palm geometry before the wrist-frame SVD.
        index = raw[5] - raw[0]
        middle = raw[9] - raw[0]
        if np.linalg.norm(middle) < 1e-6 or np.linalg.norm(np.cross(index, middle)) < 1e-8:
            return None
        kp = preprocess_keypoints(raw, self.hand, self.ev.rotation, normalize_scale=True)
        if not np.isfinite(kp).all():
            return None
        t = torch.from_numpy(kp[None]).float().to(self.device)
        alphas, targets = self.ev.targets_for(t)
        x = self.ev.net(build_features(targets, alphas))
        if self.smooth > 0:
            self._ema = x if self._ema is None else self._ema + self.smooth * (x - self._ema)
            x = self._ema
        return self.loss_fn.fk.expand(x)[0].cpu().numpy().astype(np.float64)
