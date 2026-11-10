"""Hand specifications and paths, for every supported robot hand.

Link names, offsets and asset locations come from :mod:`.robot_configs`, which
lives in this project together with the URDF/MJCF assets it names, so nothing
here reaches outside the repository. Left/right naming conventions differ per
manufacturer and are mirrored explicitly in :func:`_mirror_name`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from .robot_configs import ROBOT_CONFIGS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUT_DIR = PROJECT_ROOT / "out"
PROCESSED_DIR = PROJECT_ROOT / "data"

# MediaPipe 21-landmark indices.
MP_ORIGIN_IDX = 0
MP_MCP_INDICES = [1, 5, 9, 13, 17]
MP_PIP_INDICES = [2, 6, 10, 14, 18]
MP_DIP_INDICES = [3, 7, 11, 15, 19]
MP_TIP_INDICES = [4, 8, 12, 16, 20]

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Hands with 4 fingers map only thumb..ring; the pinky landmarks are dropped.
FOUR_FINGER_HANDS = ("allegro_hand", "leap_hand")

# Short CLI names -> robot_configs keys.
ROBOT_ALIASES = {
    "l20": "linker_l20",
    "linker_l20": "linker_l20",
    "linkerhand_l21": "linkerhand_l21",
    "shadow": "shadow_hand",
    "shadow_hand": "shadow_hand",
    "wuji": "wuji_hand",
    "wuji_hand": "wuji_hand",
    "sharpa": "sharpa_hand",
    "sharpa_hand": "sharpa_hand",
    "gaia": "gaia_hand20",
    "gaia_hand20": "gaia_hand20",
    "inspire": "inspire_hand",
    "inspire_hand": "inspire_hand",
    "ability": "ability_hand",
    "ability_hand": "ability_hand",
    "allegro": "allegro_hand",
    "allegro_hand": "allegro_hand",
    "leap": "leap_hand",
    "leap_hand": "leap_hand",
    "svh": "svh_hand",
    "svh_hand": "svh_hand",
    "rohand": "rohand",
    "unitree_dex5": "unitree_dex5_hand",
    "unitree_dex5_hand": "unitree_dex5_hand",
}

SUPPORTED_HANDS = sorted(set(ROBOT_ALIASES.values()))


def resolve_robot_name(name: str) -> str:
    """Map a CLI alias (``l20``, ``shadow``) to a ``robot_configs`` key."""
    key = name.lower()
    if key in ROBOT_ALIASES:
        return ROBOT_ALIASES[key]
    if key in SUPPORTED_HANDS:
        return key
    raise ValueError(
        f"Unknown hand {name!r}. Known: {', '.join(SUPPORTED_HANDS)}"
    )


@lru_cache(maxsize=None)
def robot_config(robot_type: str) -> dict:
    """Return the ``ROBOT_CONFIGS`` entry for a hand."""
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(
            f"Unknown robot type {robot_type!r}. Known: {sorted(ROBOT_CONFIGS)}"
        )
    return ROBOT_CONFIGS[robot_type]


def _mirror_name(name: str, robot_type: str, side: str) -> str:
    """Apply a manufacturer's left/right naming convention to a link name.

    Mirrors the per-robot branches in ``BaseOptimizer.__init__``. Getting this
    wrong is silent — the FK would raise "link not found" rather than produce a
    subtly wrong pose — but the conventions are genuinely per-vendor, so they
    are spelled out rather than guessed.
    """
    if side == "right":
        return name
    if robot_type == "shadow_hand":
        return name.replace("rh_", "lh_")
    if robot_type == "unitree_dex5_hand":
        if name == "base_link00":
            return "base_link00L"
        return f"{name[:-1]}L" if name.endswith("R") else name
    if robot_type in (
        "sharpa_hand",
        "gaia_hand20",
        "linker_l20",
        "inspire_hand",
    ):
        return name.replace("right_", "left_")
    return name


@dataclass(frozen=True)
class HandSpec:
    """Link names and offsets defining a hand's retargeting task points.

    All five hands in this project are five-fingered, but ``num_fingers`` is
    honoured because the loss and the landmark indexing both depend on it.
    """

    robot_type: str = "linker_l20"
    hand_side: str = "right"
    urdf_override: Path | None = None

    # ---- construction ----------------------------------------------------

    @classmethod
    def from_name(
        cls, name: str = "l20", hand_side: str = "right"
    ) -> "HandSpec":
        robot_type = resolve_robot_name(name)
        if hand_side not in ("left", "right"):
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side!r}")
        return cls(robot_type=robot_type, hand_side=hand_side)

    # ---- link names ------------------------------------------------------

    def _resolve(self, seq) -> tuple[str, ...]:
        return tuple(
            _mirror_name(n, self.robot_type, self.hand_side) for n in seq
        )

    @property
    def _cfg(self) -> dict:
        return robot_config(self.robot_type)

    @property
    def origin_link(self) -> str:
        return _mirror_name(
            self._cfg["origin_link"], self.robot_type, self.hand_side
        )

    @property
    def tip_links(self) -> tuple[str, ...]:
        return self._resolve(self._cfg["tip_links"])

    @property
    def link1_links(self) -> tuple[str, ...]:
        return self._resolve(self._cfg["link1_names"])

    @property
    def link3_links(self) -> tuple[str, ...]:
        return self._resolve(self._cfg["link3_names"])

    @property
    def link4_links(self) -> tuple[str, ...]:
        return self._resolve(self._cfg["link4_names"])

    @property
    def num_fingers(self) -> int:
        return int(self._cfg.get("num_fingers", 5))

    @property
    def tip_offsets(self) -> np.ndarray:
        """(num_fingers, 3) local offsets from distal frame to contact point."""
        nf = self.num_fingers
        raw = self._cfg.get("tip_offsets")
        if raw is None:
            return np.zeros((nf, 3), dtype=np.float64)
        arr = np.asarray(raw, dtype=np.float64).reshape(-1, 3)
        if len(arr) != nf:
            raise ValueError(
                f"{self.robot_type}: expected {nf} tip_offsets, got {len(arr)}"
            )
        return arr

    def _link_offsets(self, key: str) -> np.ndarray:
        """(num_fingers, 3) offsets for link3/link4 points, zero if unspecified."""
        nf = self.num_fingers
        raw = self._cfg.get(key)
        if raw is None:
            return np.zeros((nf, 3), dtype=np.float64)
        arr = np.asarray(raw, dtype=np.float64).reshape(-1, 3)
        out = np.zeros((nf, 3), dtype=np.float64)
        n = min(nf, len(arr))
        out[:n] = arr[:n]
        return out

    @property
    def link3_offsets(self) -> np.ndarray:
        return self._link_offsets("link3_offsets")

    @property
    def link4_offsets(self) -> np.ndarray:
        return self._link_offsets("link4_offsets")

    @property
    def neutral_qpos(self) -> np.ndarray | None:
        raw = self._cfg.get("neutral_qpos")
        if not raw:
            return None
        return np.asarray(raw, dtype=np.float64)

    @property
    def mp_finger_indices(self) -> list[int]:
        """Which MediaPipe fingers this hand uses."""
        return list(range(self.num_fingers))

    @property
    def tip_frame_links(self) -> tuple[str, ...]:
        """Explicit fingertip frames, when the URDF declares them as links.

        L20's URDF carries fixed ``*_tip`` joints whose origins equal
        ``tip_offsets``, which makes them an independent check on the offset
        path. Other hands have no such frame, so this returns an empty tuple
        and the cross-check is skipped.
        """
        if self.robot_type != "linker_l20":
            return ()
        # L20 link names carry no side prefix on either hand.
        return tuple(
            f"{name}_tip"
            for name in ("thumb", "index", "middle", "ring", "pinky")
        )

    # ---- file locations --------------------------------------------------

    @property
    def urdf_path(self) -> Path:
        if self.urdf_override is not None:
            return Path(self.urdf_override)

        cfg = self._cfg
        subdir = cfg.get("urdf_subdir")
        if subdir:
            files = cfg.get("urdf_file")
            if isinstance(files, dict):
                name = files[self.hand_side]
            else:
                name = f"{self.hand_side}.urdf"
            # Paths in robot_configs are project-relative ("assets/...").
            return PROJECT_ROOT / subdir / name

        raise FileNotFoundError(
            f"{self.robot_type}: no urdf_subdir configured"
        )

    @property
    def mjcf_path(self) -> Path:
        """MuJoCo model for this hand.

        Raises:
            FileNotFoundError: when the hand ships no MuJoCo model, or the model
                named in the config is missing from the assets directory.
        """
        cfg = self._cfg
        subdir = cfg.get("mjcf_subdir")
        files = cfg.get("mjcf_file")
        if subdir and files:
            if isinstance(files, dict):
                if self.hand_side not in files:
                    raise FileNotFoundError(
                        f"{self.robot_type}: no MJCF for the {self.hand_side} hand"
                    )
                return PROJECT_ROOT / subdir / files[self.hand_side]
            return PROJECT_ROOT / subdir / files

        raise FileNotFoundError(
            f"No MuJoCo model configured for {self.robot_type} ({self.hand_side})"
        )

    def has_mjcf(self) -> bool:
        try:
            return self.mjcf_path.exists()
        except FileNotFoundError:
            return False


def default_config_path(robot_type: str, input_source: str = "pico4") -> Path:
    """Path to the adaptive YAML for a hand and input source.

    Configs ship with the project under ``configs/<input source>/``.
    """
    robot_type = resolve_robot_name(robot_type)
    path = CONFIGS_DIR / input_source / f"{input_source}_{robot_type}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No adaptive config at {path}")
    return path


# Default (pico4 input, L20).
DEFAULT_CONFIG_PATH = default_config_path("linker_l20", "pico4")


__all__ = [
    "PROJECT_ROOT",
    "ASSETS_DIR",
    "CONFIGS_DIR",
    "OUT_DIR",
    "PROCESSED_DIR",
    "DEFAULT_CONFIG_PATH",
    "FINGER_NAMES",
    "FOUR_FINGER_HANDS",
    "HandSpec",
    "ROBOT_ALIASES",
    "ROBOT_CONFIGS",
    "SUPPORTED_HANDS",
    "default_config_path",
    "resolve_robot_name",
    "robot_config",
    "MP_ORIGIN_IDX",
    "MP_MCP_INDICES",
    "MP_PIP_INDICES",
    "MP_DIP_INDICES",
    "MP_TIP_INDICES",
]
