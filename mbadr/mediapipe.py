"""MediaPipe landmark frame conversion.

Vendored so this project stands alone. The transform is the standard one used
by MediaPipe-based retargeting: centre on the wrist, build a wrist rotation
frame from landmarks 0/5/9 by SVD, then rotate into the MANO convention so that
the result is independent of how the hand is oriented in space.

Input and output are both (21, 3); only translation and rotation are applied, so
units carry through unchanged (metres for every device we support).
"""

from __future__ import annotations

import numpy as np

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """Orientation of the wrist frame, in the MANO convention.

    Args:
        keypoint_3d_array: (21, 3) landmarks, wrist and index/middle MCP used.

    Returns:
        (3, 3) rotation whose columns are the frame's axes.
    """
    if keypoint_3d_array.shape != (21, 3):
        raise ValueError(f"Expected (21, 3), got {keypoint_3d_array.shape}")

    points = keypoint_3d_array[[0, 5, 9], :]
    x_vector = points[0] - points[2]

    centred = points - np.mean(points, axis=0, keepdims=True)
    _, _, v = np.linalg.svd(centred)
    normal = v[2, :]

    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # Disambiguate the normal's sign using the middle MCP.
    if np.sum(z * (centred[1] - centred[2])) < 0:
        normal = normal * -1
        z = z * -1

    return np.stack([x, normal, z], axis=1)


def apply_mediapipe_transformations(
    keypoint_3d_array: np.ndarray, hand_type: str = "right"
) -> np.ndarray:
    """Map raw device landmarks into the retargeting frame.

    Args:
        keypoint_3d_array: (21, 3) landmarks.
        hand_type: 'right' or 'left'; selects the MANO operator.

    Returns:
        (21, 3) landmarks, wrist-anchored and orientation-normalised.
    """
    hand_type = hand_type.lower()
    if hand_type not in ("right", "left"):
        raise ValueError(f"hand_type must be 'right' or 'left', got {hand_type!r}")

    kp = np.asarray(keypoint_3d_array, dtype=np.float64)
    kp = kp - kp[0:1, :]

    wrist_rot = estimate_frame_from_hand_points(kp)
    operator2mano = (
        OPERATOR2MANO_RIGHT if hand_type == "right" else OPERATOR2MANO_LEFT
    )
    return kp @ wrist_rot @ operator2mano


__all__ = [
    "OPERATOR2MANO_RIGHT",
    "OPERATOR2MANO_LEFT",
    "estimate_frame_from_hand_points",
    "apply_mediapipe_transformations",
]
