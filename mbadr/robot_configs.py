"""Link and URDF/MJCF configuration for the supported robot hands.

This is the project's own copy of the hand table, so nothing here depends on
another repository. Paths are relative to the project root (``assets/...``).

Per-entry keys:
    origin_link   base/palm frame used as the wrist for all task points
    tip_links     distal link for each finger, in thumb..pinky order
    link1_names   proximal phalanx link per finger
    link3_names   middle/PIP link per finger
    link4_names   distal/DIP link per finger
    urdf_subdir, urdf_file   URDF location, per hand side
    mjcf_subdir, mjcf_file   MuJoCo model, when the hand ships one
    num_fingers   4 or 5; 4-finger hands drop the pinky landmarks
    neutral_qpos  optional neutral joint vector
    *_offsets     optional local offsets (metres) for tip/link3/link4 points

Link names are written for the right hand; :func:`mbadr.config._mirror_name`
applies each manufacturer's left-hand naming convention.
"""

from __future__ import annotations

ROBOT_CONFIGS: dict[str, dict] = {
    # Shadow Hand, MuJoCo Menagerie style meshes with rh_/lh_ link prefixes.
    "shadow_hand": {
        "origin_link": "rh_palm",
        "tip_links": ["rh_thtip", "rh_fftip", "rh_mftip", "rh_rftip", "rh_lftip"],
        "link1_names": [
            "rh_thproximal",
            "rh_ffproximal",
            "rh_mfproximal",
            "rh_rfproximal",
            "rh_lfproximal",
        ],
        "link3_names": [
            "rh_thmiddle",
            "rh_ffmiddle",
            "rh_mfmiddle",
            "rh_rfmiddle",
            "rh_lfmiddle",
        ],
        "link4_names": [
            "rh_thdistal",
            "rh_ffdistal",
            "rh_mfdistal",
            "rh_rfdistal",
            "rh_lfdistal",
        ],
        "urdf_subdir": "assets/shadow_hand",
        "urdf_file": {"right": "right_hand_mj.urdf", "left": "left_hand_mj.urdf"},
        "mjcf_subdir": "assets/shadow_hand",
        "mjcf_file": {"right": "scene_right.xml", "left": "scene_left.xml"},
        "num_fingers": 5,
    },
    # Wuji Hand: 5 fingers x 4 joints = 20 DOF.
    "wuji_hand": {
        "origin_link": "right_palm_link",
        "tip_links": [
            "right_finger1_tip_link",
            "right_finger2_tip_link",
            "right_finger3_tip_link",
            "right_finger4_tip_link",
            "right_finger5_tip_link",
        ],
        "link1_names": [
            "right_finger1_link1",
            "right_finger2_link1",
            "right_finger3_link1",
            "right_finger4_link1",
            "right_finger5_link1",
        ],
        "link3_names": [
            "right_finger1_link3",
            "right_finger2_link3",
            "right_finger3_link3",
            "right_finger4_link3",
            "right_finger5_link3",
        ],
        "link4_names": [
            "right_finger1_link4",
            "right_finger2_link4",
            "right_finger3_link4",
            "right_finger4_link4",
            "right_finger5_link4",
        ],
        "urdf_subdir": "assets/wuji_hand",
        "urdf_file": {"right": "right.urdf", "left": "left.urdf"},
        "mjcf_subdir": "assets/wuji_hand",
        "mjcf_file": {"right": "right.xml", "left": "left.xml"},
        "num_fingers": 5,
    },
    # Gaia Hand20: 20 revolute joints, 4 of them mimic-driven.
    "gaia_hand20": {
        "origin_link": "right_base_link",
        "tip_links": [
            "right_thumb_tip",
            "right_index_tip",
            "right_middle_tip",
            "right_ring_tip",
            "right_little_tip",
        ],
        "link1_names": [
            "right_thumb_link_1",
            "right_index_link_1",
            "right_middle_link_1",
            "right_ring_link_1",
            "right_little_link_1",
        ],
        "link3_names": [
            "right_thumb_link_3",
            "right_index_link_3",
            "right_middle_link_3",
            "right_ring_link_3",
            "right_little_link_3",
        ],
        "link4_names": [
            "right_thumb_link_4",
            "right_index_link_4",
            "right_middle_link_4",
            "right_ring_link_4",
            "right_little_link_4",
        ],
        "urdf_subdir": "assets/gaia_hand20",
        "urdf_file": {
            "right": "gaiahand20_right.urdf",
            "left": "gaiahand20_left.urdf",
        },
        "mjcf_subdir": "assets/gaia_hand20",
        "mjcf_file": {
            "right": "gaiahand20_right_mujoco.xml",
            "left": "gaiahand20_left_mujoco.xml",
        },
        "num_fingers": 5,
        "neutral_qpos": [0.0] * 20,
    },
    # Inspire Hand: 12 joints, 6 mimic-driven, so only 6 independent DOF.
    # Non-thumb chain runs proximal -> intermediate -> tip (fixed), hence
    # link3 is the proximal and link4 the intermediate link.
    "inspire_hand": {
        "origin_link": "hand_base_link",
        "tip_links": ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"],
        "link1_names": [
            "thumb_proximal",
            "index_proximal",
            "middle_proximal",
            "ring_proximal",
            "pinky_proximal",
        ],
        "link3_names": [
            "thumb_proximal",
            "index_proximal",
            "middle_proximal",
            "ring_proximal",
            "pinky_proximal",
        ],
        "link4_names": [
            "thumb_intermediate",
            "index_intermediate",
            "middle_intermediate",
            "ring_intermediate",
            "pinky_intermediate",
        ],
        "urdf_subdir": "assets/inspire_hand",
        "urdf_file": {
            "right": "inspire_hand_right.urdf",
            "left": "inspire_hand_left.urdf",
        },
        "mjcf_subdir": "assets/inspire_hand",
        "mjcf_file": {"right": "inspire_hand_right_mujoco.xml"},
        "num_fingers": 5,
    },
    # Sharpa Wave: 22 DOF, thumb 5 and pinky 5, others 4.
    "sharpa_hand": {
        "origin_link": "right_hand_C_MC",
        "tip_links": [
            "right_thumb_fingertip",
            "right_index_fingertip",
            "right_middle_fingertip",
            "right_ring_fingertip",
            "right_pinky_fingertip",
        ],
        "link1_names": [
            "right_thumb_MC",
            "right_index_PP",
            "right_middle_PP",
            "right_ring_PP",
            "right_pinky_PP",
        ],
        "link3_names": [
            "right_thumb_PP",
            "right_index_MP",
            "right_middle_MP",
            "right_ring_MP",
            "right_pinky_MP",
        ],
        "link4_names": [
            "right_thumb_DP",
            "right_index_DP",
            "right_middle_DP",
            "right_ring_DP",
            "right_pinky_DP",
        ],
        "urdf_subdir": "assets/sharpa_hand",
        "urdf_file": {
            "right": "right_sharpa_wave.urdf",
            "left": "left_sharpa_wave.urdf",
        },
        "mjcf_subdir": "assets/sharpa_hand",
        "mjcf_file": {
            "right": "right_sharpa_wave.xml",
            "left": "left_sharpa_wave.xml",
        },
        "num_fingers": 5,
        "neutral_qpos": [0.0] * 22,
    },
    # Linker L20: 21 revolute joints, 5 mimic-driven (thumb_ip, 4x dip).
    # The GEORT distal frames sit at the DIP/IP joints, so tip_offsets place
    # the task points on the visible fingertip surfaces.
    "linker_l20": {
        "origin_link": "hand_base_link",
        "tip_links": [
            "thumb_distal",
            "index_distal",
            "middle_distal",
            "ring_distal",
            "pinky_distal",
        ],
        "tip_offsets": [
            [-0.0110, 0.0, 0.0277],
            [-0.0135, 0.0, 0.0188],
            [-0.0135, 0.0, 0.0188],
            [-0.0135, 0.0, 0.0188],
            [-0.0135, 0.0, 0.0188],
        ],
        "link1_names": [
            "thumb_metacarpals_base1",
            "index_metacarpals",
            "middle_metacarpals",
            "ring_metacarpals",
            "pinky_metacarpals",
        ],
        "link3_names": [
            "thumb_proximal",
            "index_middle",
            "middle_middle",
            "ring_middle",
            "pinky_middle",
        ],
        "link4_names": [
            "thumb_distal",
            "index_distal",
            "middle_distal",
            "ring_distal",
            "pinky_distal",
        ],
        "urdf_subdir": "assets/linker_l20",
        "urdf_file": {
            "right": "geort_right/linkerhand_l20_right.urdf",
            "left": "geort_left/linkerhand_l20_left.urdf",
        },
        # The MJCF sits beside the URDF rather than in its own subdirectory.
        "mjcf_subdir": "assets/linker_l20",
        "mjcf_file": {
            "right": "linker_l20_right_mujoco.xml",
            "left": "linker_l20_left_mujoco.xml",
        },
        "num_fingers": 5,
        "neutral_qpos": [0.0] * 21,
    },
}

__all__ = ["ROBOT_CONFIGS"]
