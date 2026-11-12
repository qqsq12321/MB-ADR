# Add a robot hand

[中文](ADDING_HAND.zh.md) · [Back to README](../README.md)

This guide uses a fictional five-finger model named `my_hand`. Replace every
example link and asset path with names from your model; this is not a seventh
bundled hand. Work from a full checkout and start with training and simulation.

## 1. Prepare the models

Keep the URDF, meshes and optional MuJoCo model together, for example:

```text
assets/my_hand/
  right.urdf
  right.xml
  meshes/
```

Training/FK use the URDF; MuJoCo visualisation uses MJCF. Training alone does not
require MJCF. Simulation requires a loadable MJCF; the project has no general
one-command URDF-to-MJCF converter.

- Use metres and radians. Check axes, origins and joint limits; an ordinary
  movable joint without limits is read as `[0, 0]`.
- Use a fixed-root kinematic tree. The FK handles fixed, rotational and prismatic
  joints, without free roots, ball joints or closed-loop mechanisms.
- Driven joints should share names, zero positions, directions and limits across
  URDF/MJCF. MuJoCo mapping matches names rather than document order.
- Declare coupling with URDF `<mimic joint="..." multiplier="..." offset="..."/>`.
  Its source must be an independent joint; chained mimic is unsupported.

## 2. Register task points

Add an entry to `ROBOT_CONFIGS` in [robot_configs.py](../mbadr/robot_configs.py).
This example assumes proximal, middle, distal and tip frames for each finger:

```python
"my_hand": {
    "origin_link": "palm",
    "tip_links": [
        "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip",
    ],
    "link1_names": [
        f"{f}_proximal" for f in ("thumb", "index", "middle", "ring", "pinky")
    ],
    "link3_names": [
        f"{f}_middle" for f in ("thumb", "index", "middle", "ring", "pinky")
    ],
    "link4_names": [
        f"{f}_distal" for f in ("thumb", "index", "middle", "ring", "pinky")
    ],
    "num_fingers": 5,
    "urdf_subdir": "assets/my_hand",
    "urdf_file": {"right": "right.urdf"},
    "mjcf_subdir": "assets/my_hand",
    "mjcf_file": {"right": "right.xml"},
},
```

All finger arrays use **thumb, index, middle, ring, pinky** order. Each named
frame must be a URDF link, rather than a joint name.

| Field | Meaning |
|---|---|
| `origin_link` | Palm reference frame |
| `tip_links` | Links containing fingertip task points |
| `link1_names` | Proximal frames, also used to orient the simulation scene |
| `link3_names` / `link4_names` | Links corresponding to human PIP / DIP task points |
| `tip_offsets` / `link3_offsets` / `link4_offsets` | Optional per-finger local `[x, y, z]` offsets in metres; zero by default |

Inspect the actual frame origins. If a tip frame already sits on the contact
point, do not add the same displacement again as `tip_offsets`. Offsets are
expressed in their link's local frame.

The canonical name `my_hand` is accepted automatically by `resolve_robot_name`.
For a short alias, extend `ROBOT_ALIASES` in [config.py](../mbadr/config.py).
This example registers only the right hand. To support the left, add its files
and check `_mirror_name`: a new manufacturer's entry does not automatically
inherit the `right_` → `left_` rule.

Four-finger models can set `num_fingers: 4`; the default mapping uses thumb
through ring and drops pinky. Other finger selections, orders or topologies
require changes to landmark indexing and target construction.

## 3. Add the retargeting YAML

Create `configs/pico4/pico4_my_hand.yaml`. The default path uses the **canonical
name**, rather than a short alias. This is a structural template, with values
that still need calibration for the new model:

```yaml
optimizer:
  type: AdaptiveOptimizerAnalytical
robot:
  type: my_hand
retarget:
  huber_delta: 2.0
  huber_delta_dir: 0.5
  norm_delta: 0.04
  w_pos: 5.0
  w_dir: 1.0
  w_full_hand: 1.0
  w_contact: 6.0
  alpha: 1.0
  pinch_scaling: 1.0
  lateral_scaling: 1.0
  segment_scaling:
    thumb: [1.0, 1.0, 1.0, 1.0]
    index: [1.0, 1.0, 1.0, 1.0]
    middle: [1.0, 1.0, 1.0, 1.0]
    ring: [1.0, 1.0, 1.0, 1.0]
    pinky: [1.0, 1.0, 1.0, 1.0]
  pinch_thresholds:
    index: {d1: 1.0, d2: 1.5}
    middle: {d1: 1.0, d2: 1.5}
    ring: {d1: 2.0, d2: 4.0}
    pinky: {d1: 2.0, d2: 4.0}
  mediapipe_rotation:
    right: {x: 0.0, y: 0.0, z: 0.0}
    left: {x: 0.0, y: 0.0, z: 0.0}
```

The four `segment_scaling` values scale the wrist-to-MCP, MCP-to-PIP, PIP-to-DIP
and DIP-to-TIP segments, which are accumulated into target points; they are not
motor gains. Inspect open-palm, fist and individual pinch poses
before tuning scales. `mediapipe_rotation` is an XYZ rotation in degrees after
wrist-frame normalisation. `d1/d2` are pinch thresholds in centimetres, with
`d1 < d2`. L20's calibrated values do not transfer automatically.

## 4. Inspect and train

After adding the entry and YAML, inspect parsed names, order and limits:

```bash
python - <<'PY'
from mbadr.chain import KinematicChain
from mbadr.config import HandSpec, default_config_path

spec = HandSpec.from_name("my_hand", "right")
chain = KinematicChain.from_urdf(spec.urdf_path)
print(default_config_path("my_hand"))
print("full joints:", chain.joint_names)
print("independent joints:", chain.independent_names)
print("limits:", chain.independent_limits())
PY

python scripts/verify_fk.py --robot my_hand --hand right
python scripts/verify_loss.py --robot my_hand --hand right --frames 40

python -m mbadr.train --robot my_hand --hand right \
    --data-npy data/qsq_pico4_merged_v1.npy \
    --steps 12000 --out out/net_my_hand.pt

python -m mbadr.evaluate --robot my_hand --hand right \
    --checkpoint out/net_my_hand.pt --data-npy data/qsq_pico4_merged_v1.npy \
    --limit 100 --out out/eval_my_hand.json
```

Training needs human landmarks, without robot joint-angle labels. Start with
bundled data, then capture poses representative of your task. Train a checkpoint
for the new hand; matching joint counts do not make another hand's weights
compatible. Retrain after changing task points, limits or calibration.

The verification commands need the `verification` extra. `--robot all` in
`verify_fk.py` and `verify_loss.py` currently uses fixed lists; add your hand to
include it in those checks. `verify_fk_torch.py` currently checks L20 only and
has no `--robot` option.

## 5. Replay, live simulation and hardware

Once MJCF is ready, start with recorded replay, then connect Pico:

```bash
python scripts/demo_mujoco.py --robot my_hand --hand right \
    --checkpoint out/net_my_hand.pt --recording pico4_check.pkl

# Start the Pico relay in another terminal; see the Pico guide
python scripts/demo_pico4.py --robot my_hand --hand right \
    --checkpoint out/net_my_hand.pt --device cpu
```

Inspect joint mapping, directions and fingertip positions for open-palm, fist,
individual finger flexion and pinch poses. Reuse [Pico setup](PICO_SETUP.md).
`KeypointRetargeter` returns URDF movable-joint order; simulation applies the
[MJCF name mapping](../mbadr/mjcf.py).

Model/simulation integration does not implement physical control. Each new hand
needs its own transport, joint-to-actuator mapping, zero/direction/limit
calibration and input-loss handling. The [L20 driver](../mbadr/l20.py) shows the
structure, but its registers and calibration are L20-specific.

Update the supported-hand list, model source records and `assets/manifest.json`.
For new public data/checkpoints, update `artifacts/manifest.json` too.
Full-repository checks are in [CONTRIBUTING](../CONTRIBUTING.md).
