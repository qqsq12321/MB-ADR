# Connect another input source

[中文](INPUT_SOURCES.zh.md) · [Back to README](../README.md)

Camera pipelines, other VR devices and custom network receivers can supply hand
landmarks through the Python API. `demo_pico4.py` and `mbadr-l20` still construct
`Pico4` directly, with no generic `--input` option. Write an acquisition entry
point for your source and reuse the inference interfaces below.

## 1. Match the landmark contract

[KeypointRetargeter](../mbadr/live.py) accepts a numeric `numpy.ndarray` of shape
`(21, 3)` in **metres**, in MediaPipe 21-landmark order:

| Indices | Landmarks, base to tip |
|---|---|
| 0 | Wrist |
| 1–4 | Thumb CMC, MCP, IP, TIP |
| 5–8 | Index MCP, PIP, DIP, TIP |
| 9–12 | Middle MCP, PIP, DIP, TIP |
| 13–16 | Ring MCP, PIP, DIP, TIP |
| 17–20 | Pinky MCP, PIP, DIP, TIP |

Apply these conversions in your acquisition adapter:

1. Reorder points using the SDK's landmark definitions. Truncating an arbitrary
   array to 21 points is insufficient; joint quaternions are not positions.
2. Divide millimetres by 1000, or centimetres by 100. Image pixels and normalised
   2D coordinates are not metric 3D landmarks.
3. Establish the real hand side, camera mirroring and coordinate-reflection
   conventions. Preprocessing removes global translation and rigid rotation,
   but reflection differences still matter. Inspect open-palm, thumb abduction
   and pinch poses.
4. You may subtract point 0 to centre the wrist. Preprocessing also centres it,
   so the input need not already be at the origin.
5. Drop missing, low-confidence or incomplete SDK samples. Do not replace them
   with zeros or relabel cached poses as new samples. Values must be finite,
   with a palm that has not collapsed to a point or a line.

Pico's 26-to-21 [JOINT_21_INDICES](../mbadr/pico4.py) applies to that SDK's
ordering only. Define a source-specific mapping for other devices.

[preprocess_keypoints](../mbadr/data.py) performs wrist-frame-to-MANO conversion,
the YAML calibration rotation and palm-length normalisation (wrist to middle
MCP is 0.092 m). `KeypointRetargeter` calls it automatically: **pass adapted raw
points, without preprocessing them a second time**. Save training data at the
same raw stage.

## 2. Run device-independent inference

Run this example from the repository root. It uses bundled landmarks to exercise
the API without connecting to Pico or a serial device:

```python
from pathlib import Path

import numpy as np
import torch

from mbadr.live import KeypointRetargeter

rt = KeypointRetargeter(
    checkpoint=Path("out/net_l20.pt"),
    hand="right",
    device=torch.device("cpu"),
    robot="l20",
    smooth=0.3,
)
frames = np.load("data/qsq_pico4_merged_v1.npy", allow_pickle=False)
for raw in frames:
    q = rt(raw)
    if q is not None:
        print(dict(zip(rt.chain.joint_names, q.tolist())))
        break
else:
    raise RuntimeError("No valid hand frame")
```

For your SDK, replace `raw` with one frame converted as above. The result
includes all movable joints after mimic expansion, in `rt.chain.joint_names`
order: 21 values in radians for L20. Prismatic joints in other models use
metres. Invalid frames return `None`.

Match the checkpoint, robot, hand side and calibration. Bundled checkpoints are
right-hand trained; setting `hand="left"` does not create a left-trained model.

## 3. Handle fresh samples

Keep the following metadata alongside landmarks. This is a suggested
acquisition contract, not an existing SDK base class:

| Information | Handling |
|---|---|
| `hand` | Must match the inference hand side |
| `active` / confidence | Reject lost or unreliable tracking before inference |
| `sequence` | Identifies each genuinely new sample; may include a session ID after reconnect |
| `received_at` | Local `time.monotonic()` time when the new frame arrives |

This callback example reuses `rt` from above. `points` contains raw positions
after index, unit and coordinate-convention conversion:

```python
import time

last_sequence = None

def on_sample(points, *, hand, active, sequence, received_at):
    global last_sequence
    age = time.monotonic() - received_at
    if not active or hand != rt.hand or not 0.0 <= age <= 0.5:
        return None
    if sequence == last_sequence:
        return None
    last_sequence = sequence
    q = rt(points)
    if time.monotonic() - received_at > 0.5:
        return None
    return q
```

Call it only for new samples; do not refresh `received_at` while polling a cache.
Check freshness again after inference, and send no new target on `None`.
`smooth=0.3` updates per valid fresh frame, so its time response depends on input
rate. Recreate the retargeter for a new session if its smoothing state should be
cleared after tracking loss.

Simulation output must use [build_mapping](../mbadr/mjcf.py) to convert URDF order
to MuJoCo qpos. Hardware needs the matching device driver.
[Hardware notes](HARDWARE_VALIDATION.md) describe input loss: pausing target
updates does not imply that the device has braked or released force.

## 4. Record training data

Save adapted, valid, raw positions before preprocessing as a floating-point
`(N, 21, 3)` `.npy` array in time order. Use one hand side per file. Filter
incomplete, non-finite and collapsed-palm frames, and retain the acquisition
index/unit conversion rules. Training applies the shared preprocessing, so
normalised MANO points should not be saved as raw input.

For an acquisition list named `raw_frames`:

```python
raw_frames = np.stack(raw_frames).astype(np.float32)
assert raw_frames.shape[1:] == (21, 3)
assert np.isfinite(raw_frames).all()
np.save("data/my_source_right.npy", raw_frames, allow_pickle=False)
```

If you reuse the existing calibration:

```bash
python -m mbadr.train --robot l20 --hand right \
    --data-npy data/my_source_right.npy --steps 12000 \
    --out out/net_l20_my_source.pt

python -m mbadr.evaluate --robot l20 --hand right \
    --checkpoint out/net_l20_my_source.pt --data-npy data/my_source_right.npy \
    --limit 100 --out out/eval_l20_my_source.json
```

The loader removes consecutive duplicate frames, then reserves the final 15%
for validation by default. The training segment needs at least 100 frames.
Capture open-palm, fist, individual flexion and pinch poses across sessions.
Matching the format enables integration; evaluate cross-device accuracy and
retrain on source-specific data when needed.

## 5. Use source-specific calibration

If adapted input needs different scales or `mediapipe_rotation`, copy a config:

```bash
mkdir -p configs/my_source
cp configs/pico4/pico4_linker_l20.yaml configs/my_source/my_source_linker_l20.yaml

# Edit calibration values in the new YAML, then train
python -m mbadr.train --robot l20 --hand right \
    --config configs/my_source/my_source_linker_l20.yaml \
    --data-npy data/my_source_right.npy --steps 12000 \
    --out out/net_l20_my_source.pt
```

`default_config_path("l20", "my_source")` resolves this path. For live inference,
explicitly pass the same YAML used for training as `config_path`:

```python
from pathlib import Path

import numpy as np
import torch

from mbadr.live import KeypointRetargeter

rt = KeypointRetargeter(
    checkpoint=Path("out/net_l20_my_source.pt"),
    hand="right",
    device=torch.device("cpu"),
    robot="l20",
    smooth=0.3,
    config_path=Path("configs/my_source/my_source_linker_l20.yaml"),
)
raw = np.load("data/my_source_right.npy", allow_pickle=False)[0]
q = rt(raw)
if q is None:
    raise RuntimeError("Invalid hand frame")
print(dict(zip(rt.chain.joint_names, q.tolist())))
```

The retargeter rejects invalid points and smooths independent joints. Acquisition
still needs the freshness handling in section 3. Pass the same YAML to evaluation
so the network and NLopt reference use matching objectives:

```bash
python -m mbadr.evaluate --robot l20 --hand right \
    --config configs/my_source/my_source_linker_l20.yaml \
    --checkpoint out/net_l20_my_source.pt --data-npy data/my_source_right.npy \
    --limit 100 --out out/eval_l20_my_source.json
```

`demo_mujoco.py`, `demo_pico4.py` and `mbadr-l20` also accept `--config`.
The latter two still acquire from Pico; use your own entry point for other
devices. Omitting a config preserves the Pico defaults, without inferring YAML
from the checkpoint; a missing explicitly selected file raises an error.
These custom-configuration improvements are unreleased changes on `main`.
The published `v0.1.0b1` interfaces follow that tagged source.
