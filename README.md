# MB-ADR

[中文](README.zh.md) | English

**M**odel-**B**ased **A**ny**D**ex**R**etargeting.

End-to-end, **model-based** neural retargeting for dexterous robot hands.

Replaces a per-frame NLopt solve with a learned network, while keeping the
optimiser's objective and the robot's kinematics as the source of truth.

PC-side code and assets are self-contained: hand definitions, URDF/MJCF,
retargeting configs, the NLopt reference solver, differentiable FK, the Pico 4
receiver and the L20 RS485 driver all live here. No other project or vendor SDK
is imported at runtime; Python dependencies are in `environment.yml` and
`requirements.txt`. The XR-Robotics/XRoboToolkit headset APK is installed separately; a PC relay
service is included in this project.

![Six hands retargeting the same keypoint recording](docs/demo_grid.gif)

*l20, Shadow, Wuji, Sharpa, Gaia, Inspire — same input, each hand's own
network, same scene (background and palm-forward orientation unified in
`mbadr/scene.py`).*

## Quick start

Run commands from the **repository root**, after preparing the `mbadr`
environment using [Environment](#environment). Checkpoints for all six hands
are included; replay, inference and L20 hardware control need no prior training.

| Goal | Entry point | Additional requirements |
|---|---|---|
| Recorded replay | `scripts/demo_mujoco.py` | Desktop with a working OpenGL window |
| Live Pico simulation | `scripts/demo_pico4.py` | Pico sender app, network, OpenGL window |
| Physical L20 control | `scripts/teleop_l20.py` | Pico sender app and L20 V10 RS485 connection; no window |

```bash

# Recorded replay: no headset or physical hand needed
python scripts/demo_mujoco.py --robot l20 --hand right --device cpu \
    --recording pico4_check.pkl --loop

# Live simulation: prepare the Pico data connection described below
python scripts/demo_pico4.py --robot l20 --hand right --device cpu

# Check hardware mapping: print registers without opening serial
python scripts/teleop_l20.py --hand right --device cpu --dry-run
```

Next: [Pico connection](#pico-4-data-connection) · [Physical L20](#physical-l20-pico-4--rs485) ·
[Smoothing and logs](#smoothing-and-runtime-status) · [Troubleshooting](#troubleshooting) ·
[Train](#train) · [Evaluate](#evaluate).
Close the simulation window to exit; use Ctrl+C for hardware control.

## Environment

Version `0.1.0b1` is a beta candidate. Install from a full Git checkout. CPU reproduction:

```bash
git clone https://github.com/qqsq12321/MB-ADR.git
cd MB-ADR
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[all]' -c constraints/py310-torch25.txt
python -m pip check
```

An existing Conda `mbadr` environment can be activated and used for the editable
installation above. New Conda environments can still use
`conda env create -f environment.yml`; install PyTorch separately. For a compatible
NVIDIA driver, replace the CPU installation command with the CUDA 12.1 build:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

`pyproject.toml` is the dependency source: core inference/training plus optional
`simulation`, `hardware`, `verification` and `dev` groups; `all` installs all of them.
The PyPI package supplying Pinocchio is `pin`. The CPU constraints
`constraints/py310-torch25.txt` / `constraints/py310-torch26.txt` target PyTorch
2.5.1 / 2.6.0 and constrain tested direct dependencies, not every transitive dependency.
See the [official PyTorch installation archive](https://docs.pytorch.org/get-started/previous-versions/).

Installed CLIs are `mbadr-train`, `mbadr-evaluate`, `mbadr-l20` and `mbadr-pico-relay`.
The wheel contains code/notices, not robot assets, recordings or weights. For a
non-editable installation set `MBADR_PROJECT_ROOT` to a full checkout containing
`assets/configs/data/out`. Bundled checkpoints now contain only tensors and
primitive metadata; inference explicitly uses `weights_only=True`. See
[checkpoint documentation](out/README.md) for trusted legacy-file migration.

## What "model-based" means here

The supervision is an **analytic differentiable forward-kinematics model**, not
a learned surrogate and not joint labels:

```
human keypoints ──► adaptive targets ──► loss ◄── analytic FK ◄── network output
                        (Torch)           │
                                          └─── gradients via autograd
```

Three consequences of that choice, each of which matters for fingertip accuracy:

| | learned-FK approaches | this |
|---|---|---|
| FK used in training | MLP surrogate fitted to sampled poses | exact URDF chain, matches Pinocchio to 1e-16 m |
| mimic joints | predicted as independent DOF, corrected post hoc | expanded inside FK, always consistent |
| joints the network predicts | all nq, including dependent ones | only the independent DOF |

The difference is not cosmetic. If a network predicts the distal joints and they
are afterwards overwritten from the PIPs, the fingertip moves to a place the
network never optimised for. Predicting only the independent DOF removes that
failure mode by construction, since the FK encodes the coupling and the loss
sees a pose consistent with that model. Physical zero positions, command
calibration and real-world errors still require hardware mapping and validation.

## Supported hands

| `--robot` | nq | independent DOF | mimic | note |
|---|---|---|---|---|
| `l20` | 21 | 16 | 5 | DexForce Linker L20 |
| `shadow` | 22 | 22 | — | Shadow Hand |
| `wuji` | 20 | 20 | — | Wuji Hand |
| `sharpa` | 22 | 22 | — | Sharpa Wave |
| `gaia` | 20 | 16 | 4 | Gaia Hand20 |
| `inspire` | 12 | 6 | 6 | Inspire Hand, most under-actuated |

Hand definitions live in `mbadr/robot_configs.py` with their assets under
`assets/` and configs under `configs/pico4/`. Adding a hand requires its models,
task-point definitions, scale/frame calibration and a trained checkpoint. FK,
loss, training and evaluation share the existing implementations.

All six rows have configs, assets and `out/net_<robot>.pt`; the bundled
checkpoints are right-hand trained. Physical output currently supports L20
RS485 only; the other hands support training, evaluation and simulation.
Use the short names in the table for demos: the default checkpoint filename
uses the literal `--robot` value.

## Extending the project

- [Add a robot hand](docs/ADDING_HAND.md): prepare URDF/MJCF, register task points and mimic joints, calibrate YAML, train and simulate.
- [Connect another input source](docs/INPUT_SOURCES.md): the 21-point contract, units/frames, inference API, sample freshness and training on your own data.

The live CLIs currently use Pico; other inputs connect through the Python API.
Physical control of a new robot hand requires its own output driver.

## Layout

```
mbadr/
  chain.py             URDF -> flat kinematic description, mimic bookkeeping
  config.py            HandSpec: per-hand links/offsets/URDF/MJCF resolution
  robot_configs.py     the hand table (links, offsets, asset paths)
  mediapipe.py         landmark frame conversion (wrist frame -> MANO)
  fk_numpy.py          NumPy FK + analytic Jacobians
  fk_torch.py          batched differentiable FK, mimic expansion as a linear map
  reference.py         NumPy + NLopt reference solver (the "ground truth")
  loss_torch.py        the same objective in Torch, for training
  data.py              loading, dedup, anatomically-checked augmentation
  model.py             amortising MLP, joint limits as a hard constraint
  mjcf.py              URDF joint order <-> MuJoCo qpos mapping
  scene.py             shared viewing scene: background, upright palm-forward pose
  pico4.py             Pico 4 input driver (relay / direct)
  live.py              display-free network inference in URDF joint order
  checkpoint.py        restricted loading of tensor-only checkpoints
  pico_relay.py        XRoboToolkit discovery, receiver and local relay
  l20.py               L20 RS485/Modbus protocol and calibrated hardware mapping
  train.py             self-supervised training loop
  evaluate.py          net vs NLopt on identical keypoints
scripts/
  verify_fk.py         NumPy FK/Jacobian vs Pinocchio
  verify_fk_torch.py   Torch FK vs Pinocchio (forward, autograd Jacobian, CUDA)
  verify_loss.py       Torch loss vs NumPy reference (values and gradients)
  demo_mujoco.py       replay a recording through the net in MuJoCo
  demo_pico4.py        live Pico 4 input -> net -> MuJoCo
  teleop_l20.py        live Pico 4 input -> net -> physical L20 (no window)
  check_nlopt_budget.py  is the net's advantage real or NLopt under-converged?
  summarise.py         aggregate per-hand evaluation results
assets/                URDF / MJCF / meshes for the six hands
configs/pico4/         retargeting YAML per hand
data/                  keypoint dataset (.npy) and recordings/ (.pkl, for demos)
out/                   checkpoints and evaluation JSON
tests/                 mapping, Modbus, Pico validity and complete dry-run tests
```

## Train

```bash
python -m mbadr.train --robot l20 --data-npy data/qsq_pico4_merged_v1.npy \
    --steps 12000 --out out/net_l20_custom.pt
```

Training needs only keypoints; no joint labels are collected or used. With
`--data-npy` the last 15 % of the capture is held out by time, so validation is
contiguous with, but disjoint from, training.

Input is raw `(N, 21, 3)` landmarks in metres; the loader applies the same
preprocessing as live input. Training defaults to `out/net.pt`; this example
saves `net_l20_custom.pt` explicitly to preserve the bundled model. Pass the
resulting path with `--checkpoint` when using a custom model in a demo or on hardware.

## Evaluate

```bash
python -m mbadr.evaluate --robot l20 --checkpoint out/net_l20.pt \
    --data-npy data/qsq_pico4_merged_v1.npy --out out/eval_l20.json
python scripts/summarise.py
```

Frames are stratified by pinch coverage: pinch appears in roughly a third of
real frames but in contiguous runs, so a naive first-N slice can contain none of
it and silently leave the metric that matters most unmeasured.

Replace `--checkpoint` to evaluate a custom model. The summary script reads
`out/eval_<robot>.json`, hence the explicit `--out`; the evaluation module's
own default is `out/eval.json`.

## Visualise

```bash
# recorded replay, no hardware needed
python scripts/demo_mujoco.py --robot shadow --recording pico4_check.pkl

# live Pico 4 (relay daemon on 127.0.0.1:63902, or --mode direct)
python scripts/demo_pico4.py --robot l20 --hand right
```

## Pico 4 data connection

The headset sender is **XR-Robotics / XRoboToolkit**. Obtain the upstream
[v1.1.1 APK](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk)
or inspect [other releases](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases).
See [Pico setup and protocol](docs/PICO_SETUP.md). The maintainer's installed APK
revision is not yet recorded. Connect the headset and PC over the LAN, select
the PC address in the app, and enable **Tracking — Hand** and **Send**.

```bash
# Terminal 1: included relay, keep running
python scripts/pico4_daemon.py

# Terminal 2: default relay mode
python scripts/demo_pico4.py --robot l20 --hand right --device cpu
```

| Mode | PC behaviour | Default ports | Requirement |
|---|---|---|---|
| `relay` (default) | Connect to the bundled or another compatible relay | `127.0.0.1:63902/TCP` | Start relay first; demos can restart independently |
| `direct` | The application listens and broadcasts the PC IP | `63901/TCP`, `29888/UDP` | No relay; run one receiving application |

```bash
# Alternatively use a single application's direct receiver without the relay
python scripts/demo_pico4.py --robot l20 --hand right --mode direct --device cpu
```

The hardware entry point also supports `--mode direct`. Relay, direct mode and
the upstream PC-Service must not own TCP 63901 simultaneously. Use `--relay-host`
/ `--relay-port` for a remote relay, or `--port` / `--broadcast-port` to change
direct ports in agreement with the headset. The receiver reads
`Hand.leftHand/rightHand`, `isActive` and 26 `HandJointLocations`; `p` is a
comma-separated pose string whose first three coordinates become 21
wrist-centred landmarks in metres. Screen casting or system hand tracking alone
does not stream this data.

## Physical L20 (Pico 4 + RS485)

This uses this project's `out/net_l20.pt`, without AnyDexRetarget, a vendor SDK,
or a MuJoCo window. Install `pyserial` in an existing environment with
`python -m pip install pyserial`. Defaults match the existing L20 V10 RS485
driver: `/dev/ttyUSB0`, 460800 baud, 8N1, slave `0x29` (right) / `0x2A` (left).
This entry point does not implement CAN. The headset must run a compatible
tracking sender; relay mode connects to the bundled or an existing `127.0.0.1:63902` service.

```text
Pico landmarks -> shared preprocessing -> MLP -> smooth 16 independent joints
  -> expand 5 mimic joints -> calibrated mapping by name -> 30 registers -> RS485
```

`mbadr/l20.py` implements joint directions, zero offsets, piecewise thumb
calibration, Modbus CRC and reply validation. This sends position targets;
it does not read physical encoders or force feedback.

```bash
conda activate mbadr

# Print the 30 position registers without opening any serial device
python scripts/teleop_l20.py --hand right --device cpu --dry-run --duration 10

# Open RS485 and control the physical hand
python scripts/teleop_l20.py --hand right --device cpu \
    --l20-port /dev/ttyUSB0 --l20-baudrate 460800 --l20-slave-id 0x29
```

Use the port and address configured on your hand. A stable path under
`/dev/serial/by-id/` can also be used for the serial device.

| Option | Default | Meaning |
|---|---|---|
| `--checkpoint` | Repository's `out/net_l20.pt` | L20 network trained in this project |
| `--smooth` | `0.3` | EMA coefficient on independent joints |
| `--l20-speed` / `--l20-current-limit` | Both `200` | Device register values, 0–255, not physical units |
| `--hz` | `60` | Maximum fresh-sample processing and command rate |
| `--input-timeout` | `0.5` seconds | Pause updates when the selected hand's received data is stale |
| `--duration` | `0` | Runtime; 0 means run continuously |
| `--dry-run` | Off | Compute/print registers without opening serial |
| `--l20-print-registers` | Off | Print command values |
| `--l20-open-on-exit` | Off | Send an open-palm command at shutdown |
| `--l20-clear-faults` | Off | Clear fault registers during startup |

Omit `--device cpu` to select CUDA automatically when available.

Invalid, disconnected or stale tracking (`--input-timeout 0.5` seconds) pauses
target updates; the device retains its last target and updates resume when
tracking returns. Pausing target updates is not a hardware emergency stop.
Ctrl+C closes the connection without an extra pose by default.
`--l20-open-on-exit` explicitly sends an open-palm command at shutdown;
`--l20-clear-faults` explicitly clears fault registers during startup. The bundled
checkpoint is right-hand trained; left-hand use requires a matching
`--checkpoint`, and the entry point checks the hand side. Startup configures
speed and current limits; only one controller should own the serial port.
Hardware-free protocol, mapping and virtual-serial tests do not replace
validation on the physical device.

Hardware-free validation covers mapping, Modbus frames/replies, Pico freshness,
and the complete dry-run pipeline:

```bash
python -m unittest discover -s tests -v
```

## Smoothing and runtime status

Both live entry points default to `--smooth 0.3`. For values **between 0 and 1**,
smaller coefficients give stronger smoothing and more lag; `0` is a special
value that disables smoothing. Try `--smooth 0.1` for small stationary jitter,
or `0.5` for a faster response. The YAML `lp_alpha` does not control these
scripts; use the CLI `--smooth` option.

| Entry point | Where and when smoothing runs | Log interpretation |
|---|---|---|
| `demo_pico4.py` | Expanded, mapped MuJoCo joints, every viewer iteration | `applied ... fps` counts applications of the cached pose, not headset sampling |
| `teleop_l20.py` | Independent joints, at most once per fresh valid Pico sample | `input ... fps` counts processed samples; `commands ... Hz` counts changed register outputs |

The same coefficient can therefore feel different between entry points. A
stationary hand may show zero `commands` because duplicate commands are skipped;
this does not imply disconnection. Use each entry point's `--help` for all options.

## Troubleshooting

| Symptom | Check |
|---|---|
| `No module named torch` or `serial` | Activate `mbadr`; check `which python` and install dependencies into that environment |
| `checkpoint not found` | Run from the repository root; simulation defaults to relative `out/net_<robot>.pt`, or pass `--checkpoint` explicitly |
| Waiting for tracking / `input 0.0 fps` | Confirm the sender streams the selected hand; an open relay port does not prove skeleton frames are arriving |
| `Address already in use` | Relay or another direct instance may own TCP 63901; use relay mode or stop the duplicate instance |
| Missing, inaccessible or busy serial port | Check `/dev/serial/by-id/`, device permissions and competing controller processes |
| Modbus timeout, CRC or reply error | Check power, RS485 wiring, baudrate and actual slave id; failed transport validation stops output and closes the connection |
| `Checkpoint was trained for right; requested left` | Use a checkpoint trained for that side; changing `--hand` does not convert a right-hand model |
| Legacy NumPy checkpoint cannot load in restricted mode | Bundled files are migrated; convert trusted old files using the [migration instructions](out/README.md); inference has no unrestricted-pickle fallback |

`GLX: Failed to create context` / `could not create window` concerns the
simulation's OpenGL window. Mesa software rendering is an alternative when
Mesa is installed; this was verified locally:

```bash
__GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1 \
python scripts/demo_pico4.py --robot l20 --hand right --device cpu
```

`--device cpu` alone does not change the window renderer. If `nvidia-smi` reports
`Driver/library version mismatch`, resolve the kernel/userspace driver mismatch
as described in the [Ubuntu documentation](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/#driverlibrary-version-mismatch-error).
The hardware entry point creates no OpenGL window and needs no Mesa variables.

## Contributing and release status

See [CONTRIBUTING.md](CONTRIBUTING.md) for development/testing and
[CHANGELOG.md](CHANGELOG.md) for changes. CI covers Python 3.10 with two PyTorch
CPU versions; [validation results](docs/VALIDATION.md) distinguish local checks
from hosted CI. Physical L20 control remains experimental; see the status and
run-record template in [hardware validation](docs/HARDWARE_VALIDATION.md).

Original code, configuration, maintainer-owned data and trained weights use the
[MIT license](LICENSE). Robot assets retain their terms in
[third-party notices](THIRD_PARTY_NOTICES.md). Exact data/model checksums are in
[artifacts/manifest.json](artifacts/manifest.json). See
[release steps](docs/RELEASING.md) and [citation metadata](CITATION.cff).
