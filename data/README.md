# Bundled hand-keypoint data

The maintainer confirmed on 2026-11-12 that these are self-collected recordings
and keypoint arrays with permission for public distribution under the project's
[MIT license](../LICENSE). No joint-angle labels are used in training.

| File | Role |
|---|---|
| `qsq_pico4_merged_v1.npy` | Raw metric `(N, 21, 3)` single-hand landmarks used for training/evaluation |
| `recordings/pico4_check.pkl` | Pico recording used by replay and integration checks |
| `recordings/avp_20261019_103754.pkl` | AVP landmark recording for replay/evaluation |

Recordings contain frame dictionaries with `left_fingers` / `right_fingers` and
timing metadata. The shared loader performs wrist-frame conversion, calibration
and scale normalization. `--data-npy` reserves the last 15% by time by default.

Exact sizes and SHA-256 hashes are in
[artifacts/manifest.json](../artifacts/manifest.json). New data should retain its
source, coordinate convention, units and permission to distribute. Pickle replay
files are only appropriate for trusted recordings, such as the bundled files
whose hashes are recorded here.
