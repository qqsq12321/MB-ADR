# License scope and third-party sources

The root [MIT license](LICENSE) covers original MB-ADR code, documentation,
configuration and maintainer-owned data/checkpoints. On 2026-11-12 the maintainer
confirmed that the shipped recordings/keypoints are self-collected and the
weights self-trained and may be distributed. Their byte-level inventory is
[artifacts/manifest.json](artifacts/manifest.json).

Robot model files are third-party material and retain their upstream notices.
The following records describe available evidence, not a blanket MIT grant for
`assets/`. The maintainer requested that unresolved asset-license research be
deferred; those entries remain explicitly unconfirmed.

| Scope | Source / evidence | Recorded terms |
|---|---|---|
| Pico receiver/relay and L20 protocol/mapping adaptation | AnyDexRetarget; original Copyright (c) 2025 Shiquan Qiu | [MIT notice](third_party/licenses/AnyDexRetarget-MIT.txt) |
| `assets/shadow_hand/` | [MuJoCo Menagerie Shadow Hand](https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand), existing local source README | [Apache-2.0](assets/shadow_hand/LICENSE) |
| `assets/wuji_hand/` | [Wuji hand description](https://github.com/wuji-technology/wuji-hand-description) | [Upstream MIT notice](assets/wuji_hand/LICENSE); exact imported revision not recorded |
| `assets/sharpa_hand/` | [Sharpa model repository](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml) | [Upstream Apache notice](assets/sharpa_hand/LICENSE.txt), [full Apache-2.0 text](third_party/licenses/Apache-2.0.txt); exact imported revision not recorded |
| `assets/linker_l20/` | [LinkerHand URDF](https://github.com/linker-bot/linkerhand-urdf) and the GEORT-derived model named in the local configuration | Vendor repository declares Apache-2.0; GeoRT distribution declares [CC BY-NC 4.0](third_party/licenses/GeoRT-CC-BY-NC-4.0.txt). Exact file-by-file derivation remains unconfirmed; retain the noncommercial notice for GEORT-derived material |
| `assets/inspire_hand/` | [Existing source attribution](assets/inspire_hand/LICENSE.txt), Inspire model modified by dex_urdf authors | The local file records provenance, not a complete license grant; unconfirmed |
| `assets/gaia_hand20/` | [Existing source record](assets/gaia_hand20/SOURCE.md), Stellarobot ROS packages | Package metadata reportedly declares BSD; exact terms/copyright notice are not present here; unconfirmed |
| XRoboToolkit headset application | [XR-Robotics](https://github.com/XR-Robotics), [client](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client) | External download, not bundled or relicensed |

[assets/manifest.json](assets/manifest.json) records component paths and tree
hashes so notices can be tied to the imported material. `NOASSERTION` means that
the available source records do not establish complete terms for that scope.
No missing terms are replaced with the root MIT license.

Local robot-model changes include project-relative mesh references, fingertip
frames/offsets, mimic handling, generated MuJoCo models and viewing scenes. They
do not change the attribution or upstream terms of the source model files.

The Python wheel contains code and notices only. Robot models, datasets and
pretrained checkpoints remain in the full source checkout and are not silently
included in a package described as MIT code.
