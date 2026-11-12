# Validation record

This record distinguishes local software validation from hosted CI and physical
device validation. Local beta checks completed on 2026-11-12.

## Published beta and hosted CI

[`v0.1.0b1`](https://github.com/qqsq12321/MB-ADR/releases/tag/v0.1.0b1) was
published on 2026-11-12 from commit
`8866bf0037a6d47de8a2e97b329173908aee9748`. Its
[GitHub Actions CPU checks](https://github.com/qqsq12321/MB-ADR/actions/runs/43162653034)
completed successfully with PyTorch 2.5.1 and 2.6.0 on Ubuntu 22.04 / Python 3.10.
These hosted results cover the named release commit; later changes on `main`
have their own runs in the
[CPU workflow](https://github.com/qqsq12321/MB-ADR/actions/workflows/ci.yml?query=branch%3Amain).

## Scope

- Linux/Python 3.10, clean CPU virtual environments, with user-site and ROS Python
  paths disabled.
- Unit tests, loopback Pico packets, pseudo-terminal Modbus, numerical FK/loss,
  one-step training, artifact integrity and wheel installation.
- No physical serial port is opened and no real hand actuation is performed.
- Hosted CI passed for the release commit as linked above. The local environment
  and wheel checks below provide additional validation of that release baseline.

## Reproduction

Runtime source, assets and data were exported with `git archive` from staged tree
`31e62276cf154551487887596fc3b39329e45b1b` into a separate directory without
`.git`, user packages or dependencies on sibling projects. Two new Python 3.10.21
virtual environments installed the project and its optional groups from that
export. `PYTHONPATH` and `LD_LIBRARY_PATH` were cleared; user-site packages and
CUDA were disabled. This is Python-environment isolation on the existing Linux
host, not a claim that a fresh OS image or hosted GitHub runner was tested.

Observed complete dependency versions are recorded for
[PyTorch 2.5](environments/linux-py310-torch25.txt) and
[PyTorch 2.6](environments/linux-py310-torch26.txt). The portable constraints in
`constraints/` pin direct dependencies; the observed environment records also
list the resolved transitive versions.

## Release baseline results

| Check | PyTorch 2.5.1+cpu | PyTorch 2.6.0+cpu |
|---|---|---|
| Dependency consistency (`pip check`) | Pass | Pass |
| Unit/integration suite | 32 tests passed | 32 tests passed |
| NumPy FK/Jacobian vs Pinocchio | All 6 hands passed | All 6 hands passed |
| Torch FK forward/autograd (L20) | Pass | Pass |
| Torch vs NumPy loss/gradient, 40 frames per hand | All 6 hands passed | All 6 hands passed |
| One training step, batch 8, checkpoint saved/reloaded | Pass | Pass |
| Bundled L20 checkpoint evaluated on 8 held-out frames | Pass | Pass |
| README links/CLI examples and artifact integrity | Pass | Pass |
| Installed wheel, four CLIs `--help` outside checkout | Pass | Pass |
| Installed wheel, recorded landmarks -> finite 21-joint output outside checkout | Pass | Pass |

The wheel was loaded from each virtual environment's `site-packages`, with
`MBADR_PROJECT_ROOT` pointing at the exported checkout solely for assets/data.
Both sdist and wheel builds passed. Normal inference never enabled unrestricted
pickle loading; tests also verified that unsupported pickle callbacks were not
executed. All seven bundled checkpoint migrations preserved every parameter and
joint limit exactly.

For the release candidate, changes between the exported tree and the tagged
commit were limited to validation/environment records, notice whitespace and
the matching bounded-evaluation CI command. Runtime module bytes were checked
against the tested wheel before preparing the candidate commit.

## Unreleased custom-configuration checks

The regression checks use a temporary YAML with altered scales, pinch thresholds,
loss weights and rotation. They check agreement of Torch/NumPy targets and loss,
reuse of the configured reference by the evaluation CLI, explicit live
configuration, missing-file errors, simulation adapters and the complete
Pico-loopback-to-L20 dry-run path with a custom YAML. These changes are on `main`,
outside the published `v0.1.0b1` snapshot.

On 2026-11-12 the complete suite passed **37 tests** in each of the existing
isolated Python 3.10 CPU environments (PyTorch 2.5.1 and 2.6.0), with user-site
and ROS paths disabled:

```bash
python -m unittest discover -s tests -v
```

The release-baseline table above retains its original 32-test counts.

Both environments also passed a one-step training run with custom calibration,
reload of its checkpoint, an actual CLI evaluation on eight held-out frames and
the bilingual guide's explicit-config live inference example. A six-hand default
inference/configuration smoke check passed with PyTorch 2.6.0. Repository links,
documented README options and the unchanged artifact/asset manifests passed
`scripts/check_repository.py`.

Physical status remains in [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md).
No real L20 calibration, actuation, stop behaviour or headset/APK pairing is
claimed by the software results above.
