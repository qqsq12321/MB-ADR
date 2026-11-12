# Beta candidate validation

This record distinguishes local software validation from hosted CI and physical
device validation. Local candidate checks completed on 2026-11-12.

## Scope

- Linux/Python 3.10, clean CPU virtual environments, with user-site and ROS Python
  paths disabled.
- Unit tests, loopback Pico packets, pseudo-terminal Modbus, numerical FK/loss,
  one-step training, artifact integrity and wheel installation.
- No physical serial port is opened and no real hand actuation is performed.
- The GitHub Actions workflow is prepared locally; no hosted result is claimed
  before it runs on a pushed commit.

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

## Results

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

The GitHub Actions workflow contains these software checks but has not been
observed on the remote commit yet. Source changes after the exported tree are
limited to validation/environment records, notice whitespace and the matching
bounded-evaluation CI command; runtime module bytes are checked against the
tested wheel before preparing the candidate commit.

Physical status remains in [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md).
No real L20 calibration, actuation, stop behaviour or headset/APK pairing is
claimed by the software results above.
