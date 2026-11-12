# Changelog

## Unreleased

### Fixed

- Custom evaluation YAML now defines both the network objective and the NLopt
  reference. Evaluation and recorded NLopt replay reuse that configured reference
  instead of loading the default Pico objective again.

### Added

- Optional `config_path` for `KeypointRetargeter` and `--config` for recorded
  replay, live Pico simulation and L20 teleoperation.
- Hand/input-source integration guides in English and Chinese.

### Documentation

- Recorded the published beta, hosted CI results and the distinction between the
  release snapshot and later changes on `main`.

## [0.1.0b1](https://github.com/qqsq12321/MB-ADR/releases/tag/v0.1.0b1) — 2026-11-12

Published as a GitHub beta prerelease from commit
`8866bf0037a6d47de8a2e97b329173908aee9748`, with a verified code wheel and
`SHA256SUMS`. The unreleased changes above are not included in that snapshot.

### Added

- Display-free Pico -> neural checkpoint -> L20 V10 RS485 entry point, with
  fresh-sample processing, default smoothing 0.3 and tracking timeout handling.
- Bundled Pico discovery/relay daemon and XRoboToolkit sender download/setup
  documentation; simulation and hardware share the receiver protocol.
- Python package metadata, optional dependency groups, installed CLI commands,
  CPU constraints and a two-version PyTorch CI workflow.
- Protocol, mapping, virtual-serial, loopback, checkpoint and integration checks.
- Contributor guidance, issue/PR templates, license scope, artifact checksums and
  explicit hardware validation status.

### Changed

- Bundled checkpoints contain only tensors and primitive metadata. Every network
  parameter and joint limit was preserved during the metadata migration.
- Training writes format-2 checkpoints; inference explicitly uses
  `torch.load(weights_only=True)` on CPU before moving the model to the device.
- Simulation smoothing now defaults to 0.3, as requested by the maintainer.
- Both READMEs document first-run steps, parameter meanings and troubleshooting.

### Compatibility

- Old NumPy-containing checkpoints need explicit migration with
  `scripts/migrate_checkpoint.py`; use it only for trusted local files.
- The code wheel omits robot assets, recordings and pretrained weights. Use an
  editable full checkout, or set `MBADR_PROJECT_ROOT` to one when using a wheel.
- L20 physical control is experimental: automated tests do not establish physical
  calibration, force, timing or emergency-stop performance.

### Earlier history

The original six-hand training/evaluation and simulation implementation preceded
the versioned beta candidate. No earlier semantic-version releases are claimed.
