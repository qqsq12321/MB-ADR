# Changelog

## 0.1.0b1 — unreleased beta candidate

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
