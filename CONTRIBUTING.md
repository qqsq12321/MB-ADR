# Contributing to MB-ADR

Please open an issue describing the problem or intended behaviour before a large
change. Small bug fixes and documentation corrections can go directly to a pull
request. Include a concrete reproduction and the commands you ran to validate it.

## Development environment

Clone the full repository: assets, recordings and checkpoints are used by the
integration checks. A Linux CPU environment needs no headset or physical robot:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[all]' -c constraints/py310-torch25.txt
python -m pip check
```

The constraints pin tested direct dependencies, not every transitive dependency.
Use `constraints/py310-torch26.txt` with the 2.6.0 CPU wheel for the second CI job.
`pin` is the PyPI package that supplies the `pinocchio` import.

## Checks before a pull request

```bash
python -m unittest discover -s tests -v
python scripts/verify_fk.py --robot all
python scripts/verify_fk_torch.py --device cpu
python scripts/verify_loss.py --robot all --frames 40
python scripts/check_repository.py
python -m build
```

Tests use loopback sockets and newly allocated pseudo-terminals. They must never
open `/dev/ttyUSB*`, broadcast discovery packets on the LAN, or move physical
hardware. Keep device-specific tests opt-in and document actual hardware results
in `docs/HARDWARE_VALIDATION.md`.

## Changes to retargeting or robot models

Start with [adding a hand](docs/ADDING_HAND.md) or
[connecting another input source](docs/INPUT_SOURCES.md) for integration steps.
中文版：[新增灵巧手](docs/ADDING_HAND.zh.md) · [其他输入源](docs/INPUT_SOURCES.zh.md)。

- Keep NumPy and Torch objectives numerically consistent.
- Preserve joint-name mappings, independent/mimic coupling and joint limits.
- Reuse `preprocess_keypoints` for both training and live input.
- Add regression checks for numerical or protocol changes.
- Include source and license records for new third-party assets.
- Update both README languages when CLI behaviour changes.

Checkpoints use tensors and primitive metadata and load with
`weights_only=True`. Do not add an automatic unrestricted-pickle fallback.
The standalone legacy migration script is only for files whose origin is trusted.

## Scope and licenses

Original contributions use the root MIT license. Third-party robot assets retain
their own terms; see `THIRD_PARTY_NOTICES.md`. Do not replace upstream notices
with the project's license or add recordings/weights without permission to share.

发布前请同步检查中英文说明、模型与数据的来源，以及真机验证状态；没有实测的功能应标为实验性。
