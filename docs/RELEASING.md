# Releasing MB-ADR

The current version is `0.1.0b1`, a beta candidate. The work is prepared locally;
no remote tag, GitHub release or PyPI publication is implied by the version file.

## Validate the candidate

1. Export a clean Git tree, including the assets, recordings and checkpoints.
2. Create a new Python 3.10 virtual environment with user-site/ROS paths disabled.
3. Install the CPU wheel and `.[all]` using a constraints file as in CONTRIBUTING.
4. Run the CI commands, including protocol/loopback tests, kinematics, loss and a
   one-step training/checkpoint round trip.
5. Run `python scripts/check_repository.py` and `python -m build`.
6. Install the resulting wheel and check its CLIs outside the source directory.
   For inference from a wheel, set `MBADR_PROJECT_ROOT` to the full checkout.
7. Record actual results in `docs/VALIDATION.md`. Keep physical L20 support marked
   experimental until a real run is documented in `docs/HARDWARE_VALIDATION.md`.

The constraints specify tested direct dependency versions. They are not a
complete transitive lockfile. A successful local check is not a claim that GitHub
Actions has run before the branch is pushed.

## Prepare the release record

Update `_version.py`, the changelog, READMEs and artifact hashes when the shipped
models/data change. Preserve third-party notices; unresolved provenance remains
listed in `THIRD_PARTY_NOTICES.md` rather than being declared verified. The code
wheel omits robot assets and data. Any full-source distribution includes their
separate, potentially noncommercial or unconfirmed terms.

After local checks, a local candidate tag can be prepared for review. After
passing CI on the remote commit, a maintainer can push the reviewed tag and create
the GitHub prerelease. This repository has no
automatic publish workflow and stores no publishing credentials.
