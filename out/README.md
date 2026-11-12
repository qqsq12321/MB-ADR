# Checkpoints and evaluation artifacts

The maintainer confirms that the shipped weights are self-trained and available
under the root [MIT license](../LICENSE). Third-party robot assets have separate
terms in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

The supported per-hand bundles are `net_l20.pt`, `net_shadow.pt`, `net_wuji.pt`,
`net_sharpa.pt`, `net_gaia.pt` and `net_inspire.pt`. All are right-hand trained;
their saved step is 12000. `net_all.pt` is an earlier experimental artifact, not
a universal checkpoint for all six robot morphologies.

As of beta format 2, metadata contains tensors and primitive values only.
Inference uses restricted `weights_only=True` loading on CPU, then transfers the
network to the selected device. The seven existing checkpoints were migrated
without changing any network parameter or joint limit. New training saves the
same format and includes joint-name metadata.

Older trusted files containing NumPy metadata can be converted explicitly:

```bash
python scripts/migrate_checkpoint.py old.pt --out converted.pt
```

Run that source-checkout command from the repository root, only for a file whose
origin is trusted: the migration tool intentionally enables legacy pickle
loading. Normal inference never falls back to that loader automatically.

`eval_<robot>.json` and `train_*.log` are historical results, not proof that all
future environments or physical hands have the same accuracy or timing. The
artifact manifest records exact checkpoint/data hashes; regenerated evaluations
are deliberately not hash-locked. Use another output path for new training to
preserve the shipped model.
