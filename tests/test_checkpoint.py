"""Portable checkpoint loading without an unsafe pickle fallback."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from mbadr.checkpoint import load_checkpoint
from mbadr.config import OUT_DIR

EXECUTED = False


def execute_pickle():
    global EXECUTED
    EXECUTED = True
    return "executed"


class UnsafeMetadata:
    def __reduce__(self):
        return execute_pickle, ()


class CheckpointTests(unittest.TestCase):
    def checkpoint(self):
        return dict(format_version=2, state_dict={}, in_dim=80, num_joints=2,
                    hidden=16, layers=1, lower=torch.zeros(2), upper=torch.ones(2))

    def test_all_bundled_checkpoints_load_restricted_on_cpu(self):
        paths = sorted(OUT_DIR.glob("*.pt"))
        self.assertGreaterEqual(len(paths), 6)
        for path in paths:
            with self.subTest(path=path.name):
                checkpoint = load_checkpoint(path)
                self.assertEqual(checkpoint["format_version"], 2)
                self.assertEqual(checkpoint["lower"].device.type, "cpu")

    def test_rejects_numpy_legacy_metadata_without_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pt"
            checkpoint = self.checkpoint()
            checkpoint["lower"] = np.zeros(2)
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "migrate_checkpoint"):
                load_checkpoint(path)

    def test_unsafe_metadata_is_never_executed(self):
        global EXECUTED
        EXECUTED = False
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unsafe.pt"
            checkpoint = self.checkpoint()
            checkpoint["extra"] = UnsafeMetadata()
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)
            self.assertFalse(EXECUTED)

    def test_invalid_shapes_and_nonfinite_limits_are_rejected(self):
        for replacement in (torch.zeros(3), torch.full((2,), float("nan"))):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "invalid.pt"
                checkpoint = self.checkpoint()
                checkpoint["lower"] = replacement
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, "limits"):
                    load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
