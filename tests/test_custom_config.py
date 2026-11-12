"""Custom calibration must reach live inference and both evaluation objectives."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from mbadr.config import PROJECT_ROOT, default_config_path
from mbadr.data import preprocess_keypoints
from mbadr.evaluate import Evaluator
from mbadr.live import KeypointRetargeter


class CustomConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = PROJECT_ROOT / "out/net_l20.pt"
        cls.data_path = PROJECT_ROOT / "data/qsq_pico4_merged_v1.npy"
        cls.raw = np.load(cls.data_path, allow_pickle=False)[0]
        cls.device = torch.device("cpu")

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.config_path = self.folder / "custom_sensor.yaml"
        self.cfg = yaml.safe_load(default_config_path("l20").read_text())
        rc = self.cfg["retarget"]
        rc.update(w_pos=0.35, w_dir=4.5, w_full_hand=1.7, w_contact=0.0,
                  lateral_scaling=0.83)
        rc["segment_scaling"]["thumb"] = [1.1, 0.9, 0.8, 1.2]
        rc["pinch_thresholds"]["index"] = {"d1": 0.2, "d2": 0.6}
        rc["mediapipe_rotation"]["right"] = {"x": 7.0, "y": 3.0, "z": -4.0}
        self.config_path.write_text(yaml.safe_dump(self.cfg))

    def evaluator(self):
        return Evaluator(self.config_path, "right", self.checkpoint, self.device, "l20")

    def test_custom_targets_and_loss_agree_with_reference(self):
        ev = self.evaluator()
        self.assertEqual(ev.ref.cfg.w_pos, 0.35)
        self.assertEqual(ev.ref.cfg.w_contact, 0.0)
        self.assertEqual(ev.ref.cfg.lateral_scaling, 0.83)
        ev.loss_fn.double()
        kp = preprocess_keypoints(self.raw, "right", ev.rotation, normalize_scale=True)
        alphas, targets = ev.targets_for(torch.from_numpy(kp[None]))
        ref_targets = ev.ref.targets(kp)
        np.testing.assert_allclose(alphas[0].numpy(), ref_targets["alphas"], atol=1e-8)
        for name in ("tip_vectors", "tip_dirs", "full_hand"):
            np.testing.assert_allclose(targets[name][0].numpy(), ref_targets[name], atol=1e-8)
        x = ev.chain.independent_limits().mean(axis=1)
        expected, _ = ev.ref.loss_and_grad(x, ref_targets)
        actual = ev.loss_fn(torch.from_numpy(x[None]), targets, alphas).item()
        # Evaluator stores FK buffers in float32 for inference; converting the
        # loss to double does not recover precision already rounded in them.
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-6)

    def test_evaluation_cli_uses_its_evaluators_reference(self):
        from mbadr import evaluate

        ev = self.evaluator()
        output = self.folder / "eval.json"
        argv = ["mbadr-evaluate", "--robot", "l20", "--config", str(self.config_path),
                "--checkpoint", str(self.checkpoint), "--data-npy", str(self.data_path),
                "--limit", "2", "--out", str(output)]
        with patch("sys.argv", argv), patch.object(evaluate, "Evaluator", return_value=ev) as factory:
            with patch.object(ev.ref, "solve", wraps=ev.ref.solve) as solve:
                with contextlib.redirect_stdout(io.StringIO()):
                    evaluate.main()
                rows = json.loads(output.read_text())
                self.assertEqual(solve.call_count, sum(row["frames"] for row in rows.values()))
                self.assertGreater(solve.call_count, 0)
            self.assertEqual(factory.call_args.args[0], self.config_path)

    def test_live_explicit_config_never_loads_the_default(self):
        with patch("mbadr.live.default_config_path", side_effect=AssertionError("default config used")):
            rt = KeypointRetargeter(self.checkpoint, "right", self.device,
                                   config_path=self.config_path)
        self.assertEqual(rt.config_path, self.config_path)
        self.assertEqual(rt.ev.rotation, self.cfg["retarget"]["mediapipe_rotation"]["right"])
        kp = preprocess_keypoints(self.raw, "right", rt.ev.rotation, normalize_scale=True)
        with torch.no_grad():
            expected = rt.ev.fk.expand(rt.ev.net_solve(torch.from_numpy(kp[None]).float()))[0].numpy()
        np.testing.assert_allclose(rt(self.raw), expected, atol=1e-7)

    def test_missing_explicit_config_is_not_replaced_with_default(self):
        with self.assertRaises(FileNotFoundError):
            KeypointRetargeter(self.checkpoint, "right", self.device,
                              config_path=self.folder / "missing.yaml")

    def test_simulation_adapters_use_custom_calibration(self):
        from scripts.demo_mujoco import NetRetargeter
        from scripts.demo_pico4 import LiveRetargeter

        rt = KeypointRetargeter(self.checkpoint, "right", self.device,
                               config_path=self.config_path)
        expected = rt(self.raw)
        offline = NetRetargeter(self.checkpoint, self.device, config_path=self.config_path)
        self.assertEqual(offline.ev.ref.cfg.w_pos, 0.35)
        kp = preprocess_keypoints(self.raw, "right", offline.ev.rotation, normalize_scale=True)
        np.testing.assert_allclose(offline(kp), expected, atol=1e-7)
        live = LiveRetargeter(self.checkpoint, "right", self.device,
                             config_path=self.config_path)
        np.testing.assert_allclose(live(self.raw), live.mapping.to_qpos(expected), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
