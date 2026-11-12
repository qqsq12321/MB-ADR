"""Checkpoint integration and a complete Pico relay -> dry-run subprocess."""

import json
import os
import pickle
import socket
import struct
import subprocess
import sys
import threading
import unittest
import warnings

import numpy as np
import torch

from mbadr.config import PROJECT_ROOT
from mbadr.data import preprocess_keypoints
from mbadr.live import KeypointRetargeter
from mbadr.model import build_features
from mbadr.pico4 import JOINT_21_INDICES
from scripts.teleop_l20 import validate_checkpoint


class LiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = PROJECT_ROOT / "out/net_l20.pt"
        with (PROJECT_ROOT / "data/recordings/pico4_check.pkl").open("rb") as fh:
            frames = pickle.load(fh)
        cls.raw = [np.asarray(f["right_fingers"]) for f in frames if np.any(f["right_fingers"])][:4]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            cls.rt = KeypointRetargeter(cls.checkpoint, "right", torch.device("cpu"))

    def test_inference_matches_original_path(self):
        ev = self.rt.ev
        for raw in self.raw:
            kp = preprocess_keypoints(raw, "right", ev.rotation, normalize_scale=True)
            with torch.no_grad():
                alphas, targets = ev.targets_for(torch.from_numpy(kp[None]).float())
                x = ev.net(build_features(targets, alphas))
                expected = ev.loss_fn.fk.expand(x)[0].numpy()
            np.testing.assert_allclose(self.rt(raw), expected, atol=1e-7)

    def test_invalid_palm_is_not_inferred(self):
        for raw in (np.zeros((21, 3)), np.full((21, 3), np.nan), np.zeros((26, 3))):
            self.assertIsNone(self.rt(raw))

    def test_smoothing_preserves_limits_and_mimic_coupling(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            rt = KeypointRetargeter(self.checkpoint, "right", torch.device("cpu"), smooth=0.3)
        indices = [rt.chain.joint_names.index(n) for n in rt.chain.independent_names]
        limits = rt.chain.joint_limits()
        for raw in self.raw:
            q = rt(raw)
            self.assertTrue(np.all(q >= limits[:, 0] - 1e-6))
            self.assertTrue(np.all(q <= limits[:, 1] + 1e-6))
            np.testing.assert_allclose(q, rt.chain.expand_to_full(q[indices]), atol=2e-7)

    def test_checkpoint_hand_is_checked(self):
        validate_checkpoint(self.rt, "right")
        with self.assertRaisesRegex(ValueError, "trained for right"):
            validate_checkpoint(self.rt, "left")

    def test_simulation_joint_mapping_is_preserved(self):
        from scripts.demo_pico4 import LiveRetargeter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            sim = LiveRetargeter(self.checkpoint, "right", torch.device("cpu"))
        expected = sim.mapping.to_qpos(self.rt(self.raw[0]))
        np.testing.assert_allclose(sim(self.raw[0]), expected, atol=1e-7)

    def test_pico_to_checkpoint_to_registers_without_serial_or_gl(self):
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(15)
        port = server.getsockname()[1]
        stop = threading.Event()
        errors = []

        def feed():
            try:
                with server.accept()[0] as conn:
                    for raw in self.raw:
                        locations = [{"p": "0,0,0"} for _ in range(26)]
                        for i, sdk_index in enumerate(JOINT_21_INDICES):
                            locations[sdk_index] = {"p": ",".join(str(float(v)) for v in raw[i])}
                        payload = json.dumps({"Hand": {"rightHand": {
                            "isActive": 1, "HandJointLocations": locations,
                        }}}).encode()
                        device = b"PICO-test-device-id"
                        frame = struct.pack("<I", len(device)) + device + struct.pack("<I", len(payload)) + payload
                        # Split in the middle of the relay device id.
                        conn.sendall(frame[:9])
                        stop.wait(0.005)
                        conn.sendall(frame[9:])
                        stop.wait(0.03)
                    # Keep the socket open without sending: exercise stale input.
                    stop.wait(5)
            except OSError as exc:
                if not stop.is_set():
                    errors.append(exc)

        thread = threading.Thread(target=feed, daemon=True)
        thread.start()
        try:
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            result = subprocess.run([
                sys.executable, "-u", str(PROJECT_ROOT / "scripts/teleop_l20.py"),
                "--checkpoint", str(self.checkpoint), "--device", "cpu", "--dry-run",
                "--relay-port", str(port), "--l20-port", "/nonexistent-l20-test-port",
                "--duration", "1.2", "--input-timeout", "0.1", "--stats-every", "0.3",
            ], cwd="/tmp", env=env, capture_output=True, text=True, timeout=20)
        finally:
            stop.set()
            server.close()
            thread.join(timeout=2)
        self.assertEqual(errors, [])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("DRY RUN: serial output disabled", result.stdout)
        self.assertIn("L20 registers:", result.stdout)
        self.assertIn("target updates paused", result.stdout[result.stdout.rfind("L20 registers:"):])
        self.assertNotIn("GLFW", result.stderr)
        self.assertIn("stopped", result.stdout)


if __name__ == "__main__":
    unittest.main()
