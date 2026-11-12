"""Input fragmentation and per-hand freshness used by real L20 control."""

import json
import struct
import threading
import unittest
from unittest.mock import patch

import numpy as np

from mbadr.pico4 import Pico4, _RelayFrameParser, _parse_hand_joints, _parse_tracking_json


class PicoTests(unittest.TestCase):
    def test_fragmented_relay_device_id_and_payload(self):
        device = b"PICO-long-device-id"
        payload = json.dumps({"Hand": {}}).encode()
        frame = struct.pack("<I", len(device)) + device + struct.pack("<I", len(payload)) + payload
        parser = _RelayFrameParser()
        for byte in frame[:-1]:
            parser.feed(bytes([byte]))
            self.assertIsNone(parser.try_parse())
        parser.feed(frame[-1:])
        self.assertEqual(parser.try_parse(), payload)
        self.assertIsNone(parser.try_parse())

    def test_bad_joint_positions_are_inactive(self):
        for position in ("bad,0,0", "nan,0,0", "inf,0,0", "0,0"):
            joints = {"isActive": 1, "HandJointLocations": [{"p": position} for _ in range(26)]}
            with self.subTest(position=position):
                self.assertIsNone(_parse_hand_joints(joints))
        joints = {"isActive": 1, "HandJointLocations": [{} for _ in range(26)]}
        self.assertIsNone(_parse_hand_joints(joints))

    def test_valid_tracking_is_wrist_centered(self):
        joints = {"isActive": 1, "HandJointLocations": [{"p": f"{i},1,2"} for i in range(26)]}
        _, right = _parse_tracking_json(json.dumps({"Hand": {"rightHand": joints}}).encode())
        np.testing.assert_array_equal(right[0], [0, 0, 0])
        np.testing.assert_array_equal(right[-1], [24, 0, 0])

    def test_freshness_is_per_hand_and_disconnect_invalidates(self):
        dev = Pico4.__new__(Pico4)
        dev._lock = threading.Lock()
        dev._left_hand = dev._right_hand = None
        dev._hand_updates = {"left": 0.0, "right": 0.0}
        dev._last_update = 0.0
        raw = np.ones((21, 3), dtype=np.float32)
        with patch("mbadr.pico4.time.monotonic", return_value=10.0):
            dev._update_hands(raw, raw)
        sample, timestamp = dev.get_hand_sample("right")
        self.assertEqual(timestamp, 10.0)
        sample[:] = 9
        np.testing.assert_array_equal(dev.get_hand_sample("right")[0], raw)
        with patch("mbadr.pico4.time.monotonic", return_value=20.0):
            dev._update_hands(raw, None)
        self.assertEqual(dev.get_hand_sample("left")[1], 20.0)
        self.assertEqual(dev.get_hand_sample("right"), (None, 0.0))
        # Older consumers still get the cached pose through the legacy API.
        np.testing.assert_array_equal(dev.get_fingers_data()["right_fingers"], raw)
        dev._invalidate_tracking()
        self.assertEqual(dev.get_hand_sample("left"), (None, 0.0))


if __name__ == "__main__":
    unittest.main()
