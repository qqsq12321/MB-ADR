"""Every advertised hand must have the resources required by the public API."""

import unittest

from mbadr.config import HandSpec, SUPPORTED_HANDS, default_config_path, resolve_robot_name


class ConfigurationTests(unittest.TestCase):
    def test_advertised_hands_have_bundled_right_hand_resources(self):
        for name in SUPPORTED_HANDS:
            with self.subTest(robot=name):
                spec = HandSpec.from_name(name, "right")
                self.assertTrue(spec.urdf_path.is_file())
                self.assertTrue(spec.mjcf_path.is_file())
                self.assertTrue(default_config_path(name).is_file())

    def test_missing_historical_hand_is_rejected_at_name_resolution(self):
        with self.assertRaisesRegex(ValueError, "Unknown hand"):
            resolve_robot_name("allegro")


if __name__ == "__main__":
    unittest.main()
