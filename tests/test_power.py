import importlib
import os
import unittest
from unittest.mock import patch

import virtual_shelly.power as power


class PowerOffsetTests(unittest.TestCase):
    def tearDown(self):
        importlib.reload(power)

    def test_default_offsets(self):
        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(power)
            self.assertEqual(power.apply_total_power_offset(100.0), 90.0)
            self.assertEqual(power.apply_total_power_offset(-100.0), -110.0)
            self.assertEqual(power.apply_total_power_offset(0.0), 0.0)

    def test_offsets_can_be_configured_independently(self):
        env = {"POSITIVE_POWER_OFFSET": "2.5", "NEGATIVE_POWER_OFFSET": "4.5"}
        with patch.dict(os.environ, env, clear=True):
            importlib.reload(power)
            self.assertEqual(power.apply_total_power_offset(100.0), 97.5)
            self.assertEqual(power.apply_total_power_offset(-100.0), -104.5)


if __name__ == "__main__":
    unittest.main()
