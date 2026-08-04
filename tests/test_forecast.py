import tempfile
import unittest
import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from virtual_shelly.forecast import ForecastConfig, ForecastManager, HistoryStore, LAGS, feature_names, mape, supervised
from virtual_shelly.train_forecast import incumbent_is_compatible


class ForecastTests(unittest.TestCase):
    def test_default_horizon_is_one_step(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ForecastConfig.from_env().horizon, 1)

    def test_horizon_change_disables_warm_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for phase in ("a", "b", "c"):
                (output / f"{phase}.txt").touch()
            metadata = {"horizon": 1, "features": feature_names()}
            (output / "metadata.json").write_text(json.dumps(metadata))
            self.assertTrue(incumbent_is_compatible(output, ForecastConfig(horizon=1)))
            self.assertFalse(incumbent_is_compatible(output, ForecastConfig(horizon=2)))

    def test_mape_uses_floor_for_zero_values(self):
        self.assertEqual(mape([0.0, 10.0], [10.0, 20.0], floor=10.0), 100.0)

    def test_supervised_target_is_shifted_by_horizon(self):
        rows = [(float(i), float(i), float(i * 2), float(i * 3)) for i in range(100)]
        features, targets = supervised(rows, 4)
        self.assertEqual(len(features), 100 - max(LAGS) - 4)
        self.assertEqual(targets["a"][0], max(LAGS) + 4)
        self.assertEqual(targets["c"][0], (max(LAGS) + 4) * 3)

    def test_history_is_persistent_and_ordered(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HistoryStore(str(Path(tmp) / "history.db"))
            store.append(2, (4, 5, 6))
            store.append(1, (1, 2, 3))
            self.assertEqual([row[0] for row in store.read()], [1.0, 2.0])

    def test_manager_falls_back_without_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ForecastConfig(history_path=str(Path(tmp) / "history.db"), model_dir=str(Path(tmp) / "model"))
            manager = ForecastManager(config, 2.0)
            self.assertEqual(manager.predict((1, 2, 3)), ((1.0, 2.0, 3.0), False))
            self.assertEqual(manager.status()["serving"], "fallback_actual")

    def test_daily_scheduler_only_launches_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ForecastConfig(history_path=str(Path(tmp) / "history.db"), model_dir=str(Path(tmp) / "model"), train_hour=2)
            manager = ForecastManager(config, 2.0)
            process = unittest.mock.Mock()
            process.poll.return_value = None
            with patch("virtual_shelly.forecast.subprocess.Popen", return_value=process) as popen:
                manager.maybe_launch_training(datetime(2026, 1, 1, 2, 0))
                manager.maybe_launch_training(datetime(2026, 1, 1, 2, 30))
                popen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
