import tempfile
import unittest
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from virtual_shelly.forecast import (
    ForecastConfig, ForecastManager, HistoryStore, LAGS, feature_names, make_feature, mape, supervised,
)
from virtual_shelly.train_forecast import incumbent_is_compatible


class ForecastTests(unittest.TestCase):
    def test_default_horizon_is_one_step(self):
        with patch.dict(os.environ, {}, clear=True):
            config = ForecastConfig.from_env()
            self.assertEqual(config.horizon, 1)
            self.assertEqual(config.window_size, 5)

    def test_input_window_comes_from_environment(self):
        with patch.dict(os.environ, {"FORECAST_WINDOW_SIZE": "12"}, clear=True):
            self.assertEqual(ForecastConfig.from_env().window_size, 12)

    def test_serve_interval_comes_from_environment(self):
        with patch.dict(os.environ, {"FORECAST_SERVE_INTERVAL": "0.4"}, clear=True):
            self.assertEqual(ForecastConfig.from_env().serve_interval, 0.4)

    def test_input_vector_contains_configured_number_of_readings(self):
        rows = [(float(i), float(i), float(i * 2), float(i * 3)) for i in range(12)]
        vector = make_feature(rows, 11, window_size=4)
        self.assertEqual(len(vector), 4 + 3 * 4)
        self.assertEqual(vector[4:8], [11.0, 10.0, 9.0, 8.0])

    def test_horizon_change_disables_warm_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for phase in ("a", "b", "c"):
                (output / f"{phase}.txt").touch()
            metadata = {"horizon": 1, "window_size": 5, "features": feature_names(5)}
            (output / "metadata.json").write_text(json.dumps(metadata))
            self.assertTrue(incumbent_is_compatible(output, ForecastConfig(horizon=1)))
            self.assertFalse(incumbent_is_compatible(output, ForecastConfig(horizon=2)))

    def test_input_window_change_disables_warm_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for phase in ("a", "b", "c"):
                (output / f"{phase}.txt").touch()
            metadata = {"horizon": 1, "window_size": 10, "features": feature_names(10)}
            (output / "metadata.json").write_text(json.dumps(metadata))
            self.assertTrue(incumbent_is_compatible(output, ForecastConfig(window_size=10)))
            self.assertFalse(incumbent_is_compatible(output, ForecastConfig(window_size=20)))

    def test_mape_uses_floor_for_zero_values(self):
        self.assertEqual(mape([0.0, 10.0], [10.0, 20.0], floor=10.0), 100.0)

    def test_supervised_target_is_shifted_by_horizon(self):
        rows = [(float(i), float(i), float(i * 2), float(i * 3)) for i in range(100)]
        features, targets = supervised(rows, 4)
        self.assertEqual(len(features), 100 - len(LAGS) - 4 + 1)
        self.assertEqual(targets["a"][0], len(LAGS) - 1 + 4)
        self.assertEqual(targets["c"][0], (len(LAGS) - 1 + 4) * 3)

    def test_history_is_persistent_and_ordered(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HistoryStore(str(Path(tmp) / "history.db"))
            store.append(2, (4, 5, 6))
            store.append(1, (1, 2, 3))
            self.assertEqual([row[0] for row in store.read()], [1.0, 2.0])
            self.assertEqual([row[0] for row in store.read(cutoff=1.5)], [2.0])
            self.assertEqual(HistoryStore(str(Path(tmp) / "history.db")).read(), [])
            store.flush()
            self.assertEqual([row[0] for row in HistoryStore(str(Path(tmp) / "history.db")).read()], [1.0, 2.0])

    def test_history_flushes_buffer_after_sixty_minutes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HistoryStore(str(Path(tmp) / "history.db"))
            store.append(1, (1, 2, 3))
            with patch("virtual_shelly.forecast.time.monotonic", return_value=store.last_flush_mono + 3600):
                store.append(2, (4, 5, 6))
            reopened = HistoryStore(str(Path(tmp) / "history.db"))
            self.assertEqual([row[0] for row in reopened.read()], [1.0, 2.0])

    def test_manager_falls_back_without_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ForecastConfig(history_path=str(Path(tmp) / "history.db"), model_dir=str(Path(tmp) / "model"))
            manager = ForecastManager(config, 2.0)
            self.assertEqual(manager.predict((1, 2, 3)), ((1.0, 2.0, 3.0), False))
            self.assertEqual(manager.status()["serving"], "fallback_actual")

    def test_manager_serves_clients_in_turn_without_waiting_for_new_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ForecastConfig(
                history_path=str(Path(tmp) / "history.db"),
                model_dir=str(Path(tmp) / "model"),
                serve_interval=0.05,
            )
            manager = ForecastManager(config, 2.0)
            for i in range(max(LAGS) + 1):
                manager.store.append(float(i), (i, i * 2, i * 3))
            model = unittest.mock.Mock()
            model.predict.return_value = [42.0]
            manager.models = {phase: model for phase in ("a", "b", "c")}

            with patch.object(manager, "reload"):
                self.assertEqual(manager.predict((1, 2, 3)), ((42.0, 42.0, 42.0), True))
                result = []
                waiter = threading.Thread(target=lambda: result.append(manager.predict((4, 5, 6))))
                waiter.start()
                time.sleep(0.01)
                self.assertTrue(waiter.is_alive())
                waiter.join(timeout=1)

            self.assertFalse(waiter.is_alive())
            self.assertEqual(result, [((42.0, 42.0, 42.0), True)])
            self.assertEqual(model.predict.call_count, 6)

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
