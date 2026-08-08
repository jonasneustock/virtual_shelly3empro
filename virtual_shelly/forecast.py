"""Persistent power history, LightGBM inference, and training orchestration."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PHASES = ("a", "b", "c")
DEFAULT_WINDOW_SIZE = 5
# Kept as a compatibility alias for callers that used the old maximum lag.
LAGS = tuple(range(1, DEFAULT_WINDOW_SIZE + 1))


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class ForecastConfig:
    enabled: bool = True
    horizon: int = 1
    history_path: str = "/data/power_history.sqlite3"
    model_dir: str = "/data/forecast_model"
    train_hour: int = 2
    min_samples: int = 1000
    window_size: int = DEFAULT_WINDOW_SIZE
    validation_fraction: float = 0.2
    retention_days: int = 30
    mape_floor_watts: float = 10.0

    @classmethod
    def from_env(cls) -> "ForecastConfig":
        return cls(
            enabled=_env_bool("FORECAST_ENABLE", "true"),
            horizon=max(1, int(os.getenv("FORECAST_HORIZON_STEPS", "1"))),
            history_path=os.getenv("FORECAST_HISTORY_PATH", "/data/power_history.sqlite3"),
            model_dir=os.getenv("FORECAST_MODEL_DIR", "/data/forecast_model"),
            train_hour=min(23, max(0, int(os.getenv("FORECAST_TRAIN_HOUR", "2")))),
            min_samples=max(100, int(os.getenv("FORECAST_MIN_SAMPLES", "1000"))),
            window_size=max(1, int(os.getenv("FORECAST_WINDOW_SIZE", str(DEFAULT_WINDOW_SIZE)))),
            validation_fraction=min(.4, max(.05, float(os.getenv("FORECAST_VALIDATION_FRACTION", ".2")))),
            retention_days=max(1, int(os.getenv("FORECAST_HISTORY_DAYS", "30"))),
            mape_floor_watts=max(.001, float(os.getenv("FORECAST_MAPE_FLOOR_WATTS", "10"))),
        )


def mape(actual: Sequence[float], predicted: Sequence[float], floor: float = 10.0) -> float:
    if not actual or len(actual) != len(predicted):
        raise ValueError("MAPE requires equally sized non-empty sequences")
    return 100.0 * sum(abs(a - p) / max(abs(a), floor) for a, p in zip(actual, predicted)) / len(actual)


class HistoryStore:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS power (ts REAL PRIMARY KEY, a REAL, b REAL, c REAL)")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def append(self, ts: float, powers: Sequence[float]) -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO power VALUES (?, ?, ?, ?)", (ts, *map(float, powers)))

    def read(self, cutoff: Optional[float] = None) -> List[Tuple[float, float, float, float]]:
        with self._connect() as db:
            if cutoff is None:
                return list(db.execute("SELECT ts,a,b,c FROM power ORDER BY ts"))
            return list(db.execute("SELECT ts,a,b,c FROM power WHERE ts >= ? ORDER BY ts", (cutoff,)))

    def prune(self, cutoff: float) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM power WHERE ts < ?", (cutoff,))


def feature_names(window_size: int = DEFAULT_WINDOW_SIZE) -> List[str]:
    names = ["hour_sin", "hour_cos", "week_sin", "week_cos"]
    for phase in PHASES:
        names.extend(f"{phase}_lag_{lag}" for lag in range(window_size))
    return names


def make_feature(
    rows: Sequence[Sequence[float]], index: int, window_size: int = DEFAULT_WINDOW_SIZE
) -> List[float]:
    ts = float(rows[index][0])
    dt = datetime.fromtimestamp(ts)
    hour_angle = 2 * math.pi * (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400
    week_angle = 2 * math.pi * dt.weekday() / 7
    values = [math.sin(hour_angle), math.cos(hour_angle), math.sin(week_angle), math.cos(week_angle)]
    for column in range(1, 4):
        values.extend(float(rows[index - lag][column]) for lag in range(window_size))
    return values


def supervised(
    rows: Sequence[Sequence[float]], horizon: int, window_size: int = DEFAULT_WINDOW_SIZE
) -> Tuple[List[List[float]], Dict[str, List[float]]]:
    features: List[List[float]] = []
    targets = {phase: [] for phase in PHASES}
    for i in range(window_size - 1, len(rows) - horizon):
        features.append(make_feature(rows, i, window_size))
        for column, phase in enumerate(PHASES, 1):
            targets[phase].append(float(rows[i + horizon][column]))
    return features, targets


class ForecastManager:
    """Thread-safe serving facade; training happens in a child interpreter."""

    def __init__(self, config: ForecastConfig, poll_interval: float):
        self.config = config
        self.poll_interval = poll_interval
        self.store = HistoryStore(config.history_path) if config.enabled else None
        self.lock = threading.RLock()
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.metadata_mtime = 0.0
        self.process: Optional[subprocess.Popen] = None
        self.last_launch_date: Optional[str] = None
        # A source sample may only satisfy one caller.  Serialising callers here
        # prevents several batteries from reacting to the same rise or fall.
        self.source_condition = threading.Condition()
        self.source_generation = 0
        self.served_generation = -1
        self.next_client_ticket = 0
        self.serving_client_ticket = 0
        self.reload()

    @property
    def metadata_path(self) -> Path:
        return Path(self.config.model_dir) / "metadata.json"

    def record(self, powers: Sequence[float], ts: Optional[float] = None) -> None:
        if not self.store:
            return
        self.store.append(ts or time.time(), powers)
        with self.source_condition:
            self.source_generation += 1
            self.source_condition.notify_all()
        if int(ts or time.time()) % 3600 < max(2, int(self.poll_interval)):
            self.store.prune(time.time() - self.config.retention_days * 86400)

    def reload(self) -> None:
        path = self.metadata_path
        if not self.config.enabled or not path.exists() or path.stat().st_mtime <= self.metadata_mtime:
            return
        try:
            import lightgbm as lgb
            metadata = json.loads(path.read_text())
            if (
                metadata.get("horizon") != self.config.horizon
                or metadata.get("window_size") != self.config.window_size
                or metadata.get("features") != feature_names(self.config.window_size)
                or metadata.get("validation_mape", float("inf")) >= 10.0
            ):
                return
            models = {p: lgb.Booster(model_file=str(Path(self.config.model_dir) / f"{p}.txt")) for p in PHASES}
            with self.lock:
                self.models, self.metadata = models, metadata
                self.metadata_mtime = path.stat().st_mtime
        except Exception:
            return

    def predict(self, actual: Sequence[float]) -> Tuple[Tuple[float, float, float], bool]:
        if not self.store:
            return tuple(map(float, actual)), False

        # Keep the condition held through inference and generation bookkeeping:
        # one newly recorded source value is consequently handed to one waiting
        # client, and the following client waits for the next poll.
        with self.source_condition:
            ticket = self.next_client_ticket
            self.next_client_ticket += 1
            while ticket != self.serving_client_ticket or self.source_generation <= self.served_generation:
                self.source_condition.wait()
            generation = self.source_generation
            self.reload()
            try:
                if not self.models:
                    return tuple(map(float, actual)), False
                import numpy as np
                with self.lock:
                    rows = self.store.read()
                    if len(rows) < self.config.window_size:
                        return tuple(map(float, actual)), False
                    row = make_feature(rows, len(rows) - 1, self.config.window_size)
                    matrix = np.asarray([row], dtype=float)
                    result = tuple(float(self.models[p].predict(matrix)[0]) for p in PHASES)
                return result, True
            except Exception:
                return tuple(map(float, actual)), False
            finally:
                self.served_generation = generation
                self.serving_client_ticket += 1
                self.source_condition.notify_all()

    def sample_counts(self) -> Tuple[int, int]:
        if not self.store:
            return 0, 0
        try:
            cutoff = time.time() - self.config.retention_days * 86400
            total = len(supervised(self.store.read(cutoff), self.config.horizon, self.config.window_size)[0])
        except Exception:
            return 0, 0
        if total < 2:
            return total, 0
        train = max(1, min(total - 1, int(total * (1 - self.config.validation_fraction))))
        return train, total - train

    def _reap_process(self) -> None:
        if self.process is not None and self.process.poll() is not None:
            self.process = None

    def launch_training(self) -> bool:
        if not self.config.enabled:
            return False
        self._reap_process()
        if self.process is not None:
            return False
        self.process = subprocess.Popen([sys.executable, "-m", "virtual_shelly.train_forecast"])
        return True

    def status(self) -> Dict[str, Any]:
        self.reload()
        self._reap_process()
        running = self.process is not None and self.process.poll() is None
        training_samples, current_validation_samples = self.sample_counts()
        return {
            "enabled": self.config.enabled,
            "available": bool(self.models),
            "serving": "forecast" if self.models else "fallback_actual",
            "horizon_steps": self.config.horizon,
            "horizon_seconds": self.config.horizon * self.poll_interval,
            "window_size": self.config.window_size,
            "validation_mape": self.metadata.get("validation_mape"),
            "phase_mape": self.metadata.get("phase_mape", {}),
            "training_samples": training_samples,
            "validation_samples": self.metadata.get("validation_samples"),
            "current_validation_samples": current_validation_samples,
            "min_samples": self.config.min_samples,
            "trained_at": self.metadata.get("trained_at"),
            "training": running,
            "training_status": "running" if running else ("ready" if training_samples + current_validation_samples >= self.config.min_samples else "collecting_data"),
        }

    def maybe_launch_training(self, now: Optional[datetime] = None) -> None:
        if not self.config.enabled:
            return
        now = now or datetime.now()
        today = now.date().isoformat()
        self._reap_process()
        if now.hour == self.config.train_hour and self.last_launch_date != today and self.launch_training():
            self.last_launch_date = today
