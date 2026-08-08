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
DISK_FLUSH_INTERVAL_SECONDS = 60 * 60
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
    serve_interval: float = 0.25

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
            serve_interval=max(0.0, float(os.getenv("FORECAST_SERVE_INTERVAL", ".25"))),
        )


def mape(actual: Sequence[float], predicted: Sequence[float], floor: float = 10.0) -> float:
    if not actual or len(actual) != len(predicted):
        raise ValueError("MAPE requires equally sized non-empty sequences")
    return 100.0 * sum(abs(a - p) / max(abs(a), floor) for a, p in zip(actual, predicted)) / len(actual)


class HistoryStore:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.RLock()
        self.pending: Dict[float, Tuple[float, float, float]] = {}
        self.last_flush_mono = time.monotonic()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS power (ts REAL PRIMARY KEY, a REAL, b REAL, c REAL)")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def append(
        self, ts: float, powers: Sequence[float], prune_before: Optional[float] = None
    ) -> None:
        values = tuple(map(float, powers))
        if len(values) != 3:
            raise ValueError("power history requires exactly three phase values")
        with self.lock:
            self.pending[float(ts)] = values
            if time.monotonic() - self.last_flush_mono >= DISK_FLUSH_INTERVAL_SECONDS:
                self.pending = {
                    pending_ts: pending_powers
                    for pending_ts, pending_powers in self.pending.items()
                    if prune_before is None or pending_ts >= prune_before
                }
                self.flush(prune_before=prune_before)

    def flush(self, prune_before: Optional[float] = None) -> None:
        """Commit buffered observations in one transaction at the hourly boundary."""
        with self.lock:
            if not self.pending and prune_before is None:
                self.last_flush_mono = time.monotonic()
                return
            rows = [(ts, *powers) for ts, powers in self.pending.items()]
            with self._connect() as db:
                if rows:
                    db.executemany("INSERT OR REPLACE INTO power VALUES (?, ?, ?, ?)", rows)
                if prune_before is not None:
                    db.execute("DELETE FROM power WHERE ts < ?", (prune_before,))
            self.pending.clear()
            self.last_flush_mono = time.monotonic()

    def read(self, cutoff: Optional[float] = None) -> List[Tuple[float, float, float, float]]:
        with self.lock:
            with self._connect() as db:
                if cutoff is None:
                    rows = list(db.execute("SELECT ts,a,b,c FROM power ORDER BY ts"))
                else:
                    rows = list(db.execute("SELECT ts,a,b,c FROM power WHERE ts >= ? ORDER BY ts", (cutoff,)))
            merged = {row[0]: row for row in rows}
            merged.update(
                (ts, (ts, *powers))
                for ts, powers in self.pending.items()
                if cutoff is None or ts >= cutoff
            )
            return [merged[ts] for ts in sorted(merged)]

    def prune(self, cutoff: float) -> None:
        with self.lock:
            self.pending = {ts: powers for ts, powers in self.pending.items() if ts >= cutoff}
            self.flush(prune_before=cutoff)


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
        # Serialize consumers without making them wait for another source poll.
        # A short gap keeps multiple batteries from reacting at exactly the same
        # instant while ensuring every request receives the latest known value.
        self.serve_condition = threading.Condition()
        self.next_client_ticket = 0
        self.serving_client_ticket = 0
        self.last_served_mono: Optional[float] = None
        self.reload()

    @property
    def metadata_path(self) -> Path:
        return Path(self.config.model_dir) / "metadata.json"

    def record(self, powers: Sequence[float], ts: Optional[float] = None) -> None:
        if not self.store:
            return
        now = time.time()
        self.store.append(
            ts or now,
            powers,
            prune_before=now - self.config.retention_days * 86400,
        )

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

        # Keep the condition held through inference and bookkeeping so callers
        # are served in arrival order, separated by the configured short gap.
        with self.serve_condition:
            ticket = self.next_client_ticket
            self.next_client_ticket += 1
            while ticket != self.serving_client_ticket:
                self.serve_condition.wait()
            if self.last_served_mono is not None:
                while (remaining := self.config.serve_interval - (time.monotonic() - self.last_served_mono)) > 0:
                    self.serve_condition.wait(timeout=remaining)
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
                self.last_served_mono = time.monotonic()
                self.serving_client_ticket += 1
                self.serve_condition.notify_all()

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
