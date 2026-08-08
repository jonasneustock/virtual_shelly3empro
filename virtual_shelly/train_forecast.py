"""Daily LightGBM training child process."""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from virtual_shelly.forecast import PHASES, ForecastConfig, HistoryStore, feature_names, mape, supervised


def incumbent_is_compatible(output: Path, config: ForecastConfig) -> bool:
    """Only warm-start models trained for the exact same forecast problem."""
    metadata_path = output / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, ValueError, TypeError):
        return False
    return (
        metadata.get("horizon") == config.horizon
        and metadata.get("window_size") == config.window_size
        and metadata.get("features") == feature_names(config.window_size)
        and all((output / f"{phase}.txt").is_file() for phase in PHASES)
    )


def train() -> bool:
    import lightgbm as lgb
    import numpy as np

    config = ForecastConfig.from_env()
    cutoff = time.time() - config.retention_days * 86400
    rows = HistoryStore(config.history_path).read(cutoff)
    x, targets = supervised(rows, config.horizon, config.window_size)
    if len(x) < config.min_samples:
        return False
    split = max(1, min(len(x) - 1, int(len(x) * (1 - config.validation_fraction))))
    x_train = np.asarray(x[:split], dtype=float)
    x_valid = np.asarray(x[split:], dtype=float)
    output = Path(config.model_dir)
    output.mkdir(parents=True, exist_ok=True)
    warm_start = incumbent_is_compatible(output, config)
    candidates = {}
    scores = {}
    incumbent_scores = {}
    params = {"objective": "regression_l1", "metric": "l1", "learning_rate": .05, "num_leaves": 31, "verbosity": -1, "seed": 42}
    for phase in PHASES:
        old_path = output / f"{phase}.txt"
        # A horizon or input-window change changes the training problem. Do a
        # full cold retrain rather than extending an incompatible model.
        old = lgb.Booster(model_file=str(old_path)) if warm_start else None
        train_set = lgb.Dataset(x_train, label=targets[phase][:split], feature_name=feature_names(config.window_size))
        valid_set = lgb.Dataset(x_valid, label=targets[phase][split:], reference=train_set)
        candidates[phase] = lgb.train(params, train_set, num_boost_round=300, valid_sets=[valid_set],
                                      callbacks=[lgb.early_stopping(30, verbose=False)], init_model=old)
        scores[phase] = mape(targets[phase][split:], candidates[phase].predict(x_valid), config.mape_floor_watts)
        if old:
            incumbent_scores[phase] = mape(targets[phase][split:], old.predict(x_valid), config.mape_floor_watts)
    candidate_score = sum(scores.values()) / len(scores)
    incumbent_score = sum(incumbent_scores.values()) / len(incumbent_scores) if len(incumbent_scores) == 3 else None
    # Never serve a forecast that misses the requested quality bar. The
    # manager continues returning current readings until a later run succeeds.
    if candidate_score >= 10.0:
        return False
    if incumbent_score is not None and candidate_score >= incumbent_score:
        return False
    with tempfile.TemporaryDirectory(dir=str(output.parent)) as tmp:
        tmp_path = Path(tmp)
        for phase, model in candidates.items():
            model.save_model(str(tmp_path / f"{phase}.txt"))
        metadata = {
            "horizon": config.horizon, "window_size": config.window_size,
            "features": feature_names(config.window_size), "validation_mape": candidate_score,
            "phase_mape": scores, "validation_samples": len(x_valid),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        for phase in PHASES:
            os.replace(tmp_path / f"{phase}.txt", output / f"{phase}.txt")
        os.replace(tmp_path / "metadata.json", output / "metadata.json")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if train() else 2)
