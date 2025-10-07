from __future__ import annotations

import os
import threading
from collections import deque
from typing import Deque, Dict, List

try:
    MAX_POINTS = int(os.getenv("TIMESERIES_MAX_POINTS", "720"))
except Exception:
    MAX_POINTS = 720

_lock = threading.RLock()
_ts: Deque[int] = deque(maxlen=MAX_POINTS)
_a: Deque[float] = deque(maxlen=MAX_POINTS)
_b: Deque[float] = deque(maxlen=MAX_POINTS)
_c: Deque[float] = deque(maxlen=MAX_POINTS)
_total: Deque[float] = deque(maxlen=MAX_POINTS)


def add_sample(ts: int, a: float, b: float, c: float, total: float) -> None:
    with _lock:
        _ts.append(int(ts))
        _a.append(float(a))
        _b.append(float(b))
        _c.append(float(c))
        _total.append(float(total))


def get_history(limit: int | None = None) -> List[Dict[str, float]]:
    with _lock:
        n = len(_ts)
        if limit is None or limit > n:
            limit = n
        out: List[Dict[str, float]] = []
        start = n - limit
        # Materialize as list for clients
        ts = list(_ts)[start:]
        a = list(_a)[start:]
        b = list(_b)[start:]
        c = list(_c)[start:]
        total = list(_total)[start:]
        for i in range(len(ts)):
            out.append({
                "ts": int(ts[i]),
                "a": float(a[i]),
                "b": float(b[i]),
                "c": float(c[i]),
                "total": float(total[i]),
            })
        return out


def set_history(history: List[Dict[str, float]]) -> None:
    """Replace the current buffers with the provided history (bounded)."""
    if not isinstance(history, list):
        return
    with _lock:
        _ts.clear(); _a.clear(); _b.clear(); _c.clear(); _total.clear()
        # Keep only last MAX_POINTS
        start = max(0, len(history) - MAX_POINTS)
        for item in history[start:]:
            try:
                _ts.append(int(item.get("ts")))
                _a.append(float(item.get("a", 0.0)))
                _b.append(float(item.get("b", 0.0)))
                _c.append(float(item.get("c", 0.0)))
                _total.append(float(item.get("total", 0.0)))
            except Exception:
                continue
