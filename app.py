import os
import time
import json
import threading
import socket
import asyncio
import logging
import struct
import signal
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, Deque
from collections import defaultdict, deque

import requests
from sklearn.ensemble import RandomForestRegressor
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse, HTMLResponse
from virtual_shelly import metrics as MET
from virtual_shelly import ui as UI
from virtual_shelly import rpc_core as RPC
from virtual_shelly import udp_server as UDP

# mDNS
from zeroconf import Zeroconf, ServiceInfo, InterfaceChoice
# WebSockets fan-out (6010–6022)
import websockets

try:
    from pymodbus.datastore import (
        ModbusSlaveContext,
        ModbusServerContext,
        ModbusSparseDataBlock,
        ModbusSequentialDataBlock,
    )
    from pymodbus.device import ModbusDeviceIdentification
    from pymodbus.server.async_io import StartAsyncTcpServer
except Exception:  # pragma: no cover - pymodbus optional at import time
    StartAsyncTcpServer = None
# Models
from pydantic import BaseModel

# -----------------------------
# Config (env)
# -----------------------------
HA_BASE_URL = os.getenv("HA_BASE_URL", "http://homeassistant:8123")
HA_TOKEN = os.getenv("HA_TOKEN", "")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "2.0"))
HA_SMOOTHING_ENABLE = os.getenv("HA_SMOOTHING_ENABLE", "false").lower() in ("1", "true", "yes")
HA_SMOOTHING_WINDOW = 5

A_POWER = os.getenv("A_POWER", "sensor.phase_a_power")
B_POWER = os.getenv("B_POWER", "sensor.phase_b_power")
C_POWER = os.getenv("C_POWER", "sensor.phase_c_power")

A_VOLT = os.getenv("A_VOLT", "sensor.phase_a_voltage")
B_VOLT = os.getenv("B_VOLT", "sensor.phase_b_voltage")
C_VOLT = os.getenv("C_VOLT", "sensor.phase_c_voltage")

A_CURR = os.getenv("A_CURR", "sensor.phase_a_current")
B_CURR = os.getenv("B_CURR", "sensor.phase_b_current")
C_CURR = os.getenv("C_CURR", "sensor.phase_c_current")

A_PF = os.getenv("A_PF", "sensor.phase_a_pf")
B_PF = os.getenv("B_PF", "sensor.phase_b_pf")
C_PF = os.getenv("C_PF", "sensor.phase_c_pf")

DEVICE_ID = os.getenv("DEVICE_ID", "shellypro3em-virtual-001")
APP_ID = os.getenv("APP_ID", os.getenv("APP", "shellypro3em"))
MODEL = os.getenv("MODEL", "SHPRO-3EM")
FIRMWARE = os.getenv("FIRMWARE", "1.0.0-virt")
FW_ID = os.getenv("FW_ID", FIRMWARE)
MAC = os.getenv("MAC", "AA:BB:CC:DD:EE:FF")
SN = os.getenv("SN", "VIRT3EM001")
MANUFACTURER = os.getenv("MANUFACTURER", "Allterco Robotics")
GENERATION = int(os.getenv("GENERATION", "2"))

STATE_PATH = os.getenv("STATE_PATH", "/data/state.json")

# Networking
HTTP_PORT = int(os.getenv("HTTP_PORT", "8080"))

# WebSockets (inbound) 6010–6022
WS_PORT_START = int(os.getenv("WS_PORT_START", "6010"))
WS_PORT_END = int(os.getenv("WS_PORT_END", "6022"))

# UDP RPC (inbound): comma-separated list; default both 1010 and 2220
UDP_PORTS = os.getenv("UDP_PORTS", "1010,2220")
UDP_MAX = int(os.getenv("UDP_MAX", "32768"))

# mDNS
MDNS_ENABLE = os.getenv("MDNS_ENABLE", "true").lower() in ("1", "true", "yes")
MDNS_HOSTNAME = os.getenv("MDNS_HOSTNAME", DEVICE_ID)
MDNS_IP = os.getenv("MDNS_IP")  # optional override

# Modbus TCP
MODBUS_ENABLE = os.getenv("MODBUS_ENABLE", "true").lower() in ("1", "true", "yes")
MODBUS_PORT = int(os.getenv("MODBUS_PORT", "502"))
MODBUS_BIND = os.getenv("MODBUS_BIND", "0.0.0.0")
MODBUS_UNIT_ID = int(os.getenv("MODBUS_UNIT_ID", "1"))

# Payload shape tweak (Marstek/Hoymiles consumers)
STRICT_MINIMAL_PAYLOAD = os.getenv("STRICT_MINIMAL_PAYLOAD", "false").lower() in ("1", "true", "yes")

# WS notify throttle (seconds)
WS_NOTIFY_INTERVAL = float(os.getenv("WS_NOTIFY_INTERVAL", "2.0"))
# WS notify coalescing threshold (watts); broadcast only if delta >= EPS
WS_NOTIFY_EPS = float(os.getenv("WS_NOTIFY_EPS", "0.1"))

# CORS (optional)
CORS_ENABLE = os.getenv("CORS_ENABLE", "false").lower() in ("1", "true", "yes")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

# Request-side scaling (divide power by number of active client IPs)
REQUEST_SIDE_SCALING_ENABLE = os.getenv("REQUEST_SIDE_SCALING_ENABLE", "true").lower() in ("1", "true", "yes")
try:
    REQUEST_SIDE_SCALING_CLIENTS = int(os.getenv("REQUEST_SIDE_SCALING_CLIENTS", "0"))  # 0 = auto count
except Exception:
    REQUEST_SIDE_SCALING_CLIENTS = 0

# -----------------------------
# Helpers
# -----------------------------
def ha_get(entity_id: str) -> Optional[float]:
    if not entity_id:
        return None
    url = f"{HA_BASE_URL}/api/states/{entity_id}"
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=3)
        if r.status_code != 200:
            return None
        data = r.json()
        raw = data.get("state")
        if raw in (None, "unknown", "unavailable", ""):
            return None
        value = float(raw)
        if HA_SMOOTHING_ENABLE:
            value = _smooth_value(entity_id, value)
        return value
    except Exception:
        return None


_SMOOTHING_LOCK = threading.RLock()
_SMOOTHING_BUFFERS: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=HA_SMOOTHING_WINDOW))


try:
    REQUEST_IP_TTL = float(os.getenv("REQUEST_IP_TTL", "30.0"))
except ValueError:
    REQUEST_IP_TTL = 30.0

_REQUEST_IP_LOCK = threading.RLock()
_REQUEST_IP_SEEN: Dict[str, float] = {}


def _register_request_ip(addr: Optional[str]) -> None:
    if not addr:
        return
    now = time.monotonic()
    cutoff = now - REQUEST_IP_TTL
    with _REQUEST_IP_LOCK:
        _REQUEST_IP_SEEN[addr] = now
        stale = [ip for ip, ts in _REQUEST_IP_SEEN.items() if ts < cutoff]
        for ip in stale:
            _REQUEST_IP_SEEN.pop(ip, None)


def _active_request_ip_count() -> int:
    now = time.monotonic()
    cutoff = now - REQUEST_IP_TTL
    with _REQUEST_IP_LOCK:
        stale = [ip for ip, ts in _REQUEST_IP_SEEN.items() if ts < cutoff]
        for ip in stale:
            _REQUEST_IP_SEEN.pop(ip, None)
        return len(_REQUEST_IP_SEEN)


def _apply_request_side_power_scaling(powers: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if not REQUEST_SIDE_SCALING_ENABLE:
        return powers
    total = sum(powers)
    if total == 0:
        return powers
    # Allow explicit override of active client count via env; 0 means auto
    override = int(REQUEST_SIDE_SCALING_CLIENTS or 0)
    count = override if override > 0 else _active_request_ip_count()
    count = max(1, count)
    if count <= 1:
        return powers
    return tuple(p / count for p in powers)


def _smooth_value(entity_id: str, value: float) -> float:
    with _SMOOTHING_LOCK:
        buf = _SMOOTHING_BUFFERS[entity_id]
        buf.append(value)
        if not buf:
            return value
        return sum(buf) / len(buf)

def now_ts() -> int:
    return int(time.time())

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(doc: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, STATE_PATH)

# -----------------------------
# Virtual meter state
# -----------------------------
POWER_RETENTION_DAYS = 30
POWER_RETENTION_SECONDS = POWER_RETENTION_DAYS * 24 * 3600
POWER_FORECAST_WINDOW = 30  # number of most recent samples used to fit/forecast
MODEL_TRAIN_INTERVAL = 3600.0  # seconds between retraining runs
POWER_HISTORY_SECONDS = 180  # short window for UI chart

class Phase(BaseModel):
    voltage: Optional[float] = None
    current: Optional[float] = None
    act_power: float = 0.0
    pf: Optional[float] = None

class EnergyCounters(BaseModel):
    a_import: float = 0.0
    b_import: float = 0.0
    c_import: float = 0.0
    a_export: float = 0.0
    b_export: float = 0.0
    c_export: float = 0.0
    total_import: float = 0.0
    total_export: float = 0.0
    since: int = now_ts()

class VirtualPro3EM:
    def __init__(self):
        self.lock = threading.RLock()
        self.last_poll_mono: Optional[float] = None
        self.last_persist_mono: float = time.monotonic()
        self.phases = {"a": Phase(), "b": Phase(), "c": Phase()}
        self.frequency = 50.0
        self.energy = EnergyCounters()
        self.power_history: Deque[Tuple[float, float]] = deque(maxlen=600)
        dataset_maxlen = int(POWER_RETENTION_SECONDS / max(POLL_INTERVAL, 0.5)) + 10
        self.power_dataset: Deque[Tuple[float, float]] = deque(maxlen=dataset_maxlen)
        self.last_model_train: Optional[float] = None
        self.model_params: Dict[str, Any] = {}
        self.model_metrics: Dict[str, Any] = {"mse": None, "mape": None, "n": 0}
        self.model: Optional[RandomForestRegressor] = None
        self.model_ref_ts: Optional[float] = None
        self.config: Dict[str, Any] = {
            "device": {
                "id": DEVICE_ID,
                "model": MODEL,
                "fw": FIRMWARE,
                "mac": MAC,
                "sn": SN,
                "manufacturer": MANUFACTURER,
                "app": APP_ID,
            },
            "network": {"ipv4": "192.0.2.123", "ssid": None, "eth": True},
            "rpc": {"auth": False},
        }
        persisted = load_state()
        if persisted.get("energy"):
            self.energy = EnergyCounters(**persisted["energy"])
        if persisted.get("power_dataset"):
            try:
                for ts, p in persisted.get("power_dataset", []):
                    self.power_dataset.append((float(ts), float(p)))
            except Exception:
                pass
        self._prune_power_dataset(time.time())
        # Seed short history for UI from persisted dataset
        for ts, p in list(self.power_dataset)[-len(self.power_history):]:
            self.power_history.append((ts, p))

    def persist(self):
        now = time.time()
        with self.lock:
            self._prune_power_dataset(now)
            doc = {
                "energy": json.loads(self.energy.json()),
                "power_dataset": list(self.power_dataset),
            }
        save_state(doc)

    def integrate_energy(self, dt_s: float):
        with self.lock:
            a_w = self.phases["a"].act_power or 0.0
            b_w = self.phases["b"].act_power or 0.0
            c_w = self.phases["c"].act_power or 0.0

            def w_to_kwh(w): return (w * dt_s) / 3_600_000.0

            for label, w in (("a", a_w), ("b", b_w), ("c", c_w)):
                kwh = abs(w_to_kwh(w))
                if w >= 0:
                    setattr(self.energy, f"{label}_import", getattr(self.energy, f"{label}_import") + kwh)
                else:
                    setattr(self.energy, f"{label}_export", getattr(self.energy, f"{label}_export") + kwh)

            total_w = a_w + b_w + c_w
            kwh_total = abs(w_to_kwh(total_w))
            if total_w >= 0:
                self.energy.total_import += kwh_total
            else:
                self.energy.total_export += kwh_total

    def record_power_sample(self, total_power: float) -> None:
        # Track recent total power readings for simple forecasting/UI
        now = time.time()
        series: Optional[List[Tuple[float, float]]] = None
        should_train = False
        with self.lock:
            self.power_history.append((now, float(total_power)))
            self.power_dataset.append((now, float(total_power)))
            self._prune_power_dataset(now)
            if (self.last_model_train is None or (now - self.last_model_train) >= MODEL_TRAIN_INTERVAL) and len(self.power_dataset) >= POWER_FORECAST_WINDOW:
                # Copy a small tail for training outside the lock
                series = list(self.power_dataset)[-POWER_FORECAST_WINDOW:]
                should_train = True
                self.last_model_train = now
        if should_train and series:
            try:
                self.train_power_model(series, trained_at=now)
            except Exception:
                pass

    def _prune_power_dataset(self, now_ts: float) -> None:
        cutoff = now_ts - POWER_RETENTION_SECONDS
        while self.power_dataset and self.power_dataset[0][0] < cutoff:
            self.power_dataset.popleft()

    def train_power_model(self, series: List[Tuple[float, float]], trained_at: Optional[float] = None) -> None:
        # Fit a RandomForestRegressor over recent samples using time offset as the feature
        if not series or len(series) < 2:
            return
        ref_ts = series[-1][0]
        offsets = [ts - ref_ts for ts, _ in series]
        ys = [p for _, p in series]
        X = [[x] for x in offsets]

        model = RandomForestRegressor(
            n_estimators=64,
            max_depth=6,
            random_state=42,
        )
        model.fit(X, ys)
        preds = model.predict(X)
        mse = sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(ys)
        eps = 1e-3
        mape = sum(abs((p - y) / (y if abs(y) > eps else eps)) for p, y in zip(preds, ys)) * (100.0 / len(ys))

        ts_trained = trained_at or time.time()
        with self.lock:
            self.model = model
            self.model_ref_ts = ref_ts
            self.model_params = {
                "type": "RandomForestRegressor",
                "trained_at": ts_trained,
                "n_estimators": model.n_estimators,
                "window": len(series),
            }
            self.model_metrics = {"mse": mse, "mape": mape, "n": len(series)}
            self.last_model_train = ts_trained

    def trigger_full_training(self) -> Dict[str, Any]:
        # Train using the full retained dataset (best-effort) and return metrics
        with self.lock:
            series = list(self.power_dataset)
            dataset_total = len(series)
        if len(series) < 2:
            return {"ok": False, "error": "not enough data", "dataset_total": dataset_total}
        ts = time.time()
        self.train_power_model(series, trained_at=ts)
        with self.lock:
            metrics = dict(self.model_metrics)
        return {
            "ok": True,
            "trained_at": ts,
            "dataset_total": dataset_total,
            "mse": metrics.get("mse"),
            "mape": metrics.get("mape"),
            "n": metrics.get("n"),
        }

    def get_power_history(self, window_seconds: int = 180) -> List[Tuple[float, float]]:
        cutoff = time.time() - max(1, window_seconds)
        with self.lock:
            data = [(ts, p) for ts, p in self.power_history if ts >= cutoff]
            if not data:
                total = (self.phases["a"].act_power or 0.0) + (self.phases["b"].act_power or 0.0) + (self.phases["c"].act_power or 0.0)
                data = [(time.time(), float(total))]
        return data

    def forecast_power(self, horizon_seconds: int = 3, step_seconds: float = 1.0, history: Optional[List[Tuple[float, float]]] = None) -> List[Tuple[float, float]]:
        hist = history if history is not None else self.get_power_history(window_seconds=POWER_HISTORY_SECONDS)
        if not hist:
            return []
        last_ts = hist[-1][0]
        with self.lock:
            model = self.model
            ref_ts = self.model_ref_ts if self.model_ref_ts is not None else last_ts
        steps = max(1, int(horizon_seconds / step_seconds))

        # If model is not ready, fall back to a flat forecast
        if model is None:
            base = hist[-1][1]
            return [(last_ts + i * step_seconds, base) for i in range(1, steps + 1)]

        forecast: List[Tuple[float, float]] = []
        for i in range(1, steps + 1):
            target_ts = last_ts + i * step_seconds
            delta = target_ts - ref_ts
            try:
                pred = float(model.predict([[delta]])[0])
            except Exception:
                pred = hist[-1][1]
            forecast.append((target_ts, pred))
        return forecast

    def build_power_snapshot(self) -> Dict[str, Any]:
        history = self.get_power_history(window_seconds=POWER_HISTORY_SECONDS)
        forecast = self.forecast_power(history=history)
        current_ts, current_power = history[-1]
        with self.lock:
            model = dict(self.model_params)
            metrics = dict(self.model_metrics)
            dataset_total = len(self.power_dataset)
        return {
            "ts": current_ts,
            "current": round(current_power, 2),
            "history": [{"ts": ts, "w": round(p, 2)} for ts, p in history],
            "forecast": [{"ts": ts, "w": round(p, 2)} for ts, p in forecast],
            "model": {
                "trained_at": model.get("trained_at"),
                "mse": metrics.get("mse"),
                "mape": metrics.get("mape"),
                "n": metrics.get("n"),
                "dataset_total": dataset_total,
            },
        }

    def poll_home_assistant(self):
        try:
            total_power = 0.0
            a_w = ha_get(A_POWER) or 0.0
            b_w = ha_get(B_POWER) or 0.0
            c_w = ha_get(C_POWER) or 0.0
            with self.lock:
                self.phases["a"].act_power = float(a_w)
                self.phases["b"].act_power = float(b_w)
                self.phases["c"].act_power = float(c_w)

                self.phases["a"].voltage = ha_get(A_VOLT)
                self.phases["b"].voltage = ha_get(B_VOLT)
                self.phases["c"].voltage = ha_get(C_VOLT)

                self.phases["a"].current = ha_get(A_CURR)
                self.phases["b"].current = ha_get(B_CURR)
                self.phases["c"].current = ha_get(C_CURR)

                self.phases["a"].pf = ha_get(A_PF) or None
                self.phases["b"].pf = ha_get(B_PF) or None
                self.phases["c"].pf = ha_get(C_PF) or None

                total_power = float((self.phases["a"].act_power or 0.0) + (self.phases["b"].act_power or 0.0) + (self.phases["c"].act_power or 0.0))

            self.record_power_sample(total_power)

            # Integrate using monotonic time to avoid wall clock jumps
            now_mono = time.monotonic()
            if self.last_poll_mono is not None:
                dt = max(0.0, now_mono - self.last_poll_mono)
                if dt > 0:
                    self.integrate_energy(dt)
            self.last_poll_mono = now_mono

            # Persist every 30s using monotonic cadence
            if (now_mono - self.last_persist_mono) >= 30.0:
                self.persist()
                self.last_persist_mono = now_mono
        except Exception:
            pass

    # ---------- RPC builders ----------
    def em_get_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Validate channel id; Shelly 3EM uses id=0
        chan = params.get("id", 0)
        if chan not in (0,):
            raise ValueError("invalid id")
        with self.lock:
            a, b, c = self.phases["a"], self.phases["b"], self.phases["c"]
            raw_powers = (
                float(a.act_power or 0.0),
                float(b.act_power or 0.0),
                float(c.act_power or 0.0),
            )
            a_power, b_power, c_power = _apply_request_side_power_scaling(raw_powers)
            total_w = a_power + b_power + c_power

            if STRICT_MINIMAL_PAYLOAD:
                # Minimal contract: only the 4 power keys some gateways require
                return {
                    "a_act_power": round(a_power, 2),
                    "b_act_power": round(b_power, 2),
                    "c_act_power": round(c_power, 2),
                    "total_act_power": round(total_w, 2),
                }

            # Full-ish Shelly-like payload
            return {
                "id": 0,
                "a_voltage": a.voltage if a.voltage is not None else 230.0,
                "b_voltage": b.voltage if b.voltage is not None else 230.0,
                "c_voltage": c.voltage if c.voltage is not None else 230.0,
                "a_current": a.current if a.current is not None else 0.0,
                "b_current": b.current if b.current is not None else 0.0,
                "c_current": c.current if c.current is not None else 0.0,
                "a_act_power": round(a_power, 2),
                "b_act_power": round(b_power, 2),
                "c_act_power": round(c_power, 2),
                "a_pf": a.pf if a.pf is not None else 1.0,
                "b_pf": b.pf if b.pf is not None else 1.0,
                "c_pf": c.pf if c.pf is not None else 1.0,
                "frequency": self.frequency,
                "total_act_power": round(total_w, 2),
                "ts": now_ts(),
            }

    def emdata_get_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        chan = params.get("id", 0)
        if chan not in (0,):
            raise ValueError("invalid id")
        with self.lock:
            e = self.energy
            return {
                "id": 0,
                "a_total_act_energy": round(e.a_import, 6),
                "b_total_act_energy": round(e.b_import, 6),
                "c_total_act_energy": round(e.c_import, 6),
                "a_total_act_ret_energy": round(e.a_export, 6),
                "b_total_act_ret_energy": round(e.b_export, 6),
                "c_total_act_ret_energy": round(e.c_export, 6),
                "total_act": round(e.total_import, 6),
                "total_act_ret": round(e.total_export, 6),
                "period": max(0, now_ts() - e.since),
                "ts": now_ts(),
            }

    def emdata_reset_counters(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self.energy = EnergyCounters(since=now_ts())
            self.persist()
        if MODBUS_BRIDGE:
            MODBUS_BRIDGE.update()
        return {"ok": True, "ts": now_ts()}

    def shelly_get_status(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "ts": now_ts(),
            "sys": {
                "uptime": int(time.monotonic()),
                "mac": MAC,
                "model": MODEL,
                "fw_id": FW_ID,
                "device_id": DEVICE_ID,
                "time": datetime.now(timezone.utc).isoformat(),
                "app": APP_ID,
                "gen": GENERATION,
            },
            "em:0": self.em_get_status({"id": 0}),
            "emdata:0": self.emdata_get_status({"id": 0}),
        }
        print(payload)
        return payload

    def build_device_info(self) -> Dict[str, Any]:
        auth_enabled = False
        try:
            auth_enabled = bool(self.config.get("rpc", {}).get("auth", False))
        except Exception:
            auth_enabled = False

        info: Dict[str, Any] = {
            "name": DEVICE_ID,
            "id": DEVICE_ID,
            "app": APP_ID,
            "ver": FIRMWARE,
            "fw_id": FW_ID,
            "model": MODEL,
            "gen": GENERATION,
            "mac": MAC,
            "auth_en": auth_enabled,
            "auth_domain": None,
        }
        if SN:
            info["sn"] = SN
        if MANUFACTURER:
            info["manufacturer"] = MANUFACTURER
        return info

    def shelly_get_device_info(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return self.build_device_info()

    # Gen2 Sys.* helpers
    def sys_get_info(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": DEVICE_ID,
            "id": DEVICE_ID,
            "model": MODEL,
            "mac": MAC,
            "fw_id": FW_ID,
            "ver": FIRMWARE,
            "app": APP_ID,
            "uptime": int(time.monotonic()),
        }

    def sys_set_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.shelly_set_config(params)

    def shelly_get_config(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return self.config

    def shelly_set_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            if "config" in params and isinstance(params["config"], dict):
                self.config.update(params["config"])
            else:
                self.config.update(params)
            return {"ok": True}
        return {"ok": False}

    def shelly_ping(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {"pong": True, "ts": now_ts()}

    def shelly_reboot(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "ts": now_ts()}

    def mqtt_get_config(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {"enable": False, "server": None, "client_id": DEVICE_ID, "prefix": DEVICE_ID}

    def mqtt_set_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True}

    def mqtt_status(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {"connected": False, "client_id": DEVICE_ID, "server": None}

    def sys_get_status(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ram_total": 64 * 1024 * 1024, "ram_free": 48 * 1024 * 1024,
            "fs_size": 64 * 1024 * 1024, "fs_free": 60 * 1024 * 1024,
            "time": datetime.now(timezone.utc).isoformat(), "device": DEVICE_ID,
        }


# -----------------------------
# Modbus TCP bridge
# -----------------------------
MODBUS_LOG = logging.getLogger("virtual_shelly.modbus")


def _pack_register_pair(value: float, fmt: str) -> List[int]:
    if fmt == ">f":
        raw = struct.pack(fmt, float(value))
    elif fmt == ">I":
        raw = struct.pack(fmt, int(value) & 0xFFFFFFFF)
    elif fmt == ">i":
        raw = struct.pack(fmt, int(value))
    else:
        raise ValueError(f"Unsupported pack format {fmt}")
    hi, lo = struct.unpack(">HH", raw)
    return [hi, lo]


def _encode_ascii_registers(text: str, register_count: int) -> List[int]:
    data = (text or "").encode("ascii", errors="ignore")[: register_count * 2]
    data = data.ljust(register_count * 2, b"\x00")
    regs: List[int] = []
    for i in range(register_count):
        hi = data[2 * i]
        lo = data[2 * i + 1]
        regs.append((hi << 8) | lo)
    return regs


if StartAsyncTcpServer is not None:
    class ShellyModbusBridge:
        """Maintains a Modbus register image reflecting the virtual meter state."""

    def __init__(self, vm: VirtualPro3EM):
        self.vm = vm
        self.lock = threading.RLock()
        self.registers: Dict[int, int] = {}
        self.update()

    def _write_pair(self, regs: Dict[int, int], address: int, values: List[int]) -> None:
        regs[address] = values[0]
        regs[address + 1] = values[1]

    def _write_ascii(self, regs: Dict[int, int], address: int, register_count: int, text: str) -> None:
        for idx, value in enumerate(_encode_ascii_registers(text, register_count)):
            regs[address + idx] = value

    def build_image(self) -> Dict[int, int]:
        regs: Dict[int, int] = {}
        with self.vm.lock:
            phases = self.vm.phases
            energy = self.vm.energy
            freq = float(self.vm.frequency or 0.0)

            # Instantaneous metrics (float32, big-endian)
            self._write_pair(regs, 3000, _pack_register_pair(freq, ">f"))
            self._write_pair(regs, 3002, _pack_register_pair(phases["a"].voltage or 0.0, ">f"))
            self._write_pair(regs, 3004, _pack_register_pair(phases["b"].voltage or 0.0, ">f"))
            self._write_pair(regs, 3006, _pack_register_pair(phases["c"].voltage or 0.0, ">f"))

            self._write_pair(regs, 3010, _pack_register_pair(phases["a"].current or 0.0, ">f"))
            self._write_pair(regs, 3012, _pack_register_pair(phases["b"].current or 0.0, ">f"))
            self._write_pair(regs, 3014, _pack_register_pair(phases["c"].current or 0.0, ">f"))

            raw_powers = (
                float(phases["a"].act_power or 0.0),
                float(phases["b"].act_power or 0.0),
                float(phases["c"].act_power or 0.0),
            )
            a_power, b_power, c_power = _apply_request_side_power_scaling(raw_powers)
            total_power = a_power + b_power + c_power
            self._write_pair(regs, 3020, _pack_register_pair(a_power, ">f"))
            self._write_pair(regs, 3022, _pack_register_pair(b_power, ">f"))
            self._write_pair(regs, 3024, _pack_register_pair(c_power, ">f"))
            self._write_pair(regs, 3026, _pack_register_pair(total_power, ">f"))

            self._write_pair(regs, 3030, _pack_register_pair(phases["a"].pf if phases["a"].pf is not None else 1.0, ">f"))
            self._write_pair(regs, 3032, _pack_register_pair(phases["b"].pf if phases["b"].pf is not None else 1.0, ">f"))
            self._write_pair(regs, 3034, _pack_register_pair(phases["c"].pf if phases["c"].pf is not None else 1.0, ">f"))

            # Energy counters (kWh as float32)
            self._write_pair(regs, 3100, _pack_register_pair(energy.a_import, ">f"))
            self._write_pair(regs, 3102, _pack_register_pair(energy.b_import, ">f"))
            self._write_pair(regs, 3104, _pack_register_pair(energy.c_import, ">f"))
            self._write_pair(regs, 3106, _pack_register_pair(energy.a_export, ">f"))
            self._write_pair(regs, 3108, _pack_register_pair(energy.b_export, ">f"))
            self._write_pair(regs, 3110, _pack_register_pair(energy.c_export, ">f"))
            self._write_pair(regs, 3112, _pack_register_pair(energy.total_import, ">f"))
            self._write_pair(regs, 3114, _pack_register_pair(energy.total_export, ">f"))

            # Timestamp and uptime (uint32)
            self._write_pair(regs, 3200, _pack_register_pair(now_ts(), ">I"))
            self._write_pair(regs, 3202, _pack_register_pair(int(time.monotonic()), ">I"))

            # Device metadata (ASCII, two chars per register)
            self._write_ascii(regs, 3300, 8, DEVICE_ID)
            self._write_ascii(regs, 3310, 6, MODEL)
            self._write_ascii(regs, 3320, 6, FIRMWARE)
            self._write_ascii(regs, 3330, 6, MAC)

            # Status flags (uint16)
            regs[3400] = 1  # device online
            regs[3401] = 3  # number of phases

        return regs

    def update(self) -> None:
        snapshot = self.build_image()
        with self.lock:
            self.registers = snapshot

    def get_values(self, address: int, count: int) -> List[int]:
        with self.lock:
            return [self.registers.get(address + offset, 0) & 0xFFFF for offset in range(count)]

    def handle_write(self, address: int, values: List[int]) -> None:
        if not values:
            return
        if address == 4200 and values[0] == 1:
            MODBUS_LOG.info("Modbus request: reset energy counters")
            self.vm.emdata_reset_counters({})
            self.update()

        def _write_pair(self, regs: Dict[int, int], address: int, values: List[int]) -> None:
            regs[address] = values[0]
            regs[address + 1] = values[1]

        def _write_ascii(self, regs: Dict[int, int], address: int, register_count: int, text: str) -> None:
            for idx, value in enumerate(_encode_ascii_registers(text, register_count)):
                regs[address + idx] = value

        def build_image(self) -> Dict[int, int]:
            regs: Dict[int, int] = {}
            with self.vm.lock:
                phases = self.vm.phases
                energy = self.vm.energy
                freq = float(self.vm.frequency or 0.0)

                # Instantaneous metrics (float32, big-endian)
                self._write_pair(regs, 3000, _pack_register_pair(freq, ">f"))
                self._write_pair(regs, 3002, _pack_register_pair(phases["a"].voltage or 0.0, ">f"))
                self._write_pair(regs, 3004, _pack_register_pair(phases["b"].voltage or 0.0, ">f"))
                self._write_pair(regs, 3006, _pack_register_pair(phases["c"].voltage or 0.0, ">f"))

                self._write_pair(regs, 3010, _pack_register_pair(phases["a"].current or 0.0, ">f"))
                self._write_pair(regs, 3012, _pack_register_pair(phases["b"].current or 0.0, ">f"))
                self._write_pair(regs, 3014, _pack_register_pair(phases["c"].current or 0.0, ">f"))

                total_power = (phases["a"].act_power or 0.0) + (phases["b"].act_power or 0.0) + (phases["c"].act_power or 0.0)
                self._write_pair(regs, 3020, _pack_register_pair(phases["a"].act_power, ">f"))
                self._write_pair(regs, 3022, _pack_register_pair(phases["b"].act_power, ">f"))
                self._write_pair(regs, 3024, _pack_register_pair(phases["c"].act_power, ">f"))
                self._write_pair(regs, 3026, _pack_register_pair(total_power, ">f"))

                self._write_pair(regs, 3030, _pack_register_pair(phases["a"].pf if phases["a"].pf is not None else 1.0, ">f"))
                self._write_pair(regs, 3032, _pack_register_pair(phases["b"].pf if phases["b"].pf is not None else 1.0, ">f"))
                self._write_pair(regs, 3034, _pack_register_pair(phases["c"].pf if phases["c"].pf is not None else 1.0, ">f"))

                # Energy counters (kWh as float32)
                self._write_pair(regs, 3100, _pack_register_pair(energy.a_import, ">f"))
                self._write_pair(regs, 3102, _pack_register_pair(energy.b_import, ">f"))
                self._write_pair(regs, 3104, _pack_register_pair(energy.c_import, ">f"))
                self._write_pair(regs, 3106, _pack_register_pair(energy.a_export, ">f"))
                self._write_pair(regs, 3108, _pack_register_pair(energy.b_export, ">f"))
                self._write_pair(regs, 3110, _pack_register_pair(energy.c_export, ">f"))
                self._write_pair(regs, 3112, _pack_register_pair(energy.total_import, ">f"))
                self._write_pair(regs, 3114, _pack_register_pair(energy.total_export, ">f"))

                # Timestamp and uptime (uint32)
                self._write_pair(regs, 3200, _pack_register_pair(now_ts(), ">I"))
                self._write_pair(regs, 3202, _pack_register_pair(int(time.monotonic()), ">I"))

                # Device metadata (ASCII, two chars per register)
                self._write_ascii(regs, 3300, 8, DEVICE_ID)
                self._write_ascii(regs, 3310, 6, MODEL)
                self._write_ascii(regs, 3320, 6, FIRMWARE)
                self._write_ascii(regs, 3330, 6, MAC)

                # Status flags (uint16)
                regs[3400] = 1  # device online
                regs[3401] = 3  # number of phases

            return regs

        def update(self) -> None:
            snapshot = self.build_image()
            with self.lock:
                self.registers = snapshot

        def get_values(self, address: int, count: int) -> List[int]:
            with self.lock:
                return [self.registers.get(address + offset, 0) & 0xFFFF for offset in range(count)]

        def handle_write(self, address: int, values: List[int]) -> None:
            if not values:
                return
            if address == 4200 and values[0] == 1:
                MODBUS_LOG.info("Modbus request: reset energy counters")
                self.vm.emdata_reset_counters({})
                self.update()

    class ShellyModbusInputBlock(ModbusSparseDataBlock):
        def __init__(self, bridge: ShellyModbusBridge):
            super().__init__({})
            self.bridge = bridge

        def getValues(self, address: int, count: int = 1) -> List[int]:  # type: ignore[override]
            return self.bridge.get_values(address, count)

    class ShellyModbusHoldingBlock(ModbusSparseDataBlock):
        def __init__(self, bridge: ShellyModbusBridge):
            super().__init__({})
            self.bridge = bridge

        def getValues(self, address: int, count: int = 1) -> List[int]:  # type: ignore[override]
            return self.bridge.get_values(address, count)

        def setValues(self, address: int, values: List[int]) -> None:  # type: ignore[override]
            self.bridge.handle_write(address, values)

    def start_modbus_server(bridge: ShellyModbusBridge) -> None:
        if not MODBUS_ENABLE:
            return
        # Imports available if we got here
        slave = ModbusSlaveContext(
            di=ModbusSequentialDataBlock(0, [0] * 4),
            co=ModbusSequentialDataBlock(0, [0] * 4),
            hr=ShellyModbusHoldingBlock(bridge),
            ir=ShellyModbusInputBlock(bridge),
            zero_mode=True,
        )
        context = ModbusServerContext(slaves={MODBUS_UNIT_ID: slave}, single=False)

        identity = ModbusDeviceIdentification()
        identity.VendorName = "Shelly"
        identity.ProductCode = "SP3EM"
        identity.VendorUrl = "https://shelly.cloud"
        identity.ProductName = "Shelly Pro 3EM (virtual)"
        identity.ModelName = MODEL
        identity.MajorMinorRevision = FIRMWARE
        try:
            identity.UnitIdentifier = MODBUS_UNIT_ID  # type: ignore[attr-defined]
        except Exception:
            pass

        async def _serve():
            MODBUS_LOG.info("Starting Modbus TCP on %s:%s", MODBUS_BIND, MODBUS_PORT)
            await StartAsyncTcpServer(
                context=context,
                identity=identity,
                address=(MODBUS_BIND, MODBUS_PORT),
                allow_reuse_address=True,
            )

        def _runner():
            try:
                asyncio.run(_serve())
            except Exception as exc:  # pragma: no cover - background log only
                MODBUS_LOG.error("Modbus server stopped: %s", exc)

        threading.Thread(target=_runner, daemon=True).start()

# -----------------------------
# JSON-RPC dispatcher
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("virtual_shelly")

app = FastAPI(title="Virtual Shelly Pro 3EM RPC")
VM = VirtualPro3EM()
MODBUS_BRIDGE = None
if MODBUS_ENABLE and StartAsyncTcpServer is not None:
    try:
        MODBUS_BRIDGE = ShellyModbusBridge(VM)  # type: ignore[name-defined]
        start_modbus_server(MODBUS_BRIDGE)      # type: ignore[name-defined]
    except Exception as exc:  # pragma: no cover - background log only
        MODBUS_LOG.error("Failed to start Modbus bridge: %s", exc)
        MODBUS_BRIDGE = None

if CORS_ENABLE:
    origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()] or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# -----------------------------
# WS Notify infrastructure (FastAPI /rpc)
# -----------------------------
WS_RPC_CLIENTS: "set[WebSocket]" = set()
WS_QUEUE: Optional[asyncio.Queue] = None
NOTIFIER_TASK: Optional[asyncio.Task] = None
APP_LOOP: Optional[asyncio.AbstractEventLoop] = None
_last_notify_mono: float = time.monotonic()
_last_notify_values: Optional[Tuple[float, float, float, float]] = None

# -----------------------------
# Metrics (simple Prometheus-like text output)
# -----------------------------
HTTP_RPC_METHODS = MET.HTTP_RPC_METHODS
HTTP_RPC_ERRORS = MET.HTTP_RPC_ERRORS
WS_NOTIFY_TOTAL = None  # shim for references below
WS_RPC_MESSAGES = None
UDP_PACKETS_TOTAL = None
UDP_REPLIES_TOTAL = None


def _build_notify_status() -> str:
    # Keep payload concise: include EM.GetStatus and timestamp
    payload = {
        "method": "NotifyStatus",
        "params": {
            "ts": now_ts(),
            "em:0": VM.em_get_status({"id": 0}),
        },
    }
    return json.dumps(payload, separators=(",", ":"))


def _build_notify_full_status() -> str:
    payload = {"method": "NotifyFullStatus", "params": VM.shelly_get_status({})}
    return json.dumps(payload, separators=(",", ":"))


async def _ws_notifier_worker():
    global WS_QUEUE
    assert WS_QUEUE is not None
    while True:
        msg = await WS_QUEUE.get()
        stale: list[WebSocket] = []
        for ws in list(WS_RPC_CLIENTS):
            try:
                await ws.send_text(msg)
            except Exception:
                stale.append(ws)
        for ws in stale:
            try:
                WS_RPC_CLIENTS.discard(ws)
            except Exception:
                pass


def _enqueue_ws_broadcast(msg: str):
    global APP_LOOP, WS_QUEUE
    if APP_LOOP is None or WS_QUEUE is None:
        return
    try:
        APP_LOOP.call_soon_threadsafe(WS_QUEUE.put_nowait, msg)
    except Exception:
        pass

METHODS = RPC.build_methods(VM)

def _handle_rpc_call(req: Dict[str, Any]) -> Dict[str, Any]:
    return RPC.handle_rpc_call(METHODS, req, log)

def _rpc_response_bytes(data: bytes) -> bytes:
    return RPC.rpc_response_bytes(METHODS, data, log)

# -----------------------------
# HTTP /rpc
# -----------------------------
@app.post("/rpc")
async def rpc_endpoint(request: Request) -> JSONResponse:
    client_host = getattr(request.client, "host", None) if request.client else None
    _register_request_ip(client_host)
    body = await request.json()
    try:
        if isinstance(body, dict):
            log.info("HTTP POST /rpc method=%s", body.get("method"))
            if isinstance(body, dict) and body.get("method"):
                HTTP_RPC_METHODS[body.get("method")] += 1
                try:
                    ip = request.client.host if request.client else "?"
                    MET.add_recent_http(now_ts(), ip, body.get("method"), "POST")
                except Exception:
                    pass
        else:
            log.info("HTTP POST /rpc batch size=%d", len(body) if isinstance(body, list) else 1)
    except Exception:
        pass
    if isinstance(body, list):
        resp = []
        for item in body:
            r = _handle_rpc_call(item)
            if isinstance(item, dict) and item.get("method") and "error" in r:
                HTTP_RPC_ERRORS[item.get("method")] += 1
            try:
                if isinstance(item, dict) and item.get("method"):
                    ip = request.client.host if request.client else "?"
                    MET.add_recent_http(now_ts(), ip, item.get("method"), "POST")
            except Exception:
                pass
            resp.append(r)
        return JSONResponse(resp)
    else:
        r = _handle_rpc_call(body)
        if isinstance(body, dict) and body.get("method") and "error" in r:
            HTTP_RPC_ERRORS[body.get("method")] += 1
        return JSONResponse(r)

@app.get("/rpc")
async def rpc_get(request: Request) -> JSONResponse:
    # Support GET-style RPC: /rpc?method=EM.GetStatus&id=0 or /rpc?method=Shelly.GetConfig
    client_host = getattr(request.client, "host", None) if request.client else None
    _register_request_ip(client_host)
    qp = dict(request.query_params)
    # Batch GET support via ?batch=[{...},{...}]
    if "batch" in qp:
        try:
            batch = json.loads(qp["batch"]) or []
            if not isinstance(batch, list):
                raise ValueError
        except Exception:
            return JSONResponse({"error": "invalid batch"}, status_code=400)
        results = []
        for item in batch:
            if not isinstance(item, dict):
                continue
            res = _handle_rpc_call(item)
            results.append(res.get("result", res))
        try:
            ip = request.client.host if request.client else "?"
            for item in batch:
                if isinstance(item, dict) and item.get("method"):
                    MET.add_recent_http(now_ts(), ip, item.get("method"), "GET")
        except Exception:
            pass
        return JSONResponse(results)
    method = qp.pop("method", None)
    if not method:
        return JSONResponse({"error": "method required"}, status_code=400)
    try:
        log.info("HTTP GET /rpc method=%s", method)
        HTTP_RPC_METHODS[method] += 1
    except Exception:
        pass
    try:
        ip = request.client.host if request.client else "?"
        MET.add_recent_http(now_ts(), ip, method, "GET")
    except Exception:
        pass
    # Build params: prefer explicit JSON in 'params', else use remaining query items
    params: Dict[str, Any] = {}
    if "params" in qp:
        try:
            params = json.loads(qp.pop("params")) or {}
        except Exception:
            params = {}
    # Merge remaining simple query items
    for k, v in qp.items():
        if v is None:
            continue
        # attempt int/float cast for common fields like id
        if v.isdigit():
            params[k] = int(v)
        else:
            try:
                params[k] = float(v)
            except Exception:
                params[k] = v
    req = {"id": None, "method": method, "params": params}
    resp = _handle_rpc_call(req)
    # Return only the result to mimic Shelly GET semantics
    if "result" in resp:
        return JSONResponse(resp["result"])
    return JSONResponse(resp, status_code=400)

@app.get("/rpc/{method}")
async def rpc_get_method(method: str, request: Request) -> JSONResponse:
    # Support GET-style RPC: /rpc/EM.GetStatus?id=0
    client_host = getattr(request.client, "host", None) if request.client else None
    _register_request_ip(client_host)
    qp = dict(request.query_params)
    if "batch" in qp:
        try:
            batch = json.loads(qp["batch"]) or []
            if not isinstance(batch, list):
                raise ValueError
        except Exception:
            return JSONResponse({"error": "invalid batch"}, status_code=400)
        results = []
        for params in batch:
            if not isinstance(params, dict):
                continue
            res = _handle_rpc_call({"id": None, "method": method, "params": params})
            results.append(res.get("result", res))
        try:
            ip = request.client.host if request.client else "?"
            MET.add_recent_http(now_ts(), ip, method, "GET")
        except Exception:
            pass
        return JSONResponse(results)
    try:
        log.info("HTTP GET /rpc/%s", method)
        HTTP_RPC_METHODS[method] += 1
    except Exception:
        pass
    try:
        ip = request.client.host if request.client else "?"
        MET.add_recent_http(now_ts(), ip, method, "GET")
    except Exception:
        pass
    params: Dict[str, Any] = {}
    if "params" in qp:
        try:
            params = json.loads(qp.pop("params")) or {}
        except Exception:
            params = {}
    for k, v in qp.items():
        if v is None:
            continue
        if v.isdigit():
            params[k] = int(v)
        else:
            try:
                params[k] = float(v)
            except Exception:
                params[k] = v
    req = {"id": None, "method": method, "params": params}
    resp = _handle_rpc_call(req)
    if "result" in resp:
        return JSONResponse(resp["result"])
    return JSONResponse(resp, status_code=400)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "ts": now_ts(), "device": DEVICE_ID}

@app.get("/metrics")
def metrics():
    return PlainTextResponse(MET.metrics_text(len(WS_RPC_CLIENTS)))


@app.get("/admin/overview")
def admin_overview():
    # WS connected IPs (best-effort)
    ws_ips: List[str] = []
    try:
        for ws in list(WS_RPC_CLIENTS):
            try:
                ip = ws.client.host if getattr(ws, "client", None) else None
                if not ip and hasattr(ws, "remote_address") and ws.remote_address:
                    ip = ws.remote_address[0]
                if ip:
                    ws_ips.append(ip)
            except Exception:
                pass
    except Exception:
        pass
    return JSONResponse(MET.build_admin_overview(VM, ws_ips, now_ts))

@app.get("/")
def root():
    return {"service": "virtual-shelly-pro-3em", "rpc": "/rpc"}


@app.get("/ui/power")
def ui_power_snapshot():
    return JSONResponse(VM.build_power_snapshot())


@app.post("/ui/train")
def ui_trigger_training():
    try:
        result = VM.trigger_full_training()
        status = 200 if result.get("ok") else 400
        return JSONResponse(result, status_code=status)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.get("/ui")
def ui_page():
    html = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Virtual Shelly 3EM Pro — Status</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; color: #0b0b0b; }
    h1 { font-size: 20px; margin: 0 0 12px 0; }
    h2 { font-size: 16px; margin: 18px 0 10px 0; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { border: 1px solid #e2e2e2; border-radius: 8px; padding: 12px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; font-size: 14px; }
    th { background: #fafafa; font-weight: 600; }
    .muted { color: #666; font-size: 12px; }
    code { background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }
  </style>
  <script>
    async function fetchOverview() {
      try {
        const res = await fetch('/admin/overview');
        const data = await res.json();
        render(data);
      } catch (e) {
        console.error('Fetch error', e);
      }
    }

    function fmt(n, digits=2) {
      if (n === null || n === undefined) return '-';
      if (typeof n === 'number') return n.toFixed(digits);
      return String(n);
    }

    function render(d) {
      document.getElementById('device').textContent = (d.device?.id || 'device') + ' (' + (d.device?.model || '') + ' ' + (d.device?.ver || '') + ')';
      document.getElementById('updated').textContent = new Date(d.ts * 1000).toLocaleString();

      const em = d.values?.em || {};
      const emdata = d.values?.emdata || {};

      const cur = [
        ['A Voltage (V)', fmt(em.a_voltage)],
        ['B Voltage (V)', fmt(em.b_voltage)],
        ['C Voltage (V)', fmt(em.c_voltage)],
        ['A Current (A)', fmt(em.a_current)],
        ['B Current (A)', fmt(em.b_current)],
        ['C Current (A)', fmt(em.c_current)],
        ['A Power (W)', fmt(em.a_act_power)],
        ['B Power (W)', fmt(em.b_act_power)],
        ['C Power (W)', fmt(em.c_act_power)],
        ['Power Factor A', fmt(em.a_pf, 3)],
        ['Power Factor B', fmt(em.b_pf, 3)],
        ['Power Factor C', fmt(em.c_pf, 3)],
        ['Frequency (Hz)', fmt(em.frequency, 2)],
        ['Total Power (W)', fmt(em.total_act_power)],
      ];
      document.getElementById('current-tbody').innerHTML = cur.map(([k,v]) => `<tr><td>${k}</td><td><b>${v}</b></td></tr>`).join('');

      const energy = [
        ['A Import (kWh)', fmt(emdata.a_total_act_energy, 3)],
        ['B Import (kWh)', fmt(emdata.b_total_act_energy, 3)],
        ['C Import (kWh)', fmt(emdata.c_total_act_energy, 3)],
        ['A Export (kWh)', fmt(emdata.a_total_act_ret_energy, 3)],
        ['B Export (kWh)', fmt(emdata.b_total_act_ret_energy, 3)],
        ['C Export (kWh)', fmt(emdata.c_total_act_ret_energy, 3)],
        ['Total Import (kWh)', fmt(emdata.total_act, 3)],
        ['Total Export (kWh)', fmt(emdata.total_act_ret, 3)],
        ['Period (s)', fmt(emdata.period, 0)],
      ];
      document.getElementById('energy-tbody').innerHTML = energy.map(([k,v]) => `<tr><td>${k}</td><td><b>${v}</b></td></tr>`).join('');

      // Metrics
      const http = d.metrics?.http || {};
      const ws = d.metrics?.ws || {};
      const udp = d.metrics?.udp || {};
      document.getElementById('http-total').textContent = http.total ?? 0;
      const bym = http.by_method || {};
      const rows = Object.keys(bym).sort().map(m => `<tr><td><code>${m}</code></td><td>${bym[m]}</td></tr>`).join('');
      document.getElementById('http-by-method').innerHTML = rows || '<tr><td colspan="2" class="muted">No calls yet</td></tr>';
      document.getElementById('ws-stats').textContent = `${ws.clients || 0} clients, ${ws.rpc_messages || 0} RPC msgs, ${ws.notify_total || 0} notifies`;
      document.getElementById('udp-stats').textContent = `${udp.packets || 0} packets, ${udp.replies || 0} replies`;

      // Clients
      function listToRows(arr, cols) {
        return (arr||[]).slice().reverse().map(x => `<tr>${cols.map(c => `<td>${x[c] ?? '-'}</td>`).join('')}</tr>`).join('');
      }
      document.getElementById('http-recent').innerHTML = listToRows(d.clients?.http_recent, ['ts','ip','verb','method']);
      document.getElementById('ws-recent').innerHTML = listToRows(d.clients?.ws_recent, ['ts','ip','event','method']);
      document.getElementById('udp-recent').innerHTML = listToRows(d.clients?.udp_recent, ['ts','ip','method']);
      const wsCon = (d.clients?.ws_connected || []).map(ip => `<code>${ip}</code>`).join(', ');
      document.getElementById('ws-connected').innerHTML = wsCon || '<span class="muted">None</span>';

      // Unique clients
      const httpU = d.clients?.http_unique || [];
      const wsU = d.clients?.ws_unique || [];
      const udpU = d.clients?.udp_unique || [];
      document.getElementById('http-uniq-count').textContent = httpU.length;
      document.getElementById('ws-uniq-count').textContent = wsU.length;
      document.getElementById('udp-uniq-count').textContent = udpU.length;
      document.getElementById('http-uniq-list').innerHTML = httpU.map(ip => `<code>${ip}</code>`).join(', ');
      document.getElementById('ws-uniq-list').innerHTML = wsU.map(ip => `<code>${ip}</code>`).join(', ');
      document.getElementById('udp-uniq-list').innerHTML = udpU.map(ip => `<code>${ip}</code>`).join(', ');
    }

    window.addEventListener('load', () => {
      fetchOverview();
      setInterval(fetchOverview, 5000);
    });
  </script>
</head>
<body>
  <h1>Virtual Shelly 3EM Pro — Status <span class="muted" id="device"></span></h1>
  <div class="muted">Updated: <span id="updated">-</span></div>

  <div class="grid">
    <div class="card">
      <h2>Current Values</h2>
      <table>
        <tbody id="current-tbody"></tbody>
      </table>
    </div>
    <div class="card">
      <h2>Energy Counters</h2>
      <table>
        <tbody id="energy-tbody"></tbody>
      </table>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>HTTP Metrics</h2>
      <div>Total: <b id="http-total">0</b></div>
      <table>
        <thead><tr><th>Method</th><th>Count</th></tr></thead>
        <tbody id="http-by-method"></tbody>
      </table>
    </div>
    <div class="card">
      <h2>WS & UDP Metrics</h2>
      <div>WebSocket: <b id="ws-stats">-</b></div>
      <div>UDP: <b id="udp-stats">-</b></div>
      <div style="margin-top:8px">WS Connected: <span id="ws-connected"></span></div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h2>Unique Clients</h2>
    <div>HTTP: <b id="http-uniq-count">0</b> <span id="http-uniq-list"></span></div>
    <div>WS: <b id="ws-uniq-count">0</b> <span id="ws-uniq-list"></span></div>
    <div>UDP RPC: <b id="udp-uniq-count">0</b> <span id="udp-uniq-list"></span></div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Recent HTTP Clients</h2>
      <table>
        <thead><tr><th>TS</th><th>IP</th><th>Verb</th><th>Method</th></tr></thead>
        <tbody id="http-recent"></tbody>
      </table>
    </div>
    <div class="card">
      <h2>Recent WS Clients</h2>
      <table>
        <thead><tr><th>TS</th><th>IP</th><th>Event</th><th>Method</th></tr></thead>
        <tbody id="ws-recent"></tbody>
      </table>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h2>Recent UDP Clients</h2>
    <table>
      <thead><tr><th>TS</th><th>IP</th><th>Method</th></tr></thead>
      <tbody id="udp-recent"></tbody>
    </table>
  </div>

</body>
</html>
"""
    return HTMLResponse(UI.dashboard_html())

@app.get("/shelly")
def shelly_http_info():
    # Shelly Gen2-compatible device info endpoint
    auth_enabled = False
    try:
        auth_enabled = bool(VM.config.get("rpc", {}).get("auth", False))
    except Exception:
        auth_enabled = False
    # Link/discovery info (basic static flags for emulator)
    link = {
        "wifi_sta": {"connected": False},
        "eth": {"connected": True},
        "discoverable": True,
    }
    return {
        "name": DEVICE_ID,
        "id": DEVICE_ID,
        "app": APP_ID,
        "ver": FIRMWARE,
        "fw_id": FW_ID,
        "model": MODEL,
        "gen": GENERATION,
        "mac": MAC,
        "sn": SN,
        "manufacturer": MANUFACTURER,
        "auth_en": auth_enabled,
        "auth_domain": None,
        "uptime": int(time.monotonic()),
        **link,
    }

# -----------------------------
# Background: HA poller
# -----------------------------
def poll_loop():
    while True:
        VM.poll_home_assistant()
        if MODBUS_BRIDGE:
            MODBUS_BRIDGE.update()
        # Throttled WS notify broadcast
        try:
            global _last_notify_mono
            global _last_notify_values
            now_m = time.monotonic()
            if (now_m - _last_notify_mono) >= WS_NOTIFY_INTERVAL:
                # Coalesce: only broadcast if power changed beyond EPS
                with VM.lock:
                    a = VM.phases["a"].act_power or 0.0
                    b = VM.phases["b"].act_power or 0.0
                    c = VM.phases["c"].act_power or 0.0
                total = a + b + c
                current = (a, b, c, total)
                changed = False
                if _last_notify_values is None:
                    changed = True
                else:
                    for old, new in zip(_last_notify_values, current):
                        if abs((new or 0.0) - (old or 0.0)) >= WS_NOTIFY_EPS:
                            changed = True
                            break
                if changed:
                    _enqueue_ws_broadcast(_build_notify_status())
                    try:
                        MET.WS_NOTIFY_TOTAL += 1
                    except Exception:
                        pass
                    _last_notify_values = current
                    _last_notify_mono = now_m
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)
threading.Thread(target=poll_loop, daemon=True).start()

# -----------------------------
# WebSockets 6010–6022
# -----------------------------
async def ws_handler(websocket):
    # Raw WS fan-out ports; echo JSON-RPC responses back.
    remote = None
    try:
        addr = getattr(websocket, "remote_address", None)
        if isinstance(addr, tuple) and addr:
            remote = addr[0]
    except Exception:
        remote = None
    # Track connect
    try:
        _register_request_ip(remote)
        MET.add_recent_ws(now_ts(), remote or "?", "connect")
    except Exception:
        pass
    try:
        async for message in websocket:
            try:
                _register_request_ip(remote)
                method = None
                try:
                    obj = json.loads(message)
                    if isinstance(obj, dict):
                        method = obj.get("method")
                except Exception:
                    method = None
                MET.add_recent_ws(now_ts(), remote or "?", "message", method)
            except Exception:
                pass
            resp = _rpc_response_bytes(message.encode()).decode()
            await websocket.send(resp)
    finally:
        try:
            MET.add_recent_ws(now_ts(), remote or "?", "disconnect")
        except Exception:
            pass

@app.on_event("startup")
async def _on_startup():
    global WS_QUEUE, NOTIFIER_TASK, APP_LOOP
    APP_LOOP = asyncio.get_running_loop()
    WS_QUEUE = asyncio.Queue()
    NOTIFIER_TASK = asyncio.create_task(_ws_notifier_worker())
    log.info("Service startup: device_id=%s model=%s fw=%s http_port=%s", DEVICE_ID, MODEL, FIRMWARE, HTTP_PORT)


@app.on_event("shutdown")
async def _on_shutdown():
    global NOTIFIER_TASK
    if NOTIFIER_TASK:
        try:
            NOTIFIER_TASK.cancel()
        except Exception:
            pass


@app.websocket("/rpc")
async def ws_rpc(websocket: WebSocket):
    # WebSocket endpoint compatible with Shelly Gen2 ws://<ip>/rpc
    client = getattr(websocket, "client", None)
    host = getattr(client, "host", None) if client else None
    _register_request_ip(host)
    await websocket.accept()
    WS_RPC_CLIENTS.add(websocket)
    # Initial full status
    try:
        await websocket.send_text(_build_notify_full_status())
        log.info("WS /rpc connected; clients=%d", len(WS_RPC_CLIENTS))
    except Exception:
        pass
    try:
        ip = websocket.client.host if getattr(websocket, "client", None) else "?"
        MET.add_recent_ws(now_ts(), ip or "?", "connect")
    except Exception:
        pass
    try:
        while True:
            message = await websocket.receive_text()
            try:
                MET.WS_RPC_MESSAGES += 1
            except Exception:
                pass
            try:
                ip = websocket.client.host if getattr(websocket, "client", None) else "?"
                method = None
                try:
                    obj = json.loads(message)
                    if isinstance(obj, dict):
                        method = obj.get("method")
                except Exception:
                    method = None
                MET.add_recent_ws(now_ts(), ip or "?", "message", method)
            except Exception:
                pass
            _register_request_ip(host)
            resp = _rpc_response_bytes(message.encode()).decode()
            await websocket.send_text(resp)
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        try:
            WS_RPC_CLIENTS.discard(websocket)
        except Exception:
            pass
        try:
            log.info("WS /rpc disconnected; clients=%d", len(WS_RPC_CLIENTS))
        except Exception:
            pass
        try:
            ip = websocket.client.host if getattr(websocket, "client", None) else "?"
            MET.add_recent_ws(now_ts(), ip or "?", "disconnect")
        except Exception:
            pass

async def ws_serve_on_ports(start_port: int, end_port: int):
    servers = []
    for port in range(start_port, end_port + 1):
        srv = await websockets.serve(ws_handler, "0.0.0.0", port, ping_interval=30, max_size=1 << 22)
        servers.append(srv)
    await asyncio.gather(*[srv.wait_closed() for srv in servers])

def ws_thread():
    asyncio.run(ws_serve_on_ports(WS_PORT_START, WS_PORT_END))
threading.Thread(target=ws_thread, daemon=True).start()

# -----------------------------
# UDP RPC: multi-port (e.g., 1010, 2220)
# -----------------------------
def _udp_decimal_enforcer(power: float) -> float:
    # Mirror b2500-meter behavior to always have a decimal place
    decimal_point_enforcer = 0.001
    try:
        p = float(power)
    except Exception:
        return decimal_point_enforcer
    if abs(p) < 0.1:
        return decimal_point_enforcer
    add = decimal_point_enforcer if (p == round(p) or p == 0) else 0.0
    return round(p + add, 1)


def _udp_build_response(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Implement Shelly-like UDP RPC used by b2500-meter (not JSON-RPC 2.0)
    if not isinstance(obj, dict):
        return None
    req_id = obj.get("id")
    method = obj.get("method")
    params = obj.get("params", {}) or {}
    # b2500-meter expects params.id to be an int and uses it to select channel
    if not isinstance(params.get("id"), int):
        return None

    if method == "EM.GetStatus":
        with VM.lock:
            raw_powers = (
                float(VM.phases["a"].act_power or 0.0),
                float(VM.phases["b"].act_power or 0.0),
                float(VM.phases["c"].act_power or 0.0),
            )
        scaled = _apply_request_side_power_scaling(raw_powers)
        a = _udp_decimal_enforcer(scaled[0])
        b = _udp_decimal_enforcer(scaled[1])
        c = _udp_decimal_enforcer(scaled[2])
        total = round(sum(scaled), 3)
        if total == round(total) or total == 0:
            total = total + 0.001
        return {
            "id": req_id,
            "src": DEVICE_ID,
            "dst": "unknown",
            "result": {
                "a_act_power": a,
                "b_act_power": b,
                "c_act_power": c,
                "total_act_power": total,
            },
        }
    elif method == "EM1.GetStatus":
        with VM.lock:
            raw_powers = (
                float(VM.phases["a"].act_power or 0.0),
                float(VM.phases["b"].act_power or 0.0),
                float(VM.phases["c"].act_power or 0.0),
            )
        scaled = _apply_request_side_power_scaling(raw_powers)
        total = round(sum(scaled), 3)
        if total == round(total) or total == 0:
            total = total + 0.001
        return {
            "id": req_id,
            "src": DEVICE_ID,
            "dst": "unknown",
            "result": {"act_power": total},
        }
    else:
        # Unknown methods: silently ignore (b2500-meter behavior)
        return None


class RPCUDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr):
        # Enforce max payload; ignore oversize like b2500-meter (no JSON-RPC errors)
        if len(data) > self.max_size:
            return
        try:
            host = addr[0] if isinstance(addr, tuple) else None
        except Exception:
            host = None
        _register_request_ip(host)
        try:
            text = data.decode("utf-8", errors="ignore")
            obj = json.loads(text)
        except Exception:
            return
        try:
            m = obj.get("method") if isinstance(obj, dict) else None
            log.debug("UDP packet from %s method=%s", addr, m)
            global UDP_PACKETS_TOTAL
            UDP_PACKETS_TOTAL += 1
            try:
                _add_recent_udp(addr[0] if isinstance(addr, tuple) else str(addr), m)
            except Exception:
                pass
        except Exception:
            pass
        # If payload is JSON-RPC 2.0, respond with JSON-RPC envelope
        if isinstance(obj, (dict, list)) and (
            (isinstance(obj, dict) and obj.get("jsonrpc") == "2.0") or
            (isinstance(obj, list) and any(isinstance(x, dict) and x.get("jsonrpc") == "2.0" for x in obj))
        ):
            resp = _rpc_response_bytes(text.encode("utf-8"))
            try:
                log.debug("UDP JSON-RPC response: %s", resp.decode("utf-8", errors="ignore"))
            except Exception:
                pass
            self.transport.sendto(resp, addr)
            return
        resp_obj = _udp_build_response(obj)
        if resp_obj is None:
            return
        # Compact separators to match b2500-meter
        payload = json.dumps(resp_obj, separators=(",", ":")).encode()
        try:
            log.debug("UDP Shelly-style response: %s", payload.decode("utf-8", errors="ignore"))
        except Exception:
            pass
        self.transport.sendto(payload, addr)
        try:
            log.debug("UDP reply to %s method=%s bytes=%d", addr, obj.get("method"), len(payload))
            global UDP_REPLIES_TOTAL
            UDP_REPLIES_TOTAL += 1
        except Exception:
            pass

async def start_udp_servers(ports: List[int], max_size: int):
    loop = asyncio.get_running_loop()
    transports = []
    for p in ports:
        transport, _ = await loop.create_datagram_endpoint(
            lambda: RPCUDPProtocol(max_size),
            local_addr=("0.0.0.0", p),
            allow_broadcast=True,
            reuse_port=True
        )
        transports.append(transport)
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        for t in transports:
            t.close()

def _start_udp():
    # Build ports list
    ports: List[int] = []
    for token in UDP_PORTS.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            ports.append(int(token))
        except ValueError:
            pass
    if not ports:
        ports = [1010]

    def on_packet(addr, method_name):
        try:
            log.debug("UDP packet from %s method=%s", addr, method_name)
            MET.UDP_PACKETS_TOTAL += 1
            ip = addr[0] if isinstance(addr, tuple) else str(addr)
            MET.add_recent_udp(now_ts(), ip, method_name)
        except Exception:
            pass

    def on_reply(_nbytes: int):
        try:
            MET.UDP_REPLIES_TOTAL += 1
        except Exception:
            pass

    rpc_bytes = lambda data: RPC.rpc_response_bytes(METHODS, data, log)
    udp_builder = lambda obj: RPC.udp_build_response(VM, DEVICE_ID, obj)
    UDP.start_udp_thread(ports, UDP_MAX, METHODS, rpc_bytes, udp_builder, on_packet, on_reply)

_start_udp()

# -----------------------------
# mDNS
# -----------------------------
def _get_ip_for_mdns() -> str:
    if MDNS_IP:
        return MDNS_IP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def start_mdns():
    if not MDNS_ENABLE:
        return
    ip = _get_ip_for_mdns()
    addr = socket.inet_aton(ip)
    instance = MDNS_HOSTNAME
    try:
        zc = Zeroconf(interfaces=[ip] if MDNS_IP else InterfaceChoice.Default)
    except Exception:
        zc = Zeroconf(interfaces=InterfaceChoice.Default)

    txt = {
        "gen": str(GENERATION),
        "id": DEVICE_ID,
        "app": APP_ID,
        "fw_id": FW_ID,
        "model": MODEL,
        "ver": FIRMWARE,
        "mac": MAC,
    }
    info_http = ServiceInfo(
        "_http._tcp.local.",
        f"{instance}._http._tcp.local.",
        addresses=[addr],
        port=HTTP_PORT,
        properties={k.encode(): v.encode() for k, v in txt.items()},
        server=f"{instance}.local.",
    )
    info_shelly = ServiceInfo(
        "_shelly._tcp.local.",
        f"{instance}._shelly._tcp.local.",
        addresses=[addr],
        port=HTTP_PORT,
        properties={k.encode(): v.encode() for k, v in txt.items()},
        server=f"{instance}.local.",
    )
    info_modbus = None
    if MODBUS_ENABLE and StartAsyncTcpServer is not None:
        info_modbus = ServiceInfo(
            "_modbus._tcp.local.",
            f"{instance}._modbus._tcp.local.",
            addresses=[addr],
            port=MODBUS_PORT,
            properties={k.encode(): v.encode() for k, v in txt.items()},
            server=f"{instance}.local.",
        )

    def _register():
        # Register with explicit TTL (seconds)
        try:
            zc.register_service(info_http, ttl=120)
            zc.register_service(info_shelly, ttl=120)
            if info_modbus is not None:
                zc.register_service(info_modbus, ttl=120)
        except TypeError:
            # Older zeroconf without ttl param
            zc.register_service(info_http)
            zc.register_service(info_shelly)
            if info_modbus is not None:
                zc.register_service(info_modbus)
        current_ip = ip
        while True:
            time.sleep(60)
            try:
                new_ip = _get_ip_for_mdns()
                if new_ip != current_ip:
                    new_addr = socket.inet_aton(new_ip)
                    info_http.addresses = [new_addr]
                    info_shelly.addresses = [new_addr]
                    if info_modbus is not None:
                        info_modbus.addresses = [new_addr]
                    try:
                        zc.update_service(info_http)
                        zc.update_service(info_shelly)
                        if info_modbus is not None:
                            zc.update_service(info_modbus)
                    except Exception:
                        # Fallback: unregister and register again
                        try:
                            zc.unregister_service(info_http)
                            zc.unregister_service(info_shelly)
                            if info_modbus is not None:
                                zc.unregister_service(info_modbus)
                        except Exception:
                            pass
                        try:
                            zc.register_service(info_http)
                            zc.register_service(info_shelly)
                            if info_modbus is not None:
                                zc.register_service(info_modbus)
                        except Exception:
                            pass
                    current_ip = new_ip
            except Exception:
                pass
    threading.Thread(target=_register, daemon=True).start()

start_mdns()

# -----------------------------
# Graceful shutdown: persist energy on SIGTERM/SIGINT
# -----------------------------
def _handle_term(signum, frame):
    try:
        VM.persist()
    except Exception:
        pass

try:
    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)
except Exception:
    pass
