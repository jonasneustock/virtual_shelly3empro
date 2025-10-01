import os
import time
import json
import threading
import socket
import asyncio
import logging
import struct
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import requests
from fastapi import FastAPI, Request, WebSocket
from starlette.responses import JSONResponse

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
        return float(raw)
    except Exception:
        return None

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
        self.last_poll = None
        self.phases = {"a": Phase(), "b": Phase(), "c": Phase()}
        self.frequency = 50.0
        self.energy = EnergyCounters()
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

    def persist(self):
        with self.lock:
            save_state({"energy": json.loads(self.energy.json())})

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

    def poll_home_assistant(self):
        try:
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

            now = now_ts()
            if self.last_poll is not None:
                dt = max(0.0, now - self.last_poll)
                if dt > 0:
                    self.integrate_energy(dt)
            self.last_poll = now

            if now % 30 == 0:
                self.persist()
        except Exception:
            pass

    # ---------- RPC builders ----------
    def em_get_status(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            a, b, c = self.phases["a"], self.phases["b"], self.phases["c"]
            total_w = (a.act_power or 0.0) + (b.act_power or 0.0) + (c.act_power or 0.0)

            if STRICT_MINIMAL_PAYLOAD:
                # Minimal contract: only the 4 power keys some gateways require
                return {
                    "a_act_power": round(a.act_power, 2),
                    "b_act_power": round(b.act_power, 2),
                    "c_act_power": round(c.act_power, 2),
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
                "a_act_power": round(a.act_power, 2),
                "b_act_power": round(b.act_power, 2),
                "c_act_power": round(c.act_power, 2),
                "a_pf": a.pf if a.pf is not None else 1.0,
                "b_pf": b.pf if b.pf is not None else 1.0,
                "c_pf": c.pf if c.pf is not None else 1.0,
                "frequency": self.frequency,
                "total_act_power": round(total_w, 2),
                "ts": now_ts(),
            }

    def emdata_get_status(self, _params: Dict[str, Any]) -> Dict[str, Any]:
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
        return {
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
            "em:0": self.em_get_status({}),
            "emdata:0": self.emdata_get_status({}),
        }

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
    if StartAsyncTcpServer is None:
        MODBUS_LOG.warning("pymodbus not available; Modbus TCP disabled")
        return

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
app = FastAPI(title="Virtual Shelly Pro 3EM RPC")
VM = VirtualPro3EM()
MODBUS_BRIDGE: Optional[ShellyModbusBridge] = None
if MODBUS_ENABLE:
    try:
        MODBUS_BRIDGE = ShellyModbusBridge(VM)
        start_modbus_server(MODBUS_BRIDGE)
    except Exception as exc:  # pragma: no cover - background log only
        MODBUS_LOG.error("Failed to start Modbus bridge: %s", exc)
        MODBUS_BRIDGE = None

METHODS = {
    "Shelly.GetStatus": VM.shelly_get_status,
    "Shelly.GetDeviceInfo": VM.shelly_get_device_info,
    "Shelly.GetConfig": VM.shelly_get_config,
    "Shelly.SetConfig": VM.shelly_set_config,
    "Shelly.Ping": VM.shelly_ping,
    "Shelly.Reboot": VM.shelly_reboot,
    "EM.GetStatus": VM.em_get_status,
    "EMData.GetStatus": VM.emdata_get_status,
    "EMData.ResetCounters": VM.emdata_reset_counters,
    "MQTT.GetConfig": VM.mqtt_get_config,
    "MQTT.SetConfig": VM.mqtt_set_config,
    "Sys.GetStatus": VM.sys_get_status,
}

def _handle_rpc_call(req: Dict[str, Any]) -> Dict[str, Any]:
    rid = req.get("id")
    method = req.get("method")
    params = req.get("params", {}) or {}
    if method not in METHODS:
        # Permissive dummy response for unknown methods
        return {"jsonrpc": "2.0", "id": rid, "result": {"ok": True, "dummy": True, "method": method}}
    try:
        result = METHODS[method](params)
        return {"jsonrpc": "2.0", "id": rid, "result": result}
    except Exception as e:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": str(e)}}

def _rpc_response_bytes(data: bytes) -> bytes:
    try:
        obj = json.loads(data.decode("utf-8", errors="ignore"))
    except Exception:
        return json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": "parse error"}}).encode()
    if isinstance(obj, list):
        resp = [_handle_rpc_call(x) for x in obj]
    else:
        resp = _handle_rpc_call(obj)
    return json.dumps(resp).encode()

# -----------------------------
# HTTP /rpc
# -----------------------------
@app.post("/rpc")
async def rpc_endpoint(request: Request) -> JSONResponse:
    body = await request.json()
    if isinstance(body, list):
        resp = [_handle_rpc_call(item) for item in body]
        return JSONResponse(resp)
    else:
        return JSONResponse(_handle_rpc_call(body))

@app.get("/rpc")
async def rpc_get(request: Request) -> JSONResponse:
    # Support GET-style RPC: /rpc?method=EM.GetStatus&id=0 or /rpc?method=Shelly.GetConfig
    qp = dict(request.query_params)
    method = qp.pop("method", None)
    if not method:
        return JSONResponse({"error": "method required"}, status_code=400)
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
    qp = dict(request.query_params)
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

@app.get("/")
def root():
    return {"service": "virtual-shelly-pro-3em", "rpc": "/rpc"}

@app.get("/shelly")
def shelly_http_info():
    # Shelly Gen2-compatible device info endpoint
    return VM.build_device_info()

# -----------------------------
# Background: HA poller
# -----------------------------
def poll_loop():
    while True:
        VM.poll_home_assistant()
        if MODBUS_BRIDGE:
            MODBUS_BRIDGE.update()
        time.sleep(POLL_INTERVAL)
threading.Thread(target=poll_loop, daemon=True).start()

# -----------------------------
# WebSockets 6010–6022
# -----------------------------
async def ws_handler(websocket):
    async for message in websocket:
        # websockets expects str; our dispatcher returns bytes; convert.
        resp = _rpc_response_bytes(message.encode()).decode()
        await websocket.send(resp)

@app.websocket("/rpc")
async def ws_rpc(websocket: WebSocket):
    # WebSocket endpoint compatible with Shelly Gen2 ws://<ip>/rpc
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            resp = _rpc_response_bytes(message.encode()).decode()
            await websocket.send_text(resp)
    except Exception:
        try:
            await websocket.close()
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
            powers = [
                VM.phases["a"].act_power or 0.0,
                VM.phases["b"].act_power or 0.0,
                VM.phases["c"].act_power or 0.0,
            ]
        a = _udp_decimal_enforcer(powers[0])
        b = _udp_decimal_enforcer(powers[1])
        c = _udp_decimal_enforcer(powers[2])
        total = round(sum(powers), 3)
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
            powers = [
                VM.phases["a"].act_power or 0.0,
                VM.phases["b"].act_power or 0.0,
                VM.phases["c"].act_power or 0.0,
            ]
        total = round(sum(powers), 3)
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
            obj = json.loads(data.decode("utf-8", errors="ignore"))
        except Exception:
            return
        resp_obj = _udp_build_response(obj)
        if resp_obj is None:
            return
        # Compact separators to match b2500-meter
        payload = json.dumps(resp_obj, separators=(",", ":")).encode()
        self.transport.sendto(payload, addr)

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

def udp_thread():
    ports = []
    for token in UDP_PORTS.split(","):
        token = token.strip()
        if token:
            try:
                ports.append(int(token))
            except ValueError:
                pass
    if not ports:
        ports = [1010]
    asyncio.run(start_udp_servers(ports, UDP_MAX))
threading.Thread(target=udp_thread, daemon=True).start()

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
    zc = Zeroconf(interfaces=InterfaceChoice.All)

    txt = {
        "gen": str(GENERATION),
        "id": DEVICE_ID,
        "app": APP_ID,
        "fw_id": FW_ID,
        "model": MODEL,
        "ver": FIRMWARE,
        "mac": MAC,
    }
    services = [
        ServiceInfo(
            "_http._tcp.local.",
            f"{instance}._http._tcp.local.",
            addresses=[addr],
            port=HTTP_PORT,
            properties={k.encode(): v.encode() for k, v in txt.items()},
            server=f"{instance}.local.",
        ),
        ServiceInfo(
            "_shelly._tcp.local.",
            f"{instance}._shelly._tcp.local.",
            addresses=[addr],
            port=HTTP_PORT,
            properties={k.encode(): v.encode() for k, v in txt.items()},
            server=f"{instance}.local.",
        ),
    ]

    if MODBUS_ENABLE:
        services.append(
            ServiceInfo(
                "_modbus._tcp.local.",
                f"{instance}._modbus._tcp.local.",
                addresses=[addr],
                port=MODBUS_PORT,
                properties={k.encode(): v.encode() for k, v in txt.items()},
                server=f"{instance}.local.",
            )
        )

    def _register():
        for svc in services:
            zc.register_service(svc)
        while True:
            # Zeroconf needs the thread alive so the service stays announced; sleep long.
            time.sleep(3600)
    threading.Thread(target=_register, daemon=True).start()

start_mdns()
