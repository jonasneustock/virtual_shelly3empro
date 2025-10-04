from __future__ import annotations

import json
from typing import Dict, Any, Callable


def build_methods(vm) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    methods = {
        "Shelly.GetStatus": vm.shelly_get_status,
        "Shelly.GetDeviceInfo": vm.shelly_get_device_info,
        "Shelly.GetConfig": vm.shelly_get_config,
        "Shelly.SetConfig": vm.shelly_set_config,
        "Shelly.Ping": vm.shelly_ping,
        "Shelly.Reboot": vm.shelly_reboot,
        "EM.GetStatus": vm.em_get_status,
        "EMData.GetStatus": vm.emdata_get_status,
        "EMData.ResetCounters": vm.emdata_reset_counters,
        "MQTT.GetConfig": vm.mqtt_get_config,
        "MQTT.SetConfig": vm.mqtt_set_config,
        "MQTT.Status": vm.mqtt_status,
        "Sys.GetStatus": vm.sys_get_status,
        "Sys.GetInfo": vm.sys_get_info,
        "Sys.SetConfig": vm.sys_set_config,
    }
    # Fill list methods dynamically
    methods["Shelly.ListMethods"] = lambda _params: sorted(methods.keys())
    return methods


def handle_rpc_call(methods: Dict[str, Callable[[Dict[str, Any]], Any]], req: Dict[str, Any], log=None) -> Dict[str, Any]:
    rid = req.get("id")
    method = req.get("method")
    params = req.get("params", {}) or {}
    if method not in methods:
        # Permissive dummy response for unknown methods
        return {"jsonrpc": "2.0", "id": rid, "result": {"ok": True, "dummy": True, "method": method}}
    try:
        result = methods[method](params)
        try:
            if log:
                log.debug("RPC %s response: %s", method, result)
        except Exception:
            pass
        return {"jsonrpc": "2.0", "id": rid, "result": result}
    except Exception as e:
        try:
            if log:
                log.debug("RPC %s error: %s", method, e)
        except Exception:
            pass
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": str(e)}}


def rpc_response_bytes(methods: Dict[str, Callable[[Dict[str, Any]], Any]], data: bytes, log=None) -> bytes:
    try:
        obj = json.loads(data.decode("utf-8", errors="ignore"))
    except Exception:
        return json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": "parse error"}}).encode()
    if isinstance(obj, list):
        resp = [handle_rpc_call(methods, x, log) for x in obj]
    else:
        resp = handle_rpc_call(methods, obj, log)
    return json.dumps(resp).encode()


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


def udp_build_response(vm, device_id: str, obj: Dict[str, Any]) -> Dict[str, Any] | None:
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
        with vm.lock:
            powers = [
                vm.phases["a"].act_power or 0.0,
                vm.phases["b"].act_power or 0.0,
                vm.phases["c"].act_power or 0.0,
            ]
        a = _udp_decimal_enforcer(powers[0])
        b = _udp_decimal_enforcer(powers[1])
        c = _udp_decimal_enforcer(powers[2])
        total = round(sum(powers), 3)
        if total == round(total) or total == 0:
            total = total + 0.001
        return {
            "id": req_id,
            "src": device_id,
            "dst": "unknown",
            "result": {
                "a_act_power": a,
                "b_act_power": b,
                "c_act_power": c,
                "total_act_power": total,
            },
        }
    elif method == "EM1.GetStatus":
        with vm.lock:
            powers = [
                vm.phases["a"].act_power or 0.0,
                vm.phases["b"].act_power or 0.0,
                vm.phases["c"].act_power or 0.0,
            ]
        total = round(sum(powers), 3)
        if total == round(total) or total == 0:
            total = total + 0.001
        return {
            "id": req_id,
            "src": device_id,
            "dst": "unknown",
            "result": {"act_power": total},
        }
    else:
        # Unknown methods: silently ignore (b2500-meter behavior)
        return None

