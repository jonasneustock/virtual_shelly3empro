from __future__ import annotations

import os
import time
import threading
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque


# Counters
HTTP_RPC_METHODS = defaultdict(int)
HTTP_RPC_ERRORS = defaultdict(int)
WS_NOTIFY_TOTAL = 0
WS_RPC_MESSAGES = 0
UDP_PACKETS_TOTAL = 0
UDP_REPLIES_TOTAL = 0

# Latency histograms (seconds) for HTTP /rpc
RPC_LATENCY_BUCKETS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
RPC_LATENCY_HIST = defaultdict(int)  # bucket upper bound -> count
RPC_LATENCY_SUM = 0.0
RPC_LATENCY_COUNT = 0


# Recent client tracking (for UI)
RECENT_MAX = int(os.getenv("RECENT_CLIENTS_MAX", "100"))
RECENT_HTTP = deque(maxlen=RECENT_MAX)  # items: {ts, ip, method, verb}
RECENT_WS = deque(maxlen=RECENT_MAX)    # items: {ts, ip, event, method?}
RECENT_UDP = deque(maxlen=RECENT_MAX)   # items: {ts, ip, method}
RECENT_LOCK = threading.RLock()


def add_recent_http(ts: int, ip: str, method: Optional[str], verb: str) -> None:
    try:
        with RECENT_LOCK:
            RECENT_HTTP.append({
                "ts": ts,
                "ip": ip or "?",
                "method": method or "?",
                "verb": verb,
            })
    except Exception:
        pass


def add_recent_ws(ts: int, ip: str, event: str, method: Optional[str] = None) -> None:
    try:
        with RECENT_LOCK:
            entry = {"ts": ts, "ip": ip or "?", "event": event}
            if method:
                entry["method"] = method
            RECENT_WS.append(entry)
    except Exception:
        pass


def add_recent_udp(ts: int, ip: str, method: Optional[str]) -> None:
    try:
        with RECENT_LOCK:
            RECENT_UDP.append({
                "ts": ts,
                "ip": ip or "?",
                "method": method or "?",
            })
    except Exception:
        pass


def metrics_text(ws_clients_count: int) -> str:
    lines: List[str] = []
    # HTTP
    total_http = sum(HTTP_RPC_METHODS.values())
    lines.append("# HELP virtual_shelly_http_rpc_requests_total Total HTTP RPC requests")
    lines.append("# TYPE virtual_shelly_http_rpc_requests_total counter")
    lines.append(f"virtual_shelly_http_rpc_requests_total {total_http}")
    for m, v in HTTP_RPC_METHODS.items():
        lines.append(f'virtual_shelly_http_rpc_requests_total{{method="{m}"}} {v}')
    lines.append("# HELP virtual_shelly_http_rpc_errors_total Total HTTP RPC errors")
    lines.append("# TYPE virtual_shelly_http_rpc_errors_total counter")
    for m, v in HTTP_RPC_ERRORS.items():
        lines.append(f'virtual_shelly_http_rpc_errors_total{{method="{m}"}} {v}')
    # WS
    lines.append("# HELP virtual_shelly_ws_rpc_messages_total Total WS RPC messages received")
    lines.append("# TYPE virtual_shelly_ws_rpc_messages_total counter")
    lines.append(f"virtual_shelly_ws_rpc_messages_total {WS_RPC_MESSAGES}")
    lines.append("# HELP virtual_shelly_ws_notify_total Total WS Notify messages sent")
    lines.append("# TYPE virtual_shelly_ws_notify_total counter")
    lines.append(f"virtual_shelly_ws_notify_total {WS_NOTIFY_TOTAL}")
    lines.append("# HELP virtual_shelly_ws_clients Current WS client connections")
    lines.append("# TYPE virtual_shelly_ws_clients gauge")
    lines.append(f"virtual_shelly_ws_clients {ws_clients_count}")
    # UDP
    lines.append("# HELP virtual_shelly_udp_packets_total Total UDP packets received")
    lines.append("# TYPE virtual_shelly_udp_packets_total counter")
    lines.append(f"virtual_shelly_udp_packets_total {UDP_PACKETS_TOTAL}")
    lines.append("# HELP virtual_shelly_udp_replies_total Total UDP replies sent")
    lines.append("# TYPE virtual_shelly_udp_replies_total counter")
    lines.append(f"virtual_shelly_udp_replies_total {UDP_REPLIES_TOTAL}")
    # HTTP /rpc latency histogram (simple text format)
    lines.append("# HELP virtual_shelly_http_rpc_latency_seconds Request latency for HTTP /rpc")
    lines.append("# TYPE virtual_shelly_http_rpc_latency_seconds histogram")
    cumulative = 0
    for b in sorted(RPC_LATENCY_BUCKETS):
        cumulative += RPC_LATENCY_HIST[b]
        lines.append(f'virtual_shelly_http_rpc_latency_seconds_bucket{{le="{b}"}} {cumulative}')
    # +Inf bucket
    lines.append(f'virtual_shelly_http_rpc_latency_seconds_bucket{{le="+Inf"}} {RPC_LATENCY_COUNT}')
    lines.append(f"virtual_shelly_http_rpc_latency_seconds_sum {RPC_LATENCY_SUM}")
    lines.append(f"virtual_shelly_http_rpc_latency_seconds_count {RPC_LATENCY_COUNT}")
    return "\n".join(lines)


def build_admin_overview(vm, ws_connected_ips: List[str], now_ts_fn) -> Dict[str, Any]:
    try:
        device = vm.build_device_info()
    except Exception:
        device = {"id": getattr(vm, "DEVICE_ID", None) or "device", "model": None, "ver": None}
    try:
        em = vm.em_get_status({"id": 0})
    except Exception:
        em = {}
    try:
        emdata = vm.emdata_get_status({"id": 0})
    except Exception:
        emdata = {}
    with RECENT_LOCK:
        http_recent = list(RECENT_HTTP)[-20:]
        ws_recent = list(RECENT_WS)[-20:]
        udp_recent = list(RECENT_UDP)[-20:]
        http_unique = sorted({x.get("ip") for x in RECENT_HTTP if isinstance(x, dict) and x.get("ip")})
        ws_unique = sorted({x.get("ip") for x in RECENT_WS if isinstance(x, dict) and x.get("ip")})
        udp_unique = sorted({x.get("ip") for x in RECENT_UDP if isinstance(x, dict) and x.get("ip")})
    data = {
        "device": {k: device.get(k) for k in ("id", "model", "ver", "app") if k in device},
        "values": {"em": em, "emdata": emdata},
        "metrics": {
            "http": {
                "total": int(sum(HTTP_RPC_METHODS.values())),
                "by_method": dict(HTTP_RPC_METHODS),
                "errors": dict(HTTP_RPC_ERRORS),
            },
            "ws": {
                "rpc_messages": int(WS_RPC_MESSAGES),
                "notify_total": int(WS_NOTIFY_TOTAL),
                "clients": int(len(ws_connected_ips)),
            },
            "udp": {
                "packets": int(UDP_PACKETS_TOTAL),
                "replies": int(UDP_REPLIES_TOTAL),
            },
        },
        "upstream": {"ha_connected": bool(getattr(vm, 'ha_connected', True)), "ha_failures": int(getattr(vm, 'ha_failures', 0))},
        "clients": {
            "http_recent": http_recent,
            "ws_recent": ws_recent,
            "udp_recent": udp_recent,
            "ws_connected": ws_connected_ips,
            "http_unique": http_unique,
            "ws_unique": ws_unique,
            "udp_unique": udp_unique,
        },
        "ts": int(now_ts_fn()),
    }
    return data
