from __future__ import annotations

import os
import re
from typing import List, Optional


def _to_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "on")


class Settings:
    # HA & polling
    HA_BASE_URL: str = os.getenv("HA_BASE_URL", "http://homeassistant:8123")
    HA_TOKEN: str = os.getenv("HA_TOKEN", "")
    POLL_INTERVAL: float = float(os.getenv("POLL_INTERVAL", "2.0"))
    HA_SMOOTHING_ENABLE: bool = _to_bool("HA_SMOOTHING_ENABLE", False)
    HA_SMOOTHING_WINDOW: int = int(os.getenv("HA_SMOOTHING_WINDOW", "5"))

    # Entities
    A_POWER: str = os.getenv("A_POWER", "sensor.phase_a_power")
    B_POWER: str = os.getenv("B_POWER", "sensor.phase_b_power")
    C_POWER: str = os.getenv("C_POWER", "sensor.phase_c_power")
    A_VOLT: str = os.getenv("A_VOLT", "sensor.phase_a_voltage")
    B_VOLT: str = os.getenv("B_VOLT", "sensor.phase_b_voltage")
    C_VOLT: str = os.getenv("C_VOLT", "sensor.phase_c_voltage")
    A_CURR: str = os.getenv("A_CURR", "sensor.phase_a_current")
    B_CURR: str = os.getenv("B_CURR", "sensor.phase_b_current")
    C_CURR: str = os.getenv("C_CURR", "sensor.phase_c_current")
    A_PF: str = os.getenv("A_PF", "sensor.phase_a_pf")
    B_PF: str = os.getenv("B_PF", "sensor.phase_b_pf")
    C_PF: str = os.getenv("C_PF", "sensor.phase_c_pf")

    # Identity
    DEVICE_ID: str = os.getenv("DEVICE_ID", "shellypro3em-virtual-001")
    APP_ID: str = os.getenv("APP_ID", os.getenv("APP", "shellypro3em"))
    MODEL: str = os.getenv("MODEL", "SHPRO-3EM")
    FIRMWARE: str = os.getenv("FIRMWARE", "1.0.0-virt")
    FW_ID: str = os.getenv("FW_ID", FIRMWARE)
    MAC: str = os.getenv("MAC", "AA:BB:CC:DD:EE:FF")
    SN: str = os.getenv("SN", "VIRT3EM001")
    MANUFACTURER: str = os.getenv("MANUFACTURER", "Allterco Robotics")
    GENERATION: int = int(os.getenv("GENERATION", "2"))

    STATE_PATH: str = os.getenv("STATE_PATH", "/data/state.json")

    # HTTP/WS
    HTTP_PORT: int = int(os.getenv("HTTP_PORT", "8080"))
    WS_PORT_START: int = int(os.getenv("WS_PORT_START", "6010"))
    WS_PORT_END: int = int(os.getenv("WS_PORT_END", "6022"))
    WS_NOTIFY_INTERVAL: float = float(os.getenv("WS_NOTIFY_INTERVAL", "2.0"))
    WS_NOTIFY_EPS: float = float(os.getenv("WS_NOTIFY_EPS", "0.1"))
    CORS_ENABLE: bool = _to_bool("CORS_ENABLE", False)
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")

    # UDP
    UDP_PORTS: str = os.getenv("UDP_PORTS", "1010,2220")
    UDP_MAX: int = int(os.getenv("UDP_MAX", "32768"))

    # mDNS
    MDNS_ENABLE: bool = _to_bool("MDNS_ENABLE", True)
    MDNS_HOSTNAME: Optional[str] = os.getenv("MDNS_HOSTNAME")
    MDNS_IP: Optional[str] = os.getenv("MDNS_IP")

    # Modbus
    MODBUS_ENABLE: bool = _to_bool("MODBUS_ENABLE", True)
    MODBUS_PORT: int = int(os.getenv("MODBUS_PORT", "502"))
    MODBUS_BIND: str = os.getenv("MODBUS_BIND", "0.0.0.0")
    MODBUS_UNIT_ID: int = int(os.getenv("MODBUS_UNIT_ID", "1"))

    # Runtime payload/scaling
    STRICT_MINIMAL_PAYLOAD: bool = _to_bool("STRICT_MINIMAL_PAYLOAD", False)
    REQUEST_SIDE_SCALING_ENABLE: bool = _to_bool("REQUEST_SIDE_SCALING_ENABLE", True)
    REQUEST_SIDE_SCALING_CLIENTS: int = int(os.getenv("REQUEST_SIDE_SCALING_CLIENTS", "0"))
    REQUEST_IP_TTL: float = float(os.getenv("REQUEST_IP_TTL", "30.0"))

    # Background and auth
    DISABLE_BACKGROUND: bool = _to_bool("DISABLE_BACKGROUND", False)
    BASIC_AUTH_ENABLE: bool = _to_bool("BASIC_AUTH_ENABLE", False)
    BASIC_AUTH_USER: Optional[str] = os.getenv("BASIC_AUTH_USER")
    BASIC_AUTH_PASSWORD: Optional[str] = os.getenv("BASIC_AUTH_PASSWORD")

    # Rate limiting
    RATE_LIMIT_ENABLE: bool = _to_bool("RATE_LIMIT_ENABLE", False)
    RATE_LIMIT_RPCS_PER_10S: int = int(os.getenv("RATE_LIMIT_RPCS_PER_10S", "120"))


def load_settings() -> Settings:
    # Optional .env support if python-dotenv is installed
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()  # load from .env if present
    except Exception:
        pass
    return Settings()


def validate_settings(s: Settings) -> List[str]:
    errors: List[str] = []
    # URLs
    if not re.match(r"^https?://", s.HA_BASE_URL or ""):
        errors.append(f"HA_BASE_URL must start with http:// or https:// (got {s.HA_BASE_URL!r})")
    # Ports
    def vport(p: int, name: str):
        if not (1 <= int(p) <= 65535):
            errors.append(f"{name} must be between 1 and 65535 (got {p})")
    vport(s.HTTP_PORT, "HTTP_PORT")
    vport(s.WS_PORT_START, "WS_PORT_START")
    vport(s.WS_PORT_END, "WS_PORT_END")
    if int(s.WS_PORT_START) > int(s.WS_PORT_END):
        errors.append("WS_PORT_START must be <= WS_PORT_END")
    vport(s.MODBUS_PORT, "MODBUS_PORT")
    # UDP ports
    for token in (s.UDP_PORTS or "").split(','):
        token = token.strip()
        if not token:
            continue
        try:
            vport(int(token), f"UDP_PORTS:{token}")
        except Exception:
            errors.append(f"UDP_PORTS contains invalid port {token!r}")
    # Smoothing
    if int(s.HA_SMOOTHING_WINDOW) < 1:
        errors.append("HA_SMOOTHING_WINDOW must be >= 1")
    if float(s.POLL_INTERVAL) <= 0:
        errors.append("POLL_INTERVAL must be > 0")
    # Auth
    if s.BASIC_AUTH_ENABLE and (not s.BASIC_AUTH_USER or not s.BASIC_AUTH_PASSWORD):
        errors.append("BASIC_AUTH_ENABLE=true requires BASIC_AUTH_USER and BASIC_AUTH_PASSWORD")
    return errors

