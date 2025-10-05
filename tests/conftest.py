import os
import sys
from pathlib import Path
import importlib
import types
import pytest


@pytest.fixture(scope="session")
def app_module():
    # Ensure repository root on path for `virtual_shelly` imports
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.environ.setdefault("DISABLE_BACKGROUND", "1")
    os.environ.setdefault("MDNS_ENABLE", "false")
    os.environ.setdefault("MODBUS_ENABLE", "false")
    os.environ.setdefault("STATE_PATH", "/tmp/virtual_shelly_state_test.json")
    os.environ["BASIC_AUTH_ENABLE"] = "false"
    # Ensure clean import each session
    if "virtual_shelly.metrics" in importlib.sys.modules:
        importlib.reload(importlib.import_module("virtual_shelly.metrics"))
    m = importlib.import_module("app")
    return m


@pytest.fixture(autouse=True)
def reset_metrics():
    # Reset counters before each test for reproducibility
    import virtual_shelly.metrics as MET
    MET.HTTP_RPC_METHODS.clear()
    MET.HTTP_RPC_ERRORS.clear()
    MET.WS_NOTIFY_TOTAL = 0
    MET.WS_RPC_MESSAGES = 0
    MET.UDP_PACKETS_TOTAL = 0
    MET.UDP_REPLIES_TOTAL = 0
    with MET.RECENT_LOCK:
        MET.RECENT_HTTP.clear()
        MET.RECENT_WS.clear()
        MET.RECENT_UDP.clear()
    yield


@pytest.fixture()
def client(app_module):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)
