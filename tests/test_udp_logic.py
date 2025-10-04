import json


class DummyPhase:
    def __init__(self):
        self.act_power = 0.0


class DummyVM:
    def __init__(self):
        import threading
        self.lock = threading.RLock()
        self.phases = {"a": DummyPhase(), "b": DummyPhase(), "c": DummyPhase()}


def test_udp_build_response_em_get_status():
    from virtual_shelly import rpc_core as RPC
    vm = DummyVM()
    vm.phases["a"].act_power = 123.4
    vm.phases["b"].act_power = 56.7
    vm.phases["c"].act_power = -10.0
    req = {"id": 1, "src": "cli", "method": "EM.GetStatus", "params": {"id": 0}}
    resp = RPC.udp_build_response(vm, "device-id", req)
    assert resp["id"] == 1
    result = resp["result"]
    assert "a_act_power" in result and "total_act_power" in result


def test_udp_build_response_em1_get_status():
    from virtual_shelly import rpc_core as RPC
    vm = DummyVM()
    vm.phases["a"].act_power = 1.0
    vm.phases["b"].act_power = 2.0
    vm.phases["c"].act_power = 3.0
    req = {"id": 2, "src": "cli", "method": "EM1.GetStatus", "params": {"id": 0}}
    resp = RPC.udp_build_response(vm, "device-id", req)
    assert resp["id"] == 2
    assert "act_power" in resp["result"]


def test_udp_build_response_requires_int_id_param():
    from virtual_shelly import rpc_core as RPC
    vm = DummyVM()
    req = {"id": 3, "src": "cli", "method": "EM.GetStatus", "params": {"id": "0"}}
    assert RPC.udp_build_response(vm, "device-id", req) is None
