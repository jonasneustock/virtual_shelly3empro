import json


def test_rpc_get_query_and_path(client):
    # /rpc?method=EM.GetStatus&id=0
    r = client.get("/rpc", params={"method": "EM.GetStatus", "id": 0})
    assert r.status_code == 200
    data = r.json()
    assert "a_act_power" in data
    # /rpc/EM.GetStatus?id=0
    r2 = client.get("/rpc/EM.GetStatus", params={"id": 0})
    assert r2.status_code == 200
    data2 = r2.json()
    assert "b_act_power" in data2


def test_rpc_post_jsonrpc_single_and_batch(client):
    # Single
    payload = {"id": 1, "method": "Shelly.Ping", "params": {}}
    r = client.post("/rpc", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["result"]["pong"] is True
    # Batch
    batch = [
        {"id": 10, "method": "EM.GetStatus", "params": {"id": 0}},
        {"id": 11, "method": "EMData.GetStatus", "params": {"id": 0}},
    ]
    r2 = client.post("/rpc", json=batch)
    assert r2.status_code == 200
    arr = r2.json()
    assert isinstance(arr, list) and len(arr) == 2
    assert arr[0]["result"]["id"] == 0


def test_rpc_get_batch_param(client):
    # GET /rpc?batch=[{...},{...}]
    batch = json.dumps([
        {"id": None, "method": "EM.GetStatus", "params": {"id": 0}},
        {"id": None, "method": "Shelly.GetDeviceInfo", "params": {}},
    ])
    r = client.get("/rpc", params={"batch": batch})
    assert r.status_code == 200
    arr = r.json()
    assert isinstance(arr, list) and len(arr) == 2
    assert arr[1]["id"] == "shellypro3em-virtual-001" or "id" in arr[1]


def test_rpc_get_missing_method_400(client):
    r = client.get("/rpc")
    assert r.status_code == 400
    data = r.json()
    assert data["error"] == "method required"
