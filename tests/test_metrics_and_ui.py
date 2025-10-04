def test_metrics_endpoint_text(client):
    from virtual_shelly import metrics as MET  # ensure path set by conftest
    # Trigger some HTTP RPC calls to populate metrics
    r1 = client.get("/rpc/EM.GetStatus", params={"id": 0})
    assert r1.status_code == 200
    r2 = client.get("/rpc", params={"method": "Shelly.GetDeviceInfo"})
    assert r2.status_code == 200
    # Now fetch metrics
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "virtual_shelly_http_rpc_requests_total" in text
    assert "virtual_shelly_ws_clients" in text
    # Ensure per-method counters appear
    assert "EM.GetStatus" in text
    assert "Shelly.GetDeviceInfo" in text


def test_ui_served(client):
    r = client.get("/ui")
    assert r.status_code == 200
    assert "<!doctype html>" in r.text.lower()
    assert "Virtual Shelly 3EM Pro" in r.text
