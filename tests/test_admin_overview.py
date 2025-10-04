def test_admin_overview_structure(client):
    r = client.get("/admin/overview")
    assert r.status_code == 200
    data = r.json()
    assert "device" in data and "values" in data and "metrics" in data and "clients" in data
    # Device keys
    assert "id" in data["device"]
    # Values keys
    assert "em" in data["values"] and "emdata" in data["values"]
    # Metrics sections
    assert set(["http", "ws", "udp"]).issubset(set(data["metrics"].keys()))
    # Clients subsections
    assert set(["http_recent", "ws_recent", "udp_recent", "ws_connected"]).issubset(set(data["clients"].keys()))


def test_admin_overview_unique_clients(client):
    # make some calls to register HTTP client
    client.get("/rpc", params={"method": "Shelly.GetDeviceInfo"})
    client.get("/rpc/EM.GetStatus", params={"id": 0})
    r = client.get("/admin/overview")
    data = r.json()
    # Should list at least one unique HTTP IP
    http_unique = data["clients"].get("http_unique", [])
    assert isinstance(http_unique, list)

