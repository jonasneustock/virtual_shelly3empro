def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "ts" in data
    assert "device" in data


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data.get("service") == "virtual-shelly-pro-3em"
    assert data.get("rpc") == "/rpc"

