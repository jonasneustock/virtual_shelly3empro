def test_emdata_reset_counters_via_rpc(client):
    # Call reset and verify ok
    r = client.post("/rpc", json={"id": 1, "method": "EMData.ResetCounters", "params": {}})
    assert r.status_code == 200
    data = r.json()
    assert data["result"]["ok"] is True
    # subsequent EMData.GetStatus should be present
    r2 = client.get("/rpc/EMData.GetStatus", params={"id": 0})
    assert r2.status_code == 200
    payload = r2.json()
    assert "period" in payload and payload["id"] == 0

