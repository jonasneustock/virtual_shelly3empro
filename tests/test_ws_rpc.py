import json


def test_websocket_rpc_basic_flow(client):
    with client.websocket_connect("/rpc") as ws:
        # First message should be a NotifyFullStatus
        initial = ws.receive_text()
        obj = json.loads(initial)
        assert obj.get("method") == "NotifyFullStatus"
        assert "params" in obj

        # Send a JSON-RPC ping and verify response
        ws.send_text(json.dumps({"id": 1, "method": "Shelly.Ping", "params": {}}))
        resp = json.loads(ws.receive_text())
        assert resp["id"] == 1
        assert resp["result"]["pong"] is True

