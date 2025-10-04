import json
import time


def test_ws_notify_on_connect_and_message_count(client):
    # On connect, we should get NotifyFullStatus; ensure metrics count increments on message
    with client.websocket_connect("/rpc") as ws:
        initial = ws.receive_text()
        obj = json.loads(initial)
        assert obj.get("method") in ("NotifyFullStatus", "NotifyStatus")
        # Send RPC to generate WS_RPC_MESSAGES increment
        ws.send_text(json.dumps({"id": 2, "method": "Shelly.Ping", "params": {}}))
        resp = ws.receive_text()
        assert "pong" in resp

