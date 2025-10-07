import json
import types

import httpx


def test_cli_status_and_reset(client, app_module, monkeypatch):
    # Construct ASGI transport for httpx
    transport = httpx.ASGITransport(app=app_module.app)
    import vshelly.cli as cli

    # status should print JSON and return 0
    rc = cli.cmd_status('http://test', transport=transport)
    assert rc == 0

    # reset should return ok
    rc = cli.cmd_reset('http://test', transport=transport)
    assert rc == 0


def test_cli_metrics(client, app_module):
    transport = httpx.ASGITransport(app=app_module.app)
    import vshelly.cli as cli
    rc = cli.cmd_metrics('http://test', transport=transport)
    assert rc == 0

