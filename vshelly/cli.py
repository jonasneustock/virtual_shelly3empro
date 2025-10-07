from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import time
from typing import Optional, Any

import httpx
from websockets.sync.client import connect as ws_connect  # type: ignore


def _client(base: str, timeout: float = 5.0, transport: Optional[httpx.BaseTransport] = None) -> httpx.Client:
    return httpx.Client(base_url=base.rstrip('/'), timeout=timeout, transport=transport)


def _is_async_transport(transport: Optional[Any]) -> bool:
    try:
        return hasattr(transport, 'handle_async_request')
    except Exception:
        return False


def cmd_status(base: str, transport: Optional[httpx.BaseTransport] = None) -> int:
    if _is_async_transport(transport):
        async def run():
            async with httpx.AsyncClient(base_url=base.rstrip('/'), transport=transport) as c:  # type: ignore[arg-type]
                r1 = await c.get('/shelly')
                r1.raise_for_status()
                r2 = await c.get('/rpc/EM.GetStatus', params={"id": 0})
                r2.raise_for_status()
                print(json.dumps({"shelly": r1.json(), "em": r2.json()}, indent=2))
        asyncio.run(run())
        return 0
    else:
        with _client(base, transport=transport) as c:
            r1 = c.get('/shelly'); r1.raise_for_status()
            r2 = c.get('/rpc/EM.GetStatus', params={"id": 0}); r2.raise_for_status()
            print(json.dumps({"shelly": r1.json(), "em": r2.json()}, indent=2))
        return 0


def cmd_reset(base: str, transport: Optional[httpx.BaseTransport] = None) -> int:
    if _is_async_transport(transport):
        async def run():
            async with httpx.AsyncClient(base_url=base.rstrip('/'), transport=transport) as c:  # type: ignore[arg-type]
                r = await c.post('/rpc', json={"id": 1, "method": "EMData.ResetCounters", "params": {}})
                r.raise_for_status(); print(r.text)
        asyncio.run(run()); return 0
    else:
        with _client(base, transport=transport) as c:
            r = c.post('/rpc', json={"id": 1, "method": "EMData.ResetCounters", "params": {}})
            r.raise_for_status(); print(r.text)
        return 0


def cmd_metrics(base: str, transport: Optional[httpx.BaseTransport] = None) -> int:
    if _is_async_transport(transport):
        async def run():
            async with httpx.AsyncClient(base_url=base.rstrip('/'), transport=transport) as c:  # type: ignore[arg-type]
                r = await c.get('/metrics')
                r.raise_for_status(); print(r.text)
        asyncio.run(run()); return 0
    else:
        with _client(base, transport=transport) as c:
            r = c.get('/metrics'); r.raise_for_status(); print(r.text)
        return 0


def cmd_discover(timeout_s: float = 5.0) -> int:
    # mDNS browse _shelly._tcp.local.
    from zeroconf import Zeroconf, ServiceBrowser

    class Listener:
        def __init__(self):
            self.found = {}

        def add_service(self, zc, type_, name):
            info = zc.get_service_info(type_, name)
            if not info:
                return
            addrs = []
            for a in info.addresses:
                try:
                    addrs.append(socket.inet_ntoa(a))
                except Exception:
                    pass
            self.found[name] = {
                "name": name,
                "port": info.port,
                "addresses": addrs,
                "properties": {k.decode(): v.decode() for k, v in (info.properties or {}).items()},
            }

    zc = Zeroconf()
    lst = Listener()
    ServiceBrowser(zc, "_shelly._tcp.local.", lst)
    time.sleep(max(1.0, timeout_s))
    zc.close()
    print(json.dumps(lst.found, indent=2))
    return 0


def cmd_udp_test(host: str, port: int = 2220, method: str = "EM.GetStatus", chan_id: int = 0, timeout_s: float = 1.0) -> int:
    req = json.dumps({"id": 1, "src": "cli", "method": method, "params": {"id": chan_id}}, separators=(",", ":")).encode()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout_s)
    sock.sendto(req, (host, port))
    try:
        data, _ = sock.recvfrom(8192)
        print(data.decode())
        return 0
    except socket.timeout:
        print("timeout", file=sys.stderr)
        return 2
    finally:
        sock.close()


def cmd_ws_tail(base: str, count: int = 0) -> int:
    # Connect and print Notify* messages; stop after count>0 messages
    base = base.rstrip('/')
    scheme = 'wss' if base.startswith('https') else 'ws'
    url = base.replace('http', scheme) + '/rpc'
    n = 0
    try:
        with ws_connect(url, open_timeout=5) as ws:
            while True:
                msg = ws.recv()
                if not msg:
                    break
                try:
                    obj = json.loads(msg)
                    if isinstance(obj, dict) and str(obj.get('method','')).startswith('Notify'):
                        print(msg)
                        n += 1
                        if count > 0 and n >= count:
                            break
                except Exception:
                    print(msg)
    except Exception as e:
        print(f"ws error: {e}", file=sys.stderr)
        return 2
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog='vshelly', description='CLI for Virtual Shelly Pro 3EM')
    sub = p.add_subparsers(dest='cmd', required=True)

    p_status = sub.add_parser('status', help='Show device and EM status')
    p_status.add_argument('--base', default='http://127.0.0.1:80', help='Base URL, e.g. http://host:port')

    p_reset = sub.add_parser('reset', help='Reset energy counters')
    p_reset.add_argument('--base', default='http://127.0.0.1:80')

    p_metrics = sub.add_parser('metrics', help='Fetch /metrics')
    p_metrics.add_argument('--base', default='http://127.0.0.1:80')

    p_disc = sub.add_parser('discover', help='mDNS discovery of _shelly._tcp')
    p_disc.add_argument('--timeout', type=float, default=5.0)

    p_udp = sub.add_parser('udp-test', help='Send UDP RPC test packet')
    p_udp.add_argument('host', help='Target host/IP')
    p_udp.add_argument('--port', type=int, default=2220)
    p_udp.add_argument('--method', default='EM.GetStatus')
    p_udp.add_argument('--id', type=int, default=0, help='Channel id')
    p_udp.add_argument('--timeout', type=float, default=1.0)

    p_wst = sub.add_parser('ws-tail', help='Tail Notify* messages from WS /rpc')
    p_wst.add_argument('--base', default='http://127.0.0.1:80')
    p_wst.add_argument('--count', type=int, default=0, help='Stop after N messages (0=forever)')

    args = p.parse_args(argv)
    if args.cmd == 'status':
        return cmd_status(args.base)
    if args.cmd == 'reset':
        return cmd_reset(args.base)
    if args.cmd == 'metrics':
        return cmd_metrics(args.base)
    if args.cmd == 'discover':
        return cmd_discover(args.timeout)
    if args.cmd == 'udp-test':
        return cmd_udp_test(args.host, args.port, args.method, args.id, args.timeout)
    if args.cmd == 'ws-tail':
        return cmd_ws_tail(args.base, args.count)
    return 1


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
