from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List, Tuple


class RPCUDPProtocol(asyncio.DatagramProtocol):
    def __init__(
        self,
        max_size: int,
        methods: Dict[str, Callable[[Dict[str, Any]], Any]],
        rpc_response_bytes: Callable[[bytes], bytes],
        udp_build_response: Callable[[Dict[str, Any]], Dict[str, Any] | None],
        on_packet: Callable[[Tuple[str, int], str | None], None],
        on_reply: Callable[[int], None],
    ):
        self.max_size = max_size
        self.methods = methods
        self.rpc_response_bytes = rpc_response_bytes
        self.udp_build_response = udp_build_response
        self.on_packet = on_packet
        self.on_reply = on_reply
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr):
        # Enforce max payload; ignore oversize like b2500-meter (no JSON-RPC errors)
        if len(data) > self.max_size:
            return
        try:
            text = data.decode("utf-8", errors="ignore")
            obj = json.loads(text)
        except Exception:
            return
        try:
            m = obj.get("method") if isinstance(obj, dict) else None
            self.on_packet(addr, m)
        except Exception:
            pass
        # If payload is JSON-RPC 2.0, respond with JSON-RPC envelope
        if isinstance(obj, (dict, list)) and (
            (isinstance(obj, dict) and obj.get("jsonrpc") == "2.0") or
            (isinstance(obj, list) and any(isinstance(x, dict) and x.get("jsonrpc") == "2.0" for x in obj))
        ):
            resp = self.rpc_response_bytes(text.encode("utf-8"))
            self.transport.sendto(resp, addr)
            return
        resp_obj = self.udp_build_response(obj)
        if resp_obj is None:
            return
        # Compact separators to match b2500-meter
        payload = json.dumps(resp_obj, separators=(",", ":")).encode()
        self.transport.sendto(payload, addr)
        try:
            self.on_reply(len(payload))
        except Exception:
            pass


async def start_udp_servers(
    ports: List[int],
    max_size: int,
    methods: Dict[str, Callable[[Dict[str, Any]], Any]],
    rpc_response_bytes: Callable[[bytes], bytes],
    udp_build_response: Callable[[Dict[str, Any]], Dict[str, Any] | None],
    on_packet: Callable[[Tuple[str, int], str | None], None],
    on_reply: Callable[[int], None],
):
    loop = asyncio.get_running_loop()
    transports = []
    for p in ports:
        transport, _ = await loop.create_datagram_endpoint(
            lambda: RPCUDPProtocol(max_size, methods, rpc_response_bytes, udp_build_response, on_packet, on_reply),
            local_addr=("0.0.0.0", p),
            allow_broadcast=True,
            reuse_port=True,
        )
        transports.append(transport)
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        for t in transports:
            t.close()


def start_udp_thread(
    ports: List[int],
    max_size: int,
    methods: Dict[str, Callable[[Dict[str, Any]], Any]],
    rpc_response_bytes: Callable[[bytes], bytes],
    udp_build_response: Callable[[Dict[str, Any]], Dict[str, Any] | None],
    on_packet: Callable[[Tuple[str, int], str | None], None],
    on_reply: Callable[[int], None],
):
    import threading
    def _run():
        asyncio.run(start_udp_servers(ports, max_size, methods, rpc_response_bytes, udp_build_response, on_packet, on_reply))
    threading.Thread(target=_run, daemon=True).start()

