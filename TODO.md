Project TODO

Purpose: Track high‑impact tasks to improve Shelly Gen2 compatibility, stability, and UX. Items are ordered roughly by value.

Core Compatibility

- [ ] WebSocket notifications
  - [ ] Push initial NotifyStatus/NotifyFullStatus upon WS connect (ws:/rpc).
  - [ ] Broadcast NotifyStatus to all connected WS clients after each HA poll (on state changes or at a throttled cadence).
  - [ ] Maintain a thread‑safe registry of WS clients; handle disconnects.
  - [ ] Add rate limiting/coalescing to avoid flooding during rapid updates.
- [ ] Discovery helpers
  - [ ] Shelly.ListMethods → return supported method names.
  - [ ] Sys.GetInfo → mirror Shelly device info (model, fw_id, mac, id, etc.).
  - [ ] Sys.SetConfig → partial update; reuse current SetConfig logic.
- [ ] EM/EMData polish
  - [ ] Validate params.id for EM.GetStatus/EMData.GetStatus (accept id=0 only; error otherwise).
  - [ ] Keep STRICT_MINIMAL_PAYLOAD toggle but default to full payload for Shelly app compatibility.

HTTP Endpoints

- [ ] Enrich GET /shelly
  - [ ] Include basic link info (e.g., {"wifi_sta": {"connected": false}, "eth": {"connected": true}, "discoverable": true}).
  - [ ] Optionally add uptime snapshot.
- [ ] GET /rpc batch (optional)
  - [ ] Support batch requests with a query param (rarely needed).

UDP RPC (b2500 compatibility)

- [ ] Keep current Shelly‑style UDP behavior (EM.GetStatus, EM1.GetStatus) with decimal enforcer.
- [ ] Add tests for typical/edge payloads and malformed input; ensure silent ignore on invalid packets.
- [ ] Optional: add JSON‑RPC 2.0 over UDP fallback when payload includes "jsonrpc":"2.0" (low priority).

mDNS and Discovery

- [ ] Validate TXT set against Shelly conventions (id, app, model, ver, fw_id, mac, gen).
- [ ] Set sensible TTL; re‑announce on IP changes.
- [ ] Ensure correct interface selection on multi‑homed hosts; respect MDNS_IP override.

Persistence & Timing

- [ ] Replace now % 30 persist trigger with last_persist_ts + interval logic.
- [ ] Use time.monotonic() for energy integration dt to avoid wall clock jumps.
- [ ] Flush energy state on shutdown (SIGTERM handler for Docker stop).

Configuration & Runtime

- [ ] Unify advertised HTTP port and actual listen port (either run uvicorn on HTTP_PORT or document/compute advertised port).
- [ ] Add env toggles for optional features (WS fan‑out range, UDP ports, mDNS).
- [ ] Optional CORS for browser‑side debugging.

Observability & Logging

- [ ] Structured logs with levels; concise request summaries for /rpc, ws:/rpc, UDP.
- [ ] Optional /metrics (Prometheus) for basic counters (requests, errors).

Testing

- [ ] Unit tests
  - [ ] UDP response builder (rounding/decimal rules, EM and EM1).
  - [ ] HTTP GET/POST /rpc and GET /shelly payloads.
  - [ ] WS connect + initial notify; broadcast on poll.
  - [ ] Energy integration over simulated dt and persistence cycle.
- [ ] Integration tests
  - [ ] End‑to‑end add‑by‑IP flow matching Shelly app expectations (where possible).

Documentation

- [ ] Link to official Shelly Gen2 HTTP/WS RPC docs and payload references.
- [ ] Document which methods are implemented and their response shapes.
- [ ] Provide example client snippets (HTTP, WS, UDP) in an examples/ folder.

Docker & CI

- [ ] Add Docker HEALTHCHECK (calls /healthz).
- [ ] Multi‑arch images and GHCR publish workflow (GitHub Actions).
- [ ] Optionally expose timezone/env configuration and log formatting flags.

Known Limitations (to revisit)

- [ ] mDNS reliability varies by network/container host; host networking is recommended.
- [ ] Not a full Shelly implementation; only a subset required for common consumers.

