Project TODO

Purpose: Track high‑impact tasks to improve Shelly Gen2 compatibility, stability, and UX. Items are ordered roughly by value.

Core Compatibility

- [x] WebSocket notifications
  - [x] Add coalescing to avoid flooding during rapid updates.

HTTP Endpoints

UDP RPC (b2500 compatibility)

- [ ] Add tests for typical/edge payloads and malformed input; ensure silent ignore on invalid packets.

mDNS and Discovery

- [ ] Validate TXT set against Shelly conventions (id, app, model, ver, fw_id, mac, gen).

Persistence & Timing

Configuration & Runtime

Observability & Logging

- [x] Structured logs with levels; concise request summaries for /rpc, ws:/rpc, UDP.
- [x] Optional /metrics (Prometheus) for basic counters (requests, errors).

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
- [x] Provide example client snippets (HTTP, WS, UDP) in an examples/ folder.

Docker & CI

- [ ] Multi‑arch images and GHCR publish workflow (GitHub Actions).
- [ ] Optionally expose timezone/env configuration and log formatting flags.

Known Limitations (to revisit)

- [ ] mDNS reliability varies by network/container host; host networking is recommended.
- [ ] Not a full Shelly implementation; only a subset required for common consumers.
