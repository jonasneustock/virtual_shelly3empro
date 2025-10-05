Project TODO

Purpose: Track high‑impact tasks to improve Shelly Gen2 compatibility, stability, and UX. Items are ordered roughly by value.

Core Compatibility

- [x] WebSocket notifications
  - [x] Add coalescing to avoid flooding during rapid updates.

HTTP Endpoints

UI / Dashboard

- [x] Minimal dashboard at `/ui` with current values and metrics
- [x] Add live power graphs (total + A/B/C) with 10‑minute rolling window
- [x] Add graphs for voltages and currents per phase
- [x] Add time window selector (10m/1h/6h) with simple in‑memory buffers
- [ ] Switch UI polling to WebSocket subscribe (reduce latency/overhead)
- [ ] Dark mode + responsive layout improvements (mobile friendly)
- [ ] Export CSV of recent samples from the UI

UDP RPC (b2500 compatibility)

- [ ] Add tests for typical/edge payloads and malformed input; ensure silent ignore on invalid packets.
  - [x] Typical payloads (EM.GetStatus, EM1.GetStatus) covered
  - [ ] Malformed/oversize payloads and silent ignore behavior

mDNS and Discovery

- [ ] Validate TXT set against Shelly conventions (id, app, model, ver, fw_id, mac, gen).

Persistence & Timing

- [ ] Persist recent samples for graph warm‑start after restart (bounded size)
- [x] Make `HA_SMOOTHING_WINDOW` configurable via env var
- [ ] Document and tune request‑side power scaling behaviour
  - [x] Implement opt‑out flag `REQUEST_SIDE_SCALING_ENABLE`
  - [ ] Expand README docs and examples

Configuration & Runtime

- Config management
  - [ ] Extract config/env parsing into `config.py` (single source of truth)
  - [ ] Validate config on boot (URLs, entity IDs, ports); fail‑fast with clear logs
  - [ ] Optional pydantic settings model with `.env` file support (lazy import to avoid runtime bloat)
  - [ ] Document and group env vars by concern (HA, HTTP/WS, UDP, Modbus, mDNS, UI)

- Runtime controls
  - [ ] Hot‑reload of config via `/admin/reload` or SIGHUP for non‑critical toggles (CORS, smoothing, STRICT_MINIMAL_PAYLOAD)
  - [x] Toggle request‑side power scaling via `REQUEST_SIDE_SCALING_ENABLE` (default on), with `REQUEST_IP_TTL` honored
  - [x] Make `HA_SMOOTHING_WINDOW` configurable; allow per‑sensor smoothing overrides
  - [ ] Honor `DISABLE_BACKGROUND` in docs and example Compose for testing
  - [ ] Graceful shutdown improvements: persist state, close WS/UDP sockets, unregister mDNS, flush Modbus image

- Network & bindings
  - [ ] Separate bind addresses for HTTP, WS fan‑out, UDP, Modbus (env)
  - [ ] TLS guidance (reverse proxy example with Caddy/Traefik/Nginx); optional uvicorn TLS flags for lab use
  - [ ] Timeouts and max body size for HTTP/WS; harden against oversized payloads

- AuthN/AuthZ & security
  - [x] Optional basic auth for `/ui` and `/admin/overview` (LAN‑only by default)
  - [ ] Optional token for `/rpc` (header or query), disabled by default to preserve Shelly‑like behavior
  - [ ] IP allow/deny lists for HTTP/WS/UDP endpoints

- Rate limiting & QoS
  - [x] Lightweight per‑IP rate limiting for `/rpc` HTTP (GET/POST)
  - [ ] Backpressure and debounce options for WS Notify broadcasts

- Resilience
  - [ ] HA polling: exponential backoff/jitter, mark sensors unavailable on sustained failures
  - [ ] Surface upstream status in `/shelly` and `/admin/overview` (e.g., `ha_connected: true/false`)
  - [ ] Retries + timeouts for HA requests (move to `httpx` async client with connection pool)

- CORS & headers
  - [ ] Expand CORS config (regex/explicit origin list); document preflight behavior
  - [ ] Security headers for `/ui` (CSP optional, no inline eval)

- CLI & args
  - [ ] Optional CLI flags to override env (port, bind, log‑level) for non‑Docker runs

- Safety limits
  - [ ] Enforce caps on `UDP_MAX` and WS message sizes; sanitize RPC input fields

- mDNS & Modbus
  - [ ] mDNS: TTL and interface selection configurable; auto‑refresh on IP change (already present) with logging
  - [ ] Modbus: explicit error handling when port in use; configurable unit ID/port/bind via config module


Observability & Logging

- [x] Structured logs with levels; concise request summaries for /rpc, ws:/rpc, UDP.
- [x] Optional /metrics (Prometheus) for basic counters (requests, errors).
- [x] Add request latency histograms to /metrics
- [ ] Add p95/p99 summaries to /metrics

Testing

- [x] Unit tests
  - [x] UDP response builder (rounding/decimal rules, EM and EM1).
  - [x] HTTP GET/POST /rpc (partial); add `/shelly` payload coverage.
  - [x] WS connect + initial notify; (broadcast on poll still to cover).
  - [ ] Energy integration over simulated dt and persistence cycle.
- [ ] Integration tests
  - [ ] End‑to‑end add‑by‑IP flow matching Shelly app expectations (where possible).

Documentation

- [ ] Link to official Shelly Gen2 HTTP/WS RPC docs and payload references.
- [ ] Document which methods are implemented and their response shapes.
- [x] Provide example client snippets (HTTP, WS, UDP) in an examples/ folder.

Docker & CI

- Docker image & Compose
  - [ ] Multi‑arch images (linux/amd64, linux/arm64) using buildx
  - [ ] Slimmer multi‑stage Dockerfile (non‑root user, read‑only FS, drop caps)
  - [ ] Add HEALTHCHECK and OCI labels (org.opencontainers.image.*)
  - [ ] Optional timezone/env/log formatting flags surfaced via env
  - [ ] Example Compose files: host networking vs. bridged with UDP ports
  - [ ] Document volume for `/data` and minimal permissions
  - [ ] Hadolint check for Dockerfile
- Build, Publish, Security
  - [ ] GH Actions: build matrix, cache, push to GHCR (and optional Docker Hub)
  - [ ] Tagging: semver (`vX.Y.Z`), `latest`, and git SHA tags
  - [ ] SBOM generation (syft) and image signing (cosign)
  - [ ] Vulnerability scan (Trivy/Grype) gate on critical/high
- CLI tooling
  - [ ] Python CLI (`vshelly`) to interact with the service:
    - [ ] `status` (HTTP/WS), `reset`, `metrics`, `discover`, `udp-test`, `ws-tail`
  - [ ] Package to PyPI with console‑script entrypoint; support `pipx install`
  - [ ] Shell completions (bash/zsh/fish) and `--help` docs
  - [ ] Example snippets for common flows (reset, watch total power)
  - [ ] CLI integration tests against ephemeral TestClient/uvicorn
- CI pipeline
  - [ ] Lint (ruff/flake8), format (black), import sort (isort)
  - [ ] Type‑check (mypy) and unit/integration test matrix (Py 3.11/3.12)
  - [ ] Coverage upload (Codecov) and artifacts (SBOM, reports)
  - [ ] Release workflow to cut tags, build images, publish packages

Known Limitations (to revisit)

- [ ] mDNS reliability varies by network/container host; host networking is recommended.
- [ ] Not a full Shelly implementation; only a subset required for common consumers.
