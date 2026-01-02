Virtual Shelly Pro 3EM (Home Assistant–backed)

Overview

- Emulates a Shelly Pro 3EM device. It polls power/voltage/current/PF values from Home Assistant (HA) and exposes Shelly-like RPC over HTTP, WebSocket, and UDP, with mDNS discovery and simple energy integration/persistence.
- Useful for gateways/apps that expect a Shelly 3‑phase power meter on the LAN (e.g., Shelly app, b2500 integrations, custom dashboards).
- See detailed architecture and operational notes in `TECHNICAL_DOC.md`.

Features

- HTTP API (FastAPI + Uvicorn) with:
  - GET `/shelly`: Gen2 device info (id, app, ver, model, gen, mac, sn, auth flags).
  - GET `/ui`: Minimal web interface showing current values, recent client IPs, and endpoint metrics (auto-refreshes every 5s).
  - POST `/rpc`: JSON‑RPC 2.0 envelope for Shelly.* and EM*/EMData* methods.
  - GET `/rpc?method=...` and GET `/rpc/{method}`: returns the method result directly (no envelope), matching Shelly GET semantics.
  - GET `/healthz` and GET `/`: simple health and info.
- WebSockets:
  - WS `/rpc`: JSON‑RPC over WebSocket (common for Shelly Gen2).
  - Fan‑out WS servers on TCP ports 6010–6022 echoing RPC responses (configurable).
  - On `/rpc` connect: sends `NotifyFullStatus`; then periodic `NotifyStatus` broadcasts (throttled by `WS_NOTIFY_INTERVAL`).
- UDP RPC (Shelly‑style, compatible with tomquist/b2500‑meter):
  - Supports `EM.GetStatus` (3‑phase) and `EM1.GetStatus` (single total) with identical rounding/decimal behavior.
  - Listens on configurable UDP ports (defaults include 1010 and 2220).
- Modbus TCP server (pymodbus):
  - Exposes key Shelly Pro 3EM measurements (voltage, current, power, PF, energy totals) as 32-bit floats.
  - Read-only input/holding registers share the same data; writing `1` to holding register `4200` resets the virtual energy counters.
  - Defaults to TCP port 502 with unit ID 1 (configurable via env vars).
- mDNS service advertisements (`_http._tcp` and `_shelly._tcp`) for discovery.
- Energy counters integrated from power over time and persisted at `/data/state.json` (via Docker volume).
- Simple `/metrics` endpoint with Prometheus‑style counters for HTTP/WS/UDP events.
- UI power chart with live autoencoder-based forecasting (configurable window/horizon via env vars); forecast values displayed alongside current power and plotted in red.
- HA polling resiliency: model training runs in a background thread so polling keeps running even during fits; manual training remains synchronous for immediate feedback.

Quick Start (Docker Compose)

1) Adjust `docker-compose.yaml` environment to match your HA URL, token, and entity IDs.
2) Ensure your host can use host networking (recommended for UDP + mDNS).
3) Bring it up:
   - `docker compose up -d`
4) Verify:
   - `curl http://<device-ip>/shelly`
   - `curl "http://<device-ip>/rpc/EM.GetStatus?id=0"`
   - `curl http://<device-ip>/healthz`

Manual Run (no Docker)

- Python 3.11+
- `pip install -r requirements.txt`
- `uvicorn app:app --host 0.0.0.0 --port 80`

Configuration (env vars)

- Home Assistant
  - `HA_BASE_URL`: e.g. `http://homeassistant:8123` or `http://192.168.x.x:8123`
  - `HA_TOKEN`: HA Long‑Lived Access Token (read‑only)
  - `POLL_INTERVAL`: seconds between polls (default 2.0)
  - `HA_SMOOTHING_ENABLE`: when `true`, average each sensor reading over the last 5 values
  - Entity IDs (override as needed): `A_POWER`, `B_POWER`, `C_POWER`, `A_VOLT`, `B_VOLT`, `C_VOLT`, `A_CURR`, `B_CURR`, `C_CURR`, `A_PF`, `B_PF`, `C_PF`
- Device identity
  - `DEVICE_ID`: Device identifier reported in RPC/mDNS (default `shellypro3em-virtual-001`)
  - `APP_ID`: Shelly application identifier (default `shellypro3em`)
  - `MODEL`: Hardware model string (default `SHPRO-3EM`)
  - `FIRMWARE`: Firmware version reported as `ver` (default `1.0.0-virt`)
  - `FW_ID`: Firmware build identifier reported as `fw_id` (defaults to the same as `FIRMWARE`)
  - `MAC`, `SN`, `MANUFACTURER`
  - `GENERATION`: Shelly generation advertised via HTTP and mDNS TXT records (default `2`)
- HTTP/WebSocket
  - `HTTP_PORT`: container listens on this port and advertises it via mDNS (default `80`).
  - `WS_PORT_START`, `WS_PORT_END`: WS fan‑out TCP range (default 6010–6022)
  - `WS_NOTIFY_INTERVAL`: throttle seconds for WS `/rpc` `NotifyStatus` (default `2.0`).
  - `WS_NOTIFY_EPS`: coalescing threshold in watts; only broadcast when change ≥ EPS (default `0.1`).
  - `CORS_ENABLE`: enable CORS middleware (`true|false`, default `false`).
  - `CORS_ORIGINS`: comma‑separated allowed origins (default `*`).
  - Request‑side scaling (divide power by active client IPs)
    - `REQUEST_SIDE_SCALING_ENABLE`: `true|false` (default `true`)
    - `REQUEST_SIDE_SCALING_CLIENTS`: integer override for client count (default `0` = auto by active IPs)
- UDP RPC
  - `UDP_PORTS`: comma‑separated list (e.g. `1010,2220`) for old/new Shelly Pro 3EM styles
  - `UDP_MAX`: max UDP payload size (bytes)
- mDNS
  - `MDNS_ENABLE`: `true|false`
  - `MDNS_HOSTNAME`: instance name (defaults to `DEVICE_ID`)
  - `MDNS_IP`: optional explicit IP to advertise (helps with multi‑homed hosts)
- Modbus TCP
  - `MODBUS_ENABLE`: `true|false`
  - `MODBUS_PORT`: TCP port for the Modbus server (default `502`)
  - `MODBUS_BIND`: Bind address (default `0.0.0.0`)
  - `MODBUS_UNIT_ID`: Unit identifier (default `1`)
- Payload shape
  - `STRICT_MINIMAL_PAYLOAD`: when `true`, HTTP/WS `EM.GetStatus` returns only `{a_act_power,b_act_power,c_act_power,total_act_power}` (some gateways prefer this).
- Persistence
  - `STATE_PATH`: defaults to `/data/state.json` (mounted via volume in Compose).
- Forecasting / ML
  - `POWER_RETENTION_DAYS`: how long to retain the training dataset (default `30`).
  - `POWER_FEATURE_WINDOW`: number of most recent samples used as model input features (default `30`).
  - `FORECAST_HORIZON_STEPS`: number of forward steps predicted each cycle (default `30`).
  - `MODEL_TRAIN_INTERVAL`: seconds between automatic retraining runs (default `3600.0`).
  - `POWER_HISTORY_SECONDS`: window of power history returned for the UI chart (default `180`).

APIs

- HTTP JSON‑RPC (POST `/rpc`): send a JSON‑RPC 2.0 request, e.g.
  - `{ "id": 1, "method": "EM.GetStatus", "params": {"id": 0} }`
- HTTP GET RPC:
  - `/rpc?method=EM.GetStatus&id=0`
  - `/rpc/EM.GetStatus?id=0`
  - Response is the method result object, e.g. `{ "a_act_power": 123.4, ... }`
- WebSocket RPC:
  - Connect `ws://<ip>/rpc` and send the same JSON‑RPC envelopes as POST `/rpc`.
  - On connect you’ll receive a `NotifyFullStatus`; during operation `NotifyStatus` messages are broadcast.
- Shelly‑style UDP RPC (b2500 compatible):
  - Send to UDP port `1010` or `2220` (configurable via `UDP_PORTS`)
  - Request (example): `{"id":1,"src":"cli","method":"EM.GetStatus","params":{"id":0}}`
  - Response (example): `{"id":1,"src":"<DEVICE_ID>","dst":"unknown","result":{"a_act_power":X.X,"b_act_power":Y.Y,"c_act_power":Z.Z,"total_act_power":T.TTT}}`
  - Also supports `EM1.GetStatus` -> `{ "result": { "act_power": T.TTT } }`
  - If payload includes `"jsonrpc":"2.0"`, responds with JSON‑RPC envelope (fallback handler).

Example Commands

- HTTP GET:
  - `curl http://<ip>/shelly`
  - `curl "http://<ip>/rpc/EM.GetStatus?id=0"`
- HTTP POST JSON‑RPC:
  - `curl -s http://<ip>/rpc -H 'Content-Type: application/json' -d '{"id":1,"method":"EM.GetStatus","params":{"id":0}}'`
- WebSocket JSON‑RPC (using websocat):
  - `websocat ws://<ip>/rpc`
  - Then send: `{"id":1,"method":"EM.GetStatus","params":{"id":0}}`
- UDP (netcat):
  - `echo -n '{"id":1,"src":"cli","method":"EM.GetStatus","params":{"id":0}}' | nc -u -w1 <ip> 2220`

Home Assistant polling

- Each `POLL_INTERVAL`, HA sensors are fetched. Missing or `unknown/unavailable` values are treated as `None` (or `0.0` for power). Phase power values feed energy integration (kWh) over time.
- Energy counters persist to `STATE_PATH`. You can reset counters via RPC: `EMData.ResetCounters`.

Shelly app notes

- Use host networking in Docker for best compatibility with mDNS/UDP. Ensure the phone and host are on the same subnet/VLAN.
- If discovery fails:
  - Try adding by IP in the app.
  - Set `MDNS_IP` to the actual LAN IP and restart.
  - Ensure firewall allows UDP 5353 (mDNS) and the chosen UDP RPC ports (e.g., 1010/2220).
  - Keep HTTP on port 80 (default in image) as some apps assume it.

Troubleshooting

- Check health: `curl http://<ip>/healthz` (image includes a Docker HEALTHCHECK)
- Inspect metrics: `curl http://<ip>/metrics`
- Open the UI: http://<ip>/ui (auto-refresh every 5s)
- Inspect logs: `docker logs -f shelly3em-virtual`
- Verify endpoints hit by the app (look for GET /shelly, /rpc calls, WS /rpc handshakes).
- Confirm UDP replies with netcat; try both 1010 and 2220 depending on your consumer.

Security

- Do not commit real HA tokens. Use a read‑only long‑lived token.
- This service is intended for trusted LANs. It provides unauthenticated endpoints by design to mimic Shelly devices.

Modbus register map (summary)

- Instantaneous values (float32, big-endian; accessible via holding or input registers):
  - `3000/3001`: Line frequency (Hz)
  - `3002-3007`: Phase A/B/C voltages (V)
  - `3010-3015`: Phase A/B/C currents (A)
  - `3020-3027`: Phase A/B/C active power and total active power (W)
  - `3030-3035`: Power factor for phases A/B/C
- Energy counters (float32 kWh):
  - `3100-3105`: Import energy per phase (A/B/C)
  - `3106-3111`: Export energy per phase (A/B/C)
  - `3112/3113`: Total import energy; `3114/3115`: Total export energy
- Diagnostics and metadata:
  - `3200/3201`: Current UNIX timestamp; `3202/3203`: uptime seconds
  - `3300+`: Device ID, model, firmware, MAC encoded as ASCII (2 chars per register)
  - `3400`: Device ready flag (`1`); `3401`: phase count (`3`)
- Commands:
  - Write `1` to holding register `4200` to reset accumulated energy counters.

Limitations

- Not a full Shelly implementation; only a subset of RPC is supported.
- Energy math is approximate and depends on poll timing from HA.
- mDNS behavior can vary across networks/containers; host networking is recommended.

License

- See repository license(s) of dependencies. This project is provided as‑is for personal/home use.

Shoutouts

- tomquist/b2500-meter: https://github.com/tomquist/b2500-meter
  - This project’s Shelly-style UDP RPC behavior (EM.GetStatus/EM1.GetStatus formatting and decimal handling) is aligned for compatibility with b2500-meter.
- sdeigm/uni-meter: https://github.com/sdeigm/uni-meter
  - A universal meter emulator inspiring the broader idea of adaptable meter emulation on the LAN.
