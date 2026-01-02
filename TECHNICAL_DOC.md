# Technical Documentation: Virtual Shelly Pro 3EM Emulator

## Architecture Overview
- **Language/Frameworks**: Python 3.11+, FastAPI + Uvicorn for HTTP/WS; UDP RPC custom handler; pymodbus for Modbus TCP; scikit-learn MLP for forecasting.
- **Main entry**: `app.py`
- **Packages**: Core logic in `virtual_shelly/` (`metrics.py`, `rpc_core.py`, `udp_server.py`, `ui.py`).
- **Persistence**: Energy counters and training dataset persisted to `STATE_PATH` (default `/data/state.json`).
- **Model persistence**: Trained model persisted via `MODEL_PATH` (default `STATE_PATH + ".model"`); metadata saved alongside state.
- **Networking**:
  - HTTP/WS server on `HTTP_PORT`.
  - WS fan-out servers on TCP ports `WS_PORT_START`–`WS_PORT_END`.
  - UDP RPC on `UDP_PORTS` (Shelly-style EM/EM1).
  - Optional Modbus TCP server on `MODBUS_PORT`.
  - Optional mDNS announcements.

## Data Flow
1) **Polling Home Assistant**: every `POLL_INTERVAL`, HA sensors (`A_POWER`, `B_POWER`, `C_POWER`, volt, current, PF) are fetched (`ha_get`). Smoothing is optional (`HA_SMOOTHING_ENABLE`).
2) **State update**:
   - Power/voltage/current/PF stored per phase in `VirtualPro3EM`.
   - Energy integrated over time (kWh import/export per phase + totals).
   - Total power sample appended to:
     - `power_history` (short UI window).
     - `power_dataset` (retained for ML; pruned to `POWER_RETENTION_DAYS`).
3) **Persistence**: Every ~30s (monotonic cadence) the energy + `power_dataset` are saved to `STATE_PATH`.
4) **APIs**:
   - HTTP `/rpc`, `/rpc?method=...`, `/rpc/{method}` -> JSON-RPC handlers via `rpc_core`.
   - WS `/rpc` -> bidirectional JSON-RPC with NotifyStatus broadcasting.
   - UDP RPC -> Shelly-style EM/EM1 handlers (`udp_server`).
   - Modbus TCP -> exposes measurements via pymodbus blocks (if enabled).
   - `/metrics` -> Prometheus-style counters; `/healthz`; `/shelly` device info.
   - `/ui` -> dashboard (HTML from `virtual_shelly/ui.py`).
   - `/ui/power` -> returns current power, history window, forecast, and model metrics.
   - `/ui/train` -> manual training on full retained dataset.
5) **Forecasting (Autoencoder MLP)**:
   - Window: last `POWER_FEATURE_WINDOW` samples as features.
   - Horizon: predicts next `FORECAST_HORIZON_STEPS` values jointly.
   - Model: `MLPRegressor` with hidden layers (64,16,64), trained hourly (`MODEL_TRAIN_INTERVAL`) or on manual trigger.
   - Metrics: MSE, MAPE, sample count; exposed in `/ui/power` for display.
   - Logging: training start/skip/errors, forecast inputs/outputs.
   - Poller resiliency: all training (scheduled and manual) runs in a background thread so HA polling isn’t blocked; check logs/UI metrics for completion.

## Files of Interest
- `app.py`: FastAPI app, state machine (`VirtualPro3EM`), polling, forecasting, endpoints.
- `virtual_shelly/rpc_core.py`: JSON-RPC dispatch.
- `virtual_shelly/udp_server.py`: UDP RPC server.
- `virtual_shelly/metrics.py`: counters + admin overview.
- `virtual_shelly/ui.py`: HTML/JS dashboard.
- `docker-compose.yaml`: sample deployment.
- `.env` (optional): environment overrides.

## Environment Variables (Key)
- **Home Assistant**: `HA_BASE_URL`, `HA_TOKEN`, `POLL_INTERVAL`, `HA_SMOOTHING_ENABLE`, entity IDs (`A_POWER`, `B_POWER`, `C_POWER`, `A_VOLT`, `B_VOLT`, `C_VOLT`, `A_CURR`, `B_CURR`, `C_CURR`, `A_PF`, `B_PF`, `C_PF`).
- **Device identity**: `DEVICE_ID`, `APP_ID`, `MODEL`, `FIRMWARE`, `FW_ID`, `MAC`, `SN`, `MANUFACTURER`, `GENERATION`.
- **HTTP/WS**: `HTTP_PORT`, `WS_PORT_START`, `WS_PORT_END`, `WS_NOTIFY_INTERVAL`, `WS_NOTIFY_EPS`, `CORS_ENABLE`, `CORS_ORIGINS`.
- **Scaling**: `REQUEST_SIDE_SCALING_ENABLE`, `REQUEST_SIDE_SCALING_CLIENTS`, `REQUEST_IP_TTL`.
- **UDP RPC**: `UDP_PORTS`, `UDP_MAX`.
- **mDNS**: `MDNS_ENABLE`, `MDNS_HOSTNAME`, `MDNS_IP`.
- **Modbus**: `MODBUS_ENABLE`, `MODBUS_PORT`, `MODBUS_BIND`, `MODBUS_UNIT_ID`.
- **Payload shape**: `STRICT_MINIMAL_PAYLOAD`.
- **Persistence**: `STATE_PATH`.
- **Forecasting/ML**: `POWER_RETENTION_DAYS`, `POWER_FEATURE_WINDOW`, `FORECAST_HORIZON_STEPS`, `MODEL_TRAIN_INTERVAL`, `POWER_HISTORY_SECONDS`.
- **Logging**: `LOG_LEVEL`.

## Forecasting Pipeline Details
- **Training trigger**: Hourly (`MODEL_TRAIN_INTERVAL`) if enough samples (`POWER_FEATURE_WINDOW + FORECAST_HORIZON_STEPS`), or manual POST `/ui/train`.
- **Data retention**: `POWER_RETENTION_DAYS` worth of samples; pruned on every insert/persist.
- **Features/targets**: Sliding windows of length `POWER_FEATURE_WINDOW` -> targets of length `FORECAST_HORIZON_STEPS`.
- **Model**: `MLPRegressor(hidden_layer_sizes=(64,16,64), activation="relu", solver="adam", max_iter=400, random_state=42)`.
- **Outputs**: Next `FORECAST_HORIZON_STEPS` power predictions; shown in UI (chart + textual list).
- **Metrics**: MSE, MAPE across all horizon steps; sample count; last train timestamp; dataset size.
- **Logging**: input tail and forecast preview are logged at debug level; training progress/errors at info/error.

## UI Overview
- Served at `/ui`; auto-refreshes overview every 5s and power/forecast every 2s.
- Power card: current total, forecast horizon, forecast value preview, chart (history in blue, forecast in red).
- Model stats: MSE, MAPE, samples used, last train time and sample count, dataset size on the train button.
- Manual training button triggers `/ui/train` and refreshes metrics/forecast.

## Deploy/Run
- **Docker Compose**: adjust `.env` or `docker-compose.yaml` env block; `docker compose up -d`.
- **Manual**: `pip install -r requirements.txt`; `uvicorn app:app --host 0.0.0.0 --port 80`.
- **Persistence**: mount `/data` to retain energy counters and ML dataset (`STATE_PATH`).
- **Networks**: host networking recommended for UDP/mDNS; ensure firewall allows chosen ports.

## Monitoring & Troubleshooting
- **Health**: `GET /healthz`.
- **Metrics**: `GET /metrics`.
- **Logs**: Training/forecast logs include input/outputs and errors; check container logs or stdout.
- **Admin overview**: `GET /admin/overview` (counts + recent clients).
- **HA polling visibility**: `/healthz` includes `last_poll_ok`, `last_poll_age`, and `last_poll_error` to surface HA issues; polling logs warn/error on HA failures and do not block the loop.
- **UDP**: test with `nc -u`; try configured `UDP_PORTS`.
- **WS**: connect to `/rpc`; expect `NotifyFullStatus` then throttled `NotifyStatus`.

## Security Notes
- No auth by design (to mimic Shelly LAN behavior). Deploy only on trusted networks.
- Protect HA token; use read-only long-lived token; do not commit secrets.

## Extensibility
- RPC methods are centralized in `rpc_core.py`; extend `VirtualPro3EM` for new data fields.
- Forecasting model/config is env-driven; adjust architecture (layers/iter) in `train_power_model`.
- UI is a single HTML/JS template in `virtual_shelly/ui.py`; customize charts/metrics there.
