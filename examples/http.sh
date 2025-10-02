#!/usr/bin/env bash
set -euo pipefail
HOST=${1:-127.0.0.1}
PORT=${2:-80}

curl -sS http://$HOST:$PORT/shelly | jq . || curl -sS http://$HOST:$PORT/shelly
curl -sS "http://$HOST:$PORT/rpc/EM.GetStatus?id=0" | jq . || true
curl -sS http://$HOST:$PORT/healthz | jq . || true
curl -sS http://$HOST:$PORT/metrics || true

