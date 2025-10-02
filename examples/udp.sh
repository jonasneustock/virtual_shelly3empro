#!/usr/bin/env bash
set -euo pipefail
HOST=${1:-127.0.0.1}
PORT=${2:-2220}
REQ='{"id":1,"src":"cli","method":"EM.GetStatus","params":{"id":0}}'
echo -n "$REQ" | nc -u -w1 $HOST $PORT

