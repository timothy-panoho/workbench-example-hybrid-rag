#!/bin/bash

CHAIN_SERVER_CMD="$HOME/.conda/envs/api-env/bin/python -m uvicorn chain_server.server:app --port=8000 --host=0.0.0.0"
PROFILE_FILE="/project/.model-profile"

# ── Helpers ────────────────────────────────────────────────────────────────────

start_chain_server() {
    echo "Starting chain server..."
    cd /project/code/ && $CHAIN_SERVER_CMD &

    ATTEMPTS=0
    MAX_ATTEMPTS=30
    while [ "$(/usr/bin//usr/bin/curl -o /dev/null -s -w "%{http_code}" "http://localhost:8000/health")" != "200" ]; do
        ATTEMPTS=$((ATTEMPTS+1))
        if [ "$ATTEMPTS" -eq "$MAX_ATTEMPTS" ]; then
            echo "Max attempts reached ($MAX_ATTEMPTS). Chain server failed to start."
            exit 1
        fi
        echo "Polling chain server. Awaiting status 200; trying again in 5s."
        sleep 5
    done
    echo "Chain server is up."
}

auto_launch_model() {
    # If Docker is available and a profile was previously chosen, restart it.
    if ! command -v docker >/dev/null 2>&1; then
        return
    fi
    if [ ! -f "$PROFILE_FILE" ]; then
        return
    fi
    LAST_PROFILE=$(cat "$PROFILE_FILE")
    if [ -z "$LAST_PROFILE" ]; then
        return
    fi
    echo "Auto-launching local model profile: $LAST_PROFILE"
    /project/code/scripts/launch-model.sh "$LAST_PROFILE" &
}

# ── Main logic ─────────────────────────────────────────────────────────────────

if pgrep -x "milvus" > /dev/null; then

    # Milvus already running — make sure chain server is also up
    if [[ "$(/usr/bin/curl -o /dev/null -s -w "%{http_code}" --max-time 3 "http://localhost:8000/health")" != "200" ]]; then
        echo "Chain server not responding — restarting..."
        start_chain_server
    fi

    # Check Milvus REST API
    if [[ "$(/usr/bin/curl -o /dev/null -s -w "%{http_code}" --max-time 3 "http://localhost:19530/v1/vector/collections")" != "200" ]]; then
        echo "Error: Milvus REST API not responding."
        exit 2
    fi

    # Auto-launch the last-used local model (idempotent — skipped if already running)
    auto_launch_model

    echo "All services running."
    exit 0

else

    # Fresh start — bring up Milvus, chain server, and local model
    echo "Starting Milvus..."
    $HOME/.local/bin/milvus-server --data /mnt/milvus/ &

    start_chain_server

    # Auto-launch the last-used local model
    auto_launch_model

    echo "Service reachable. Happy chatting!"
    exit 2

fi
