#!/bin/bash
# Launch or stop a local model container via Docker Compose.
# Usage:
#   launch-model.sh <profile>   # gemma | qwen3 | llama | hf
#   launch-model.sh stop        # stop all profile containers

set -e

PROFILE="${1:-}"
COMPOSE_FILE="/project/compose.yaml"
CONTAINER="project-hybrid-rag"
NETWORK="hybrid-rag"

if [ -z "$PROFILE" ]; then
    echo "Usage: $0 <gemma|qwen3|llama|hf|stop>" >&2
    exit 1
fi

if [ "$PROFILE" = "stop" ]; then
    docker compose -f "$COMPOSE_FILE" down 2>&1 || true
    echo "All local model containers stopped."
    exit 0
fi

# Stop any currently running profile containers cleanly
docker compose -f "$COMPOSE_FILE" down 2>&1 || true

# Start the requested profile
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d

# Create the network if it doesn't already exist
docker network create "$NETWORK" 2>/dev/null || true

# Connect the project container to the model network (idempotent)
docker network connect "$NETWORK" "$CONTAINER" 2>/dev/null || true

echo "Profile '$PROFILE' is up and network '$NETWORK' is connected."
