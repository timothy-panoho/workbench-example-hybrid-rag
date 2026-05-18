#!/bin/bash
# Launch or stop a local model container via Docker Compose.
#
# Usage:
#   launch-model.sh gemma              # Ollama  — gemma3:4b
#   launch-model.sh qwen3              # Ollama  — qwen3:4b
#   launch-model.sh llama              # NIM     — meta/llama-3.2-3b-instruct
#   launch-model.sh hf [MODEL_ID]      # vLLM    — any HuggingFace model ID
#                                      #   default: google/gemma-3-4b-it
#   launch-model.sh stop               # stop all profile containers

set -e

PROFILE="${1:-}"
HF_MODEL="${2:-}"
COMPOSE_FILE="/project/compose.yaml"
ENV_FILE="/project/.env"
CONTAINER="project-hybrid-rag"
NETWORK="hybrid-rag"

if [ -z "$PROFILE" ]; then
    echo "Usage: $0 <gemma|qwen3|llama|hf [model_id]|stop>" >&2
    exit 1
fi

if [ "$PROFILE" = "stop" ]; then
    for c in local-ollama local-nim-llama local-hf; do
        if docker inspect "$c" >/dev/null 2>&1; then
            echo "Stopping $c..."
            docker stop "$c" 2>/dev/null || true
            docker rm   "$c" 2>/dev/null || true
        fi
    done
    echo "All local model containers stopped."
    exit 0
fi

# ── For the hf (vLLM) profile, write model ID to .env so compose picks it up ──
if [ "$PROFILE" = "hf" ]; then
    MODEL_ID="${HF_MODEL:-google/gemma-3-4b-it}"
    # Preserve any existing entries, then upsert HF_MODEL_ID
    touch "$ENV_FILE"
    grep -v "^HF_MODEL_ID=" "$ENV_FILE" > "${ENV_FILE}.tmp" || true
    echo "HF_MODEL_ID=${MODEL_ID}" >> "${ENV_FILE}.tmp"
    mv "${ENV_FILE}.tmp" "$ENV_FILE"
    echo "Model ID set to: ${MODEL_ID}"
fi

# Ensure the shared network exists before compose tries to use it (external: true)
docker network create "$NETWORK" 2>/dev/null || true

# Stop any running model containers by name.
# (docker compose down would silently do nothing because the compose project name
# differs between the host and inside this container — stop by name is reliable.)
for c in local-ollama local-nim-llama local-hf; do
    if docker inspect "$c" >/dev/null 2>&1; then
        echo "Stopping $c..."
        docker stop "$c" 2>/dev/null || true
        docker rm   "$c" 2>/dev/null || true
    fi
done

# The compose project name must match what the host uses (the repo directory name).
# Dynamic detection via bind-mount paths is unreliable inside Docker Desktop WSL2
# containers (the path contains a hash, not the directory name). Hardcode it.
PROJECT_NAME="timothy-panoho-workbench-example-hybrid-rag"

# Start the requested profile using the correct project name
docker compose \
    -f "$COMPOSE_FILE" \
    --env-file "$ENV_FILE" \
    --project-name "$PROJECT_NAME" \
    --profile "$PROFILE" \
    up -d

# Connect the project container to the model network (idempotent)
docker network connect "$NETWORK" "$CONTAINER" 2>/dev/null || true

echo "Profile '$PROFILE' is up (project: $PROJECT_NAME) and network '$NETWORK' is connected."
