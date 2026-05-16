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
    docker compose -f "$COMPOSE_FILE" down 2>&1 || true
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

# Stop any currently running profile containers cleanly
docker compose -f "$COMPOSE_FILE" down 2>&1 || true

# Start the requested profile
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" --profile "$PROFILE" up -d

# Create the network if it doesn't already exist
docker network create "$NETWORK" 2>/dev/null || true

# Connect the project container to the model network (idempotent)
docker network connect "$NETWORK" "$CONTAINER" 2>/dev/null || true

echo "Profile '$PROFILE' is up and network '$NETWORK' is connected."
