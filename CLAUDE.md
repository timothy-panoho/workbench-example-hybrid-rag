# hybrid-rag — Local NIM Setup

NVIDIA Workbench project: hybrid RAG chat over your documents using a locally-running NIM (llama-3.2-3b-instruct) or cloud endpoints.

## Architecture

| Component | What it is | Port |
|---|---|---|
| Project container | Chain server (API) + Chat UI (Gradio) + Milvus vector DB | 8000 (API), 8080 (UI) |
| local-nim (compose) | llama-3.2-3b-instruct NIM, runs outside the project container | 8000 |

The project container and the local-nim compose service communicate over the `hybrid-rag` Docker network. Inside the container the NIM is reached via `host.docker.internal:8000`.

## GPU Allocation

| Service | GPU |
|---|---|
| Project container | **0** — explicitly disabled in `spec.yaml` (`resources.gpu.requested: 0`). Embeddings run on CPU (`EMBEDDING_DEVICE=cpu`). |
| local-nim (compose) | **1** — `compose.yaml` reserves `count: 1` GPU, using the GPU not taken by the project container. |

On a single-GPU machine comment out `requested: 0` and the `count: 1` reservation will overlap — pick one or the other.

## Local NIM (Docker Desktop)

The NIM runs as a standalone Docker Compose service, **not** managed by Workbench. Start it separately before launching the chat app.

```yaml
# compose.yaml
services:
  local-nim:
    image: nvcr.io/nim/meta/llama-3.2-3b-instruct:latest
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8000:8000"
    volumes:
      - type: bind
        source: /mnt/c/nim-cache   # maps to C:\nim-cache on the Windows host
        target: /opt/nim/.cache/
    environment:
      - NGC_API_KEY=${NVIDIA_API_KEY}
networks:
  default:
    name: hybrid-rag
```

```bash
# From the project root (WSL or Docker Desktop terminal):
docker compose up -d          # start NIM
docker compose logs -f        # tail logs; wait for "Server started" before using chat
docker compose down           # stop NIM
```

The model cache lives at `C:\nim-cache` on Windows (bind-mounted via WSL path `/mnt/c/nim-cache`). First boot downloads the model; subsequent starts are fast.

## Network Setup

Docker Compose creates a bridge network named `hybrid-rag`. The project container connects to it automatically at Workbench startup. The chain server reaches the NIM using the special DNS name `host.docker.internal` (resolves to the Docker Desktop host).

Set in `variables.env`:
```
NIM_ENDPOINT_URL=http://host.docker.internal:8000
```

## pkg_resources Patch (required after every container rebuild)

The chain server's `api-env` conda environment needs `setuptools==69.5.1` pinned; without it `pkg_resources` fails at import. This is already baked into `postBuild.bash`:

```bash
$HOME/.conda/envs/api-env/bin/pip install 'setuptools==69.5.1' --quiet
```

If you see a `pkg_resources` / `DistributionNotFound` traceback when starting the chain server after a rebuild, run the line above manually inside the container:

```bash
# Inside the Workbench project terminal:
~/.conda/envs/api-env/bin/pip install 'setuptools==69.5.1' --quiet
```

This is a workaround for an upstream incompatibility between `unstructured` and newer setuptools; the pin must survive each rebuild because `postBuild.bash` re-creates the env from scratch.

## Environment Variables

`variables.env` is **not committed** (listed in `.gitignore`). Copy from this template and fill in your tokens:

```env
# NIM endpoint (local RTX)
NIM_ENDPOINT_URL=http://host.docker.internal:8000

# HuggingFace
HUGGINGFACE_HUB_CACHE=/data/
HUGGING_FACE_HUB_TOKEN=hf_...

# Embedding device: cpu or cuda:0
EMBEDDING_DEVICE=cpu
```

`NVIDIA_API_KEY` is managed as a Workbench secret (see `spec.yaml` → `execution.secrets`) **and** in a local `.env` file for the compose service. `.env` is not committed — keep it out of git.

## Starting the Full Stack

1. Start the NIM: `docker compose up -d` (wait for healthy)
2. Open the project in NVIDIA Workbench and start the **chat** app
3. Navigate to `http://localhost:8080`

## Scripts

| Script | Purpose |
|---|---|
| `code/scripts/rag-consolidated.sh` | Starts Milvus + chain server; used by the Workbench app launcher |
| `code/scripts/upload-docs.sh` | Uploads documents into the vector store |
| `code/scripts/clear-docs.sh` | Clears the Milvus collection |