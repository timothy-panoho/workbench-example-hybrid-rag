"""
Model Manager FastAPI application.
All docker calls use: sudo -n /home/workbench/.local/bin/docker ...
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Generator, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent / "static"

DOCKER = ["sudo", "-n", "/home/workbench/.local/bin/docker"]
KNOWN_CONTAINERS = ["local-ollama", "local-nim-llama", "local-hf"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _container_running(name: str) -> bool:
    result = _run(DOCKER + ["inspect", "--format", "{{.State.Running}}", name])
    return result.stdout.strip().lower() == "true"


def _container_port(name: str) -> Optional[str]:
    """Return the first exposed host port for a container."""
    result = _run(DOCKER + ["inspect", "--format", "{{json .NetworkSettings.Ports}}", name])
    if result.returncode != 0:
        return None
    try:
        ports = json.loads(result.stdout.strip())
        for bindings in ports.values():
            if bindings:
                return bindings[0].get("HostPort")
    except Exception:
        pass
    return None


def _fetch_models(host: str, port: str) -> list[str]:
    try:
        resp = requests.get(f"http://{host}:{port}/v1/models", timeout=5)
        if resp.ok:
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class OllamaPullRequest(BaseModel):
    model: str


class OllamaDeleteRequest(BaseModel):
    model: str


class LaunchRequest(BaseModel):
    profile: str
    hf_model: Optional[str] = None


class ChatRequest(BaseModel):
    host: str = "local-ollama"
    port: int = 8000
    model: str
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 1024


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(proxy_prefix: str = "") -> FastAPI:
    app = FastAPI(title="Model Manager", root_path=proxy_prefix)

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ------------------------------------------------------------------
    # Serve index.html at root
    # ------------------------------------------------------------------
    @app.get("/")
    async def index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    @app.get("/api/status")
    async def api_status():
        containers = {}
        for name in KNOWN_CONTAINERS:
            running = _container_running(name)
            models: list[str] = []
            if running:
                port = _container_port(name) or "8000"
                models = _fetch_models(name, port)
            containers[name] = {"running": running, "models": models}

        # Chain server health
        chain_healthy = False
        try:
            r = requests.get("http://localhost:8000/health", timeout=3)
            chain_healthy = r.ok
        except Exception:
            pass

        return JSONResponse({
            "containers": containers,
            "chain_server": {"healthy": chain_healthy},
        })

    # ------------------------------------------------------------------
    # GPU info
    # ------------------------------------------------------------------
    @app.get("/api/gpu")
    async def api_gpu():
        query = "index,name,memory.total,memory.used,memory.free,utilization.gpu"
        result = _run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            timeout=10,
        )
        gpus = []
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpus.append({
                        "index": parts[0],
                        "name": parts[1],
                        "total_mb": int(parts[2]) if parts[2].isdigit() else 0,
                        "used_mb": int(parts[3]) if parts[3].isdigit() else 0,
                        "free_mb": int(parts[4]) if parts[4].isdigit() else 0,
                        "utilization": int(parts[5]) if parts[5].isdigit() else 0,
                    })
        return JSONResponse({"gpus": gpus})

    # ------------------------------------------------------------------
    # Ollama models
    # ------------------------------------------------------------------
    @app.get("/api/ollama/models")
    async def api_ollama_models():
        result = _run(DOCKER + ["exec", "local-ollama", "ollama", "list"])
        models = []
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            # Skip header line
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 4:
                    # NAME  ID  SIZE  MODIFIED...
                    name = parts[0]
                    size = parts[2] + " " + parts[3] if len(parts) > 3 else parts[2]
                    modified = " ".join(parts[4:]) if len(parts) > 4 else ""
                    models.append({"name": name, "size": size, "modified": modified})
        return JSONResponse({"models": models, "error": result.stderr if result.returncode != 0 else ""})

    # ------------------------------------------------------------------
    # Ollama pull (background)
    # ------------------------------------------------------------------
    @app.post("/api/ollama/pull")
    async def api_ollama_pull(req: OllamaPullRequest):
        cmd = DOCKER + ["exec", "local-ollama", "ollama", "pull", req.model]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return JSONResponse({"status": "pulling", "model": req.model})

    # ------------------------------------------------------------------
    # Ollama delete
    # ------------------------------------------------------------------
    @app.post("/api/ollama/delete")
    async def api_ollama_delete(req: OllamaDeleteRequest):
        result = _run(DOCKER + ["exec", "local-ollama", "ollama", "rm", req.model])
        return JSONResponse({
            "status": "deleted" if result.returncode == 0 else "error",
            "output": result.stdout + result.stderr,
        })

    # ------------------------------------------------------------------
    # HuggingFace cache
    # ------------------------------------------------------------------
    @app.get("/api/hf/cache")
    async def api_hf_cache():
        result = _run(DOCKER + ["exec", "local-hf", "ls", "/root/.cache/huggingface/hub/"])
        entries = []
        if result.returncode == 0:
            for item in result.stdout.strip().splitlines():
                item = item.strip()
                if item:
                    # Convert "models--org--name" -> "org/name"
                    readable = re.sub(r"^models--", "", item).replace("--", "/")
                    entries.append(readable)
        return JSONResponse({
            "models": entries,
            "error": result.stderr if result.returncode != 0 else "",
        })

    # ------------------------------------------------------------------
    # Launch model
    # ------------------------------------------------------------------
    @app.post("/api/launch")
    async def api_launch(req: LaunchRequest):
        script = "/project/code/scripts/launch-model.sh"
        cmd = [script, req.profile]
        if req.hf_model:
            cmd.append(req.hf_model)
        result = _run(cmd, timeout=30)
        return JSONResponse({
            "status": "launched" if result.returncode == 0 else "error",
            "output": result.stdout + result.stderr,
        })

    # ------------------------------------------------------------------
    # Stop models
    # ------------------------------------------------------------------
    @app.post("/api/stop")
    async def api_stop():
        script = "/project/code/scripts/launch-model.sh"
        result = _run([script, "stop"], timeout=30)
        return JSONResponse({
            "status": "stopped" if result.returncode == 0 else "error",
            "output": result.stdout + result.stderr,
        })

    # ------------------------------------------------------------------
    # Container logs
    # ------------------------------------------------------------------
    @app.get("/api/logs/{container}")
    async def api_logs(container: str):
        # Validate container name to prevent injection
        if container not in KNOWN_CONTAINERS:
            return JSONResponse({"error": "Unknown container"}, status_code=400)
        result = _run(DOCKER + ["logs", "--tail", "100", container])
        return JSONResponse({
            "logs": result.stdout + result.stderr,
        })

    # ------------------------------------------------------------------
    # Chat (non-streaming)
    # ------------------------------------------------------------------
    @app.post("/api/chat")
    async def api_chat(req: ChatRequest):
        url = f"http://{req.host}:{req.port}/v1/chat/completions"
        payload = {
            "model": req.model,
            "messages": req.messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            return JSONResponse(resp.json(), status_code=resp.status_code)
        except requests.exceptions.RequestException as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)

    # ------------------------------------------------------------------
    # Chat streaming (SSE)
    # ------------------------------------------------------------------
    @app.post("/api/chat/stream")
    async def api_chat_stream(req: ChatRequest):
        url = f"http://{req.host}:{req.port}/v1/chat/completions"
        payload = {
            "model": req.model,
            "messages": req.messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": True,
        }

        def generate() -> Generator[str, None, None]:
            try:
                with requests.post(url, json=payload, stream=True, timeout=120) as resp:
                    for line in resp.iter_lines():
                        if line:
                            decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                            if decoded.startswith("data: "):
                                yield decoded + "\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )

    return app
