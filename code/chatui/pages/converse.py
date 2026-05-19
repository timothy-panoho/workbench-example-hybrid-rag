# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### This module contains the chatui gui for having a conversation. ###

import functools
import logging
from typing import Any, Dict, List, Tuple, Union
from pathlib import Path

import gradio as gr
import json
import requests
import shutil
import os
import subprocess
import time
# import torch
import tiktoken
import fnmatch
import traceback

from chatui import assets, chat_client
from chatui.pages import info
from chatui.pages import utils

from datetime import datetime
import uuid

_LOGGER = logging.getLogger(__name__)
PATH = "/"
TITLE = "Hybrid RAG: Chat UI"
OUTPUT_TOKENS = 1024
MAX_DOCS = 5

_HISTORY_FILE = "/project/chat_history.json"


def _save_history(history: list, metrics: dict) -> None:
    """Persist chat history to disk so sessions can be resumed."""
    try:
        with open(_HISTORY_FILE, "w") as f:
            json.dump({"history": history, "metrics": metrics}, f)
    except Exception:
        pass


def _load_history() -> tuple:
    """Load saved chat history from disk. Returns (history, metrics) or ([], {})."""
    try:
        with open(_HISTORY_FILE) as f:
            data = json.load(f)
        return data.get("history", []), data.get("metrics", {})
    except Exception:
        return [], {}


_SESSIONS_DIR = "/project/chat_sessions"
_CURRENT_SESSION_FILE = "/project/current_session"


def _ensure_sessions_dir() -> None:
    os.makedirs(_SESSIONS_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    return os.path.join(_SESSIONS_DIR, f"{session_id}.json")


def _save_session(session_id: str, name: str, history: list, metrics: dict, documents: list) -> None:
    _ensure_sessions_dir()
    try:
        existing = _load_session_data(session_id) or {}
        data = {
            "id": session_id,
            "name": name,
            "created_at": existing.get("created_at", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat(),
            "history": history,
            "metrics": metrics,
            "documents": documents,
        }
        with open(_session_path(session_id), "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _load_session_data(session_id: str) -> dict:
    try:
        with open(_session_path(session_id)) as f:
            return json.load(f)
    except Exception:
        return {}


def _delete_session_file(session_id: str) -> None:
    try:
        path = _session_path(session_id)
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _list_sessions() -> list:
    _ensure_sessions_dir()
    sessions = []
    try:
        for fname in sorted(os.listdir(_SESSIONS_DIR), reverse=True):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(_SESSIONS_DIR, fname)) as f:
                        sessions.append(json.load(f))
                except Exception:
                    pass
    except Exception:
        pass
    return sessions


def _get_current_session_id() -> str:
    try:
        with open(_CURRENT_SESSION_FILE) as f:
            return f.read().strip()
    except Exception:
        return ""


def _set_current_session_id(session_id: str) -> None:
    try:
        with open(_CURRENT_SESSION_FILE, "w") as f:
            f.write(session_id)
    except Exception:
        pass


def _new_session_id() -> str:
    return str(uuid.uuid4())[:8]


def _session_choices() -> list:
    sessions = _list_sessions()
    choices = []
    for s in sessions:
        date_part = s.get("updated_at", s.get("created_at", ""))[:10]
        label = f"{s.get('name', 'Chat')}  ({date_part})"
        choices.append((label, s["id"]))
    return choices


def _migrate_legacy_history() -> str:
    _ensure_sessions_dir()
    if _list_sessions():
        sid = _get_current_session_id()
        return sid if sid else (_list_sessions()[0]["id"] if _list_sessions() else "")
    legacy_history, legacy_metrics = _load_history()
    sid = _new_session_id()
    _save_session(sid, "Restored Chat", legacy_history, legacy_metrics, [])
    _set_current_session_id(sid)
    return sid


### Load in CSS here for components that need custom styling. ###

_LOCAL_CSS = """
#contextbox {
    overflow-y: scroll !important;
    max-height: 650px;
}
#rag-inputs .svelte-1gfkn6j {
    color: #76b900;
}
#token-counter {
    font-size: 0.82em;
    opacity: 0.72;
    padding: 2px 0 0 2px;
}
#secondary-btns button {
    font-size: 0.85em;
}
#sidebar-col {
    border-right: 1px solid rgba(255,255,255,0.08);
    padding-right: 6px;
}
#session-list .wrap {
    gap: 2px !important;
}
#session-list label {
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 0.87em;
    cursor: pointer;
}
#session-list label:hover {
    background: rgba(118,185,0,0.12);
}
#hamburger-btn {
    min-width: 40px !important;
    max-width: 44px !important;
}
"""

def build_page(client: chat_client.ChatClient) -> gr.Blocks:
    """
    Build the gradio page to be mounted in the frame.

    Parameters:
        client (chat_client.ChatClient): The chat client running the application.

    Returns:
        page (gr.Blocks): A Gradio page.
    """
    kui_theme, kui_styles = assets.load_theme("kaizen")

    with gr.Blocks(title=TITLE, theme=kui_theme, css=kui_styles + _LOCAL_CSS) as page:

        # ── Header: hamburger + title ─────────────────────────────────────────
        with gr.Row(equal_height=False):
            hamburger_btn = gr.Button("☰", size="sm", scale=0, min_width=40, elem_id="hamburger-btn")
            gr.Markdown(f"# {TITLE}")

        # State
        which_nim_tab = gr.State(0)
        is_local_nim = gr.State(False)
        vdb_active = gr.State(False)
        metrics_history = gr.State({})
        docs_history = gr.State({})
        sidebar_visible = gr.State(True)
        current_session_id = gr.State(_get_current_session_id())

        # ── Main layout ───────────────────────────────────────────────────────
        with gr.Row(equal_height=True):

            # ── LEFT SIDEBAR ──────────────────────────────────────────────────
            with gr.Column(scale=3, min_width=240, visible=True, elem_id="sidebar-col") as sidebar_col:

                # Session management
                new_chat_btn = gr.Button("＋ New Chat", variant="primary", size="sm")
                gr.Markdown("#### Recent Chats")
                session_radio = gr.Radio(
                    choices=_session_choices(),
                    value=_get_current_session_id() or None,
                    label="",
                    interactive=True,
                    elem_id="session-list",
                )
                delete_session_btn = gr.Button("🗑 Delete this chat", size="sm", variant="stop")

                gr.Markdown("---")

                # ── Generation Parameters ─────────────────────────────────────
                with gr.Accordion("⚙ Parameters", open=False, elem_id="accordion"):
                    gr.Markdown("Changes take effect on the next message.")
                    num_token_slider = gr.Slider(
                        0, utils.preset_max_tokens()[1],
                        value=utils.preset_max_tokens()[0],
                        label="Max Tokens in Response",
                        info="Increase if responses are cut off. 1 token ≈ ¾ of a word.",
                        interactive=True,
                    )
                    temp_slider = gr.Slider(
                        0, 1, value=0.7, label="Temperature",
                        info="Higher = more creative. Lower = focused. 0.7 is a good default.",
                        interactive=True,
                    )
                    top_p_slider = gr.Slider(
                        0.001, 0.999, value=0.999, label="Top P",
                        info="Leave at 0.999 and tune Temperature instead.",
                        interactive=True,
                    )
                    freq_pen_slider = gr.Slider(
                        -2, 2, value=0, label="Frequency Penalty",
                        info="0.3–0.8 reduces repetition.",
                        interactive=True,
                    )
                    pres_pen_slider = gr.Slider(
                        -2, 2, value=0, label="Presence Penalty",
                        info="0.3–0.8 encourages new topics.",
                        interactive=True,
                    )

                # ── Model / Inference ─────────────────────────────────────────
                with gr.Accordion("🤖 Model", open=True, elem_id="accordion"):
                    with gr.Column(visible=True) as setup_group:
                        gr.Markdown(info.setup)
                        rag_start_button = gr.Button(value="Set Up RAG Backend", variant="primary")

                    inference_mode = gr.Radio(
                        ["Local System", "Cloud Endpoint", "Self-Hosted Microservice"],
                        label="Inference Mode", info=info.inf_mode_info,
                        value="Self-Hosted Microservice", visible=False,
                    )

                    with gr.Tabs(selected=2, visible=False) as tabs:
                        with gr.TabItem("Local System", id=0, interactive=False, visible=False) as local:
                            with gr.Accordion("Prerequisites", open=True, elem_id="accordion"):
                                gr.Markdown(info.local_prereqs)
                            with gr.Accordion("Instructions", open=False, elem_id="accordion"):
                                gr.Markdown(info.local_info)
                            with gr.Accordion("Troubleshooting", open=False, elem_id="accordion"):
                                gr.Markdown(info.local_trouble)
                            gate_checkbox = gr.CheckboxGroup(
                                ["Ungated Models", "Gated Models"], value=["Ungated Models"],
                                label="Select which models types to show", interactive=True, elem_id="rag-inputs")
                            local_model_id = gr.Dropdown(
                                choices=["nvidia/Llama3-ChatQA-1.5-8B", "microsoft/Phi-3-mini-128k-instruct"],
                                value="nvidia/Llama3-ChatQA-1.5-8B", interactive=True,
                                label="Select a model (or input your own).", allow_custom_value=True, elem_id="rag-inputs")
                            local_model_quantize = gr.Dropdown(
                                choices=["None", "8-Bit", "4-Bit"], value=utils.preset_quantization(),
                                interactive=True, label="Select model quantization.", elem_id="rag-inputs")
                            with gr.Row(equal_height=True):
                                download_model = gr.Button(value="Load Model", size="sm")
                                start_local_server = gr.Button(value="Start Server", interactive=False, size="sm")
                                stop_local_server = gr.Button(value="Stop Server", interactive=False, size="sm")

                        with gr.TabItem("Cloud Endpoint", id=1, interactive=False, visible=False) as cloud:
                            with gr.Accordion("Prerequisites", open=True, elem_id="accordion"):
                                gr.Markdown(info.cloud_prereqs)
                            with gr.Accordion("Instructions", open=False, elem_id="accordion"):
                                gr.Markdown(info.cloud_info)
                            with gr.Accordion("Troubleshooting", open=False, elem_id="accordion"):
                                gr.Markdown(info.cloud_trouble)
                            nvcf_model_family = gr.Dropdown(
                                choices=["Select", "NVIDIA", "MistralAI", "Meta", "Google",
                                         "Microsoft", "Upstage", "AI21 Labs"],
                                value="Select", interactive=True,
                                label="Select a model family.", elem_id="rag-inputs")
                            nvcf_model_id = gr.Dropdown(
                                choices=["Select"], value="Select",
                                interactive=True,
                                label="Select a model.", visible=False, elem_id="rag-inputs")

                        with gr.TabItem("Self-Hosted Microservice", id=2, interactive=False, visible=False) as microservice:
                            with gr.Accordion("Prerequisites", open=True, elem_id="accordion"):
                                gr.Markdown(info.nim_prereqs)
                            with gr.Accordion("Instructions", open=False, elem_id="accordion"):
                                gr.Markdown(info.nim_info)
                            with gr.Accordion("Troubleshooting", open=False, elem_id="accordion"):
                                gr.Markdown(info.nim_trouble)
                            remote_nim_msg = gr.Markdown("<br />Enter the details below. Then start chatting!")
                            with gr.Row(equal_height=True):
                                nim_model_ip = gr.Textbox(value="localhost",
                                    label="Microservice Host",
                                    info="localhost, a Docker container name, or a remote IP address",
                                    elem_id="rag-inputs", scale=2)
                                nim_model_port = gr.Textbox(value="8000",
                                    label="Port", info="Optional, (default: 8000)",
                                    elem_id="rag-inputs", scale=1)
                            nim_model_id = gr.Dropdown(
                                choices=[
                                    "meta/llama-3.2-3b-instruct",
                                    "gemma3:4b", "qwen3:4b",
                                    "google/gemma-3-4b-it", "Qwen/Qwen3-4B",
                                    "meta-llama/Llama-3.2-3B-Instruct",
                                ],
                                value="meta/llama-3.2-3b-instruct",
                                label="Model running in microservice.",
                                info="NIM/vLLM: org/name — Ollama: name:tag. Must match the active compose profile.",
                                allow_custom_value=True, elem_id="rag-inputs")
                            with gr.Row():
                                fetch_models_btn = gr.Button("🔍 Fetch Models", size="sm", scale=1, variant="secondary")
                                fetch_models_status = gr.Textbox(
                                    value="", show_label=False, interactive=False,
                                    scale=3, max_lines=1,
                                    placeholder="← click to load models from the running microservice")

                # ── Local Model Launcher ──────────────────────────────────────
                with gr.Accordion("🚀 Launcher", open=False, elem_id="accordion"):
                    gr.Markdown("### 🚀 Local Model Launcher\nStart a local model container without leaving Workbench.")
                    with gr.Row():
                        launch_profile_dd = gr.Dropdown(
                            choices=[
                                "ollama — Ollama  (any model — manage below)",
                                "llama  — NIM     (meta/llama-3.2-3b-instruct)",
                                "hf     — vLLM    (HuggingFace — pick model below)",
                            ],
                            value="ollama — Ollama  (any model — manage below)",
                            label="Model profile", scale=3,
                        )
                        launch_model_btn = gr.Button("▶ Launch", variant="primary", scale=1)
                        stop_model_btn = gr.Button("⏹ Stop", variant="secondary", scale=1)
                    launch_status = gr.Textbox(
                        value="", label="Status", interactive=False,
                        placeholder="No model started yet — click Launch to begin.",
                    )
                    gr.Markdown("---\n### Model Manager")
                    with gr.Group(visible=True) as ollama_group:
                        gr.Markdown("Ollama holds **all downloaded models in one container** — switch instantly. Pull new models on demand.")
                        with gr.Row():
                            ollama_model_dd = gr.Dropdown(
                                choices=[], value=None, label="Downloaded models",
                                allow_custom_value=True, interactive=True, scale=3,
                            )
                            refresh_ollama_btn = gr.Button("🔄 Refresh", size="sm", scale=1)
                            use_ollama_model_btn = gr.Button("▶ Use This Model", variant="primary", size="sm", scale=1)
                        with gr.Row():
                            pull_model_input = gr.Textbox(
                                label="Pull a new model",
                                placeholder="e.g.  llama3.2:3b  |  qwen2.5:7b  |  mistral:7b",
                                info="Pulls into the running Ollama container",
                                interactive=True, scale=3,
                            )
                            pull_model_btn = gr.Button("⬇ Pull", variant="secondary", scale=1)
                    with gr.Group(visible=False) as nim_group:
                        gr.Markdown(
                            "**NIM containers serve one optimised model** — unlike Ollama you cannot swap models inside a running NIM.\n\n"
                            "| | |\n|---|---|\n"
                            "| **Container** | `local-nim-llama` |\n"
                            "| **Model** | `meta/llama-3.2-3b-instruct` |\n"
                            "| **Engine** | TensorRT-LLM FP8 (RTX 4090 optimised) |\n\n"
                            "To use a different model, stop this container and select a different profile."
                        )
                    with gr.Group(visible=False) as hf_group:
                        hf_model_input = gr.Textbox(
                            value="google/gemma-3-4b-it", label="HuggingFace Model ID",
                            info="Add HUGGING_FACE_HUB_TOKEN to .env for gated models",
                            placeholder="e.g. google/gemma-3-4b-it  |  Qwen/Qwen3-4B",
                            interactive=True,
                        )
                        gr.Markdown("**Downloaded models in vLLM cache**")
                        with gr.Row():
                            refresh_hf_btn = gr.Button("🔄 Refresh Cache", size="sm", scale=1)
                        hf_cache_output = gr.Textbox(
                            value="", label="Cached models", interactive=False, lines=4, max_lines=6,
                        )
                    with gr.Accordion("Container Logs", open=False, elem_id="accordion"):
                        gr.Markdown("Live tail of the running model container.")
                        with gr.Row():
                            refresh_logs_btn = gr.Button("🔄 Refresh Logs", size="sm", scale=1)
                            log_container_dd = gr.Dropdown(
                                choices=["local-ollama", "local-nim-llama", "local-hf"],
                                value="local-ollama", label="Container", scale=2,
                            )
                        log_output = gr.Textbox(
                            value="", label="", interactive=False,
                            lines=12, max_lines=12, show_copy_button=True,
                        )

                # ── Documents ─────────────────────────────────────────────────
                with gr.Accordion("📄 Documents", open=False, elem_id="accordion"):
                    gr.Markdown(info.update_kb_info)
                    file_output = gr.File(
                        interactive=True, show_label=False,
                        file_types=["text", ".pdf", ".html", ".doc", ".docx",
                                    ".txt", ".odt", ".rtf", ".tex"],
                        file_count="multiple")
                    with gr.Row():
                        doc_show = gr.Button(value="Show Documents", size="sm")
                        doc_hide = gr.Button(value="Hide Documents", visible=False, size="sm")
                        clear_docs = gr.Button(value="Clear Database", interactive=True, size="sm")
                    gr.Markdown("**Uploaded in this session:**")
                    session_docs_md = gr.Markdown("*(none yet)*")

            # ── MAIN CHAT AREA ────────────────────────────────────────────────
            with gr.Column(scale=17, min_width=400):

                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=400):
                        chatbot = gr.Chatbot(show_label=False, height=640)

                    context = gr.JSON(
                        scale=1, label="Retrieved Context",
                        visible=False, elem_id="contextbox",
                    )
                    metrics = gr.JSON(
                        scale=1, label="Metrics",
                        visible=False, elem_id="contextbox",
                    )
                    docs = gr.JSON(
                        scale=1, label="Documents",
                        visible=False, elem_id="contextbox",
                    )

                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=200):
                        msg = gr.Textbox(
                            show_label=False, lines=3,
                            placeholder="Enter text and press SUBMIT",
                            container=False, interactive=True,
                        )
                    with gr.Column(scale=1, min_width=100):
                        kb_checkbox = gr.CheckboxGroup(
                            ["Toggle to use Vector Database"],
                            label="Vector Database",
                            info="Supply your uploaded documents to the chatbot",
                        )

                with gr.Row():
                    token_counter_md = gr.Markdown(
                        "✏️ Prompt: **0 tokens**",
                        elem_id="token-counter",
                    )

                with gr.Row():
                    submit_btn = gr.Button(value="[NOT READY] Submit", interactive=False, scale=4, variant="primary")
                    clear_btn  = gr.Button(value="🗑 Clear", scale=1, size="sm")

                with gr.Row(elem_id="secondary-btns"):
                    mtx_show = gr.Button(value="📊 Show Metrics",  size="sm", scale=1)
                    mtx_hide = gr.Button(value="📊 Hide Metrics",  size="sm", scale=1, visible=False)
                    ctx_show = gr.Button(value="📄 Show Context",  size="sm", scale=1)
                    ctx_hide = gr.Button(value="📄 Hide Context",  size="sm", scale=1, visible=False)

                with gr.Accordion("📊 What do the metrics mean?", open=False, elem_id="accordion"):
                    gr.Markdown(info.metrics_guide)

        # ── Event handlers ────────────────────────────────────────────────────

        def _toggle_gated(models: List[str]) -> Dict[gr.component, Dict[Any, Any]]:
            """" Event listener to toggle local models displayed to the user. """
            if len(models) == 0:
                choices = []
                selected = ""
            elif len(models) == 1 and models[0] == "Ungated Models":
                choices = ["nvidia/Llama3-ChatQA-1.5-8B",
                           "microsoft/Phi-3-mini-128k-instruct"]
                selected = "nvidia/Llama3-ChatQA-1.5-8B"
            elif len(models) == 1 and models[0] == "Gated Models":
                choices = ["mistralai/Mistral-7B-Instruct-v0.1",
                           "mistralai/Mistral-7B-Instruct-v0.2",
                           "meta-llama/Llama-2-7b-chat-hf",
                           "meta-llama/Meta-Llama-3-8B-Instruct"]
                selected = "mistralai/Mistral-7B-Instruct-v0.1"
            else:
                choices = ["nvidia/Llama3-ChatQA-1.5-8B",
                           "microsoft/Phi-3-mini-128k-instruct",
                           "mistralai/Mistral-7B-Instruct-v0.1",
                           "mistralai/Mistral-7B-Instruct-v0.2",
                           "meta-llama/Llama-2-7b-chat-hf",
                           "meta-llama/Meta-Llama-3-8B-Instruct"]
                selected = "nvidia/Llama3-ChatQA-1.5-8B"
            return {
                local_model_id: gr.update(choices=choices, value=selected),
            }

        gate_checkbox.change(_toggle_gated, [gate_checkbox], [local_model_id])

        def _toggle_info(btn: str) -> Dict[gr.component, Dict[Any, Any]]:
            """" Event listener to toggle context and/or metrics panes visible to the user. """
            if "Show Context" in btn:
                out = [True, False, False, False, True, True, False, True, False]
            elif "Hide Context" in btn:
                out = [False, False, False, True, False, True, False, True, False]
            elif "Show Metrics" in btn:
                out = [False, True, False, True, False, False, True, True, False]
            elif "Hide Metrics" in btn:
                out = [False, False, False, True, False, True, False, True, False]
            elif "Show Documents" in btn:
                out = [False, False, True, True, False, True, False, False, True]
            elif "Hide Documents" in btn:
                out = [False, False, False, True, False, True, False, True, False]
            else:
                # fallback: no-op — hide all panels, show all "show" buttons
                out = [False, False, False, True, False, True, False, True, False]
            return {
                context: gr.update(visible=out[0]),
                metrics: gr.update(visible=out[1]),
                docs: gr.update(visible=out[2]),
                ctx_show: gr.update(visible=out[3]),
                ctx_hide: gr.update(visible=out[4]),
                mtx_show: gr.update(visible=out[5]),
                mtx_hide: gr.update(visible=out[6]),
                doc_show: gr.update(visible=out[7]),
                doc_hide: gr.update(visible=out[8]),
            }

        ctx_show.click(_toggle_info, [ctx_show], [context, metrics, docs, ctx_show, ctx_hide, mtx_show, mtx_hide, doc_show, doc_hide])
        ctx_hide.click(_toggle_info, [ctx_hide], [context, metrics, docs, ctx_show, ctx_hide, mtx_show, mtx_hide, doc_show, doc_hide])
        mtx_show.click(_toggle_info, [mtx_show], [context, metrics, docs, ctx_show, ctx_hide, mtx_show, mtx_hide, doc_show, doc_hide])
        mtx_hide.click(_toggle_info, [mtx_hide], [context, metrics, docs, ctx_show, ctx_hide, mtx_show, mtx_hide, doc_show, doc_hide])
        doc_show.click(_toggle_info, [doc_show], [context, metrics, docs, ctx_show, ctx_hide, mtx_show, mtx_hide, doc_show, doc_hide])
        doc_hide.click(_toggle_info, [doc_hide], [context, metrics, docs, ctx_show, ctx_hide, mtx_show, mtx_hide, doc_show, doc_hide])

        def _toggle_model_download(btn: str, model: str, start: str, stop: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to download model weights locally for Hugging Face TGI local inference. """
            if model != "nvidia/Llama3-ChatQA-1.5-8B" and model != "microsoft/Phi-3-mini-128k-instruct" and model != "" and os.environ.get('HUGGING_FACE_HUB_TOKEN') is None:
                gr.Warning("You are accessing a gated model and HUGGING_FACE_HUB_TOKEN is not detected!")
                return {
                    download_model: gr.update(),
                    start_local_server: gr.update(),
                    stop_local_server: gr.update(),
                }
            else:
                if btn == "Load Model":
                    progress(0.25, desc="Initializing Task")
                    time.sleep(0.75)
                    progress(0.5, desc="Downloading Model (may take a few moments)")
                    rc = subprocess.call("/bin/bash /project/code/scripts/download-local.sh " + model, shell=True)
                    if rc == 0:
                        msg_val = "Model Downloaded"
                        colors = "primary"
                        interactive = False
                        start_interactive = True if (start == "Start Server") else False
                        stop_interactive = True if (stop == "Stop Server") else False
                    else:
                        msg_val = "Error, Try Again"
                        colors = "stop"
                        interactive = True
                        start_interactive = False
                        stop_interactive = False
                progress(0.75, desc="Cleaning Up")
                time.sleep(0.75)
                return {
                    download_model: gr.update(value=msg_val, variant=colors, interactive=interactive),
                    start_local_server: gr.update(interactive=start_interactive),
                    stop_local_server: gr.update(interactive=stop_interactive),
                }

        download_model.click(_toggle_model_download,
                             [download_model, local_model_id, start_local_server, stop_local_server],
                             [download_model, start_local_server, stop_local_server, msg])

        def _toggle_model_select(model: str, start: str, stop: str) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to select different models to use for Hugging Face TGI local inference. """
            if model != "nvidia/Llama3-ChatQA-1.5-8B" and model != "microsoft/Phi-3-mini-128k-instruct" and model != "" and os.environ.get('HUGGING_FACE_HUB_TOKEN') is None:
                gr.Warning("You are accessing a gated model and HUGGING_FACE_HUB_TOKEN is not detected!")
            return {
                download_model: gr.update(value="Load Model",
                                          variant="secondary",
                                          interactive=(False if start == "Server Started" else True)),
                start_local_server: gr.update(interactive=False),
                stop_local_server: gr.update(interactive=(False if stop == "Server Stopped" else True)),
            }

        local_model_id.change(_toggle_model_select,
                              [local_model_id, start_local_server, stop_local_server],
                              [download_model, start_local_server, stop_local_server])

        def _toggle_nvcf_family(family: str) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to select a different family of model for cloud endpoint inference. """
            interactive = True
            submit_value = "Submit"
            msg_value = "Enter text and press SUBMIT"
            if family == "NVIDIA":
                choices = ["Llama3 ChatQA-1.5 8B",
                           "Llama3 ChatQA-1.5 70B",
                           "Nemotron Mini 4B",
                           "Nemotron-4 340B Instruct",
                           "Mistral-NeMo 12B Instruct"]
                value = "Llama3 ChatQA-1.5 8B"
                visible = True
            elif family == "MistralAI":
                choices = ["Mistral 7B Instruct v0.2",
                           "Mistral 7B Instruct v0.3",
                           "Mixtral 8x7B Instruct v0.1",
                           "Mixtral 8x22B Instruct v0.1",
                           "Mistral-NeMo 12B Instruct",
                           "Mamba Codestral 7B v0.1"]
                value = "Mistral 7B Instruct v0.2"
                visible = True
            elif family == "Meta":
                choices = ["Llama 3 8B",
                           "Llama 3 70B",
                           "Llama 3.1 8B",
                           "Llama 3.1 70B",
                           "Llama 3.1 405B"]
                value = "Llama 3 8B"
                visible = True
            elif family == "Google":
                choices = ["Gemma 2B", "Gemma 7B", "Code Gemma 7B"]
                value = "Gemma 2B"
                visible = True
            elif family == "Microsoft":
                choices = ["Phi-3 Mini (4k)",
                           "Phi-3 Mini (128k)",
                           "Phi-3 Small (8k)",
                           "Phi-3 Small (128k)",
                           "Phi-3 Medium (4k)",
                           "Phi-3 Medium (128k)",
                           "Phi-3.5 Mini Instruct",
                           "Phi-3.5 MoE Instruct"]
                value = "Phi-3 Mini (4k)"
                visible = True
            elif family == "Upstage":
                choices = ["Solar 10.7B Instruct"]
                value = "Solar 10.7B Instruct"
                visible = True
            elif family == "AI21 Labs":
                choices = ["Jamba-1.5 Mini Instruct", "Jamba-1.5 Large Instruct"]
                value = "Jamba-1.5 Mini Instruct"
                visible = True
            else:
                choices = ["Select"]
                value = "Select"
                visible = False
                interactive = False
                submit_value = "[NOT READY] Submit"
                msg_value = "[NOT READY] Select a model OR Select a Different Inference Mode."
            return {
                nvcf_model_id: gr.update(choices=choices, value=value, visible=visible),
                submit_btn: gr.update(value=submit_value, interactive=interactive),
                msg: gr.update(interactive=True,
                               placeholder=msg_value),
            }

        nvcf_model_family.change(_toggle_nvcf_family,
                              [nvcf_model_family],
                              [nvcf_model_id, submit_btn, msg])

        def _toggle_local_server(btn: str, model: str, quantize: str, download: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to run and/or shut down the Hugging Face TGI local inference server. """
            if model != "nvidia/Llama3-ChatQA-1.5-8B" and model != "microsoft/Phi-3-mini-128k-instruct" and model != "" and btn != "Stop Server" and os.environ.get('HUGGING_FACE_HUB_TOKEN') is None:
                gr.Warning("You are accessing a gated model and HUGGING_FACE_HUB_TOKEN is not detected!")
                return {
                    start_local_server: gr.update(),
                    stop_local_server: gr.update(),
                    msg: gr.update(),
                    submit_btn: gr.update(),
                    download_model: gr.update(),
                }
            else:
                if btn == "Start Server":
                    progress(0.2, desc="Initializing Task")
                    time.sleep(0.5)
                    progress(0.4, desc="Setting Up RAG Backend (one-time process, may take a few moments)")
                    rc = subprocess.call("/bin/bash /project/code/scripts/rag-consolidated.sh ", shell=True)
                    time.sleep(0.5)
                    progress(0.6, desc="Starting Inference Server (may take a few moments)")
                    rc = subprocess.call("/bin/bash /project/code/scripts/start-local.sh "
                                              + model + " " + utils.quant_to_config(quantize), shell=True)
                    if rc == 0:
                        out = ["Server Started", "Stop Server"]
                        colors = ["primary", "secondary"]
                        interactive = [False, True, True, False]
                    else:
                        gr.Warning("ERR: You may have timed out or are facing memory issues. In AI Workbench, check Output > Chat for details.")
                        out = ["Internal Server Error, Try Again", "Stop Server"]
                        colors = ["stop", "secondary"]
                        interactive = [False, True, False, False]
                    progress(0.8, desc="Cleaning Up")
                    time.sleep(0.5)
                elif btn == "Stop Server":
                    progress(0.25, desc="Initializing")
                    time.sleep(0.5)
                    progress(0.5, desc="Stopping Server")
                    rc = subprocess.call("/bin/bash /project/code/scripts/stop-local.sh", shell=True)
                    if rc == 0:
                        out = ["Start Server", "Server Stopped"]
                        colors = ["secondary", "primary"]
                        interactive = [True, False, False, False if (download == "Model Downloaded") else True]
                    else:
                        out = ["Start Server", "Internal Server Error, Try Again"]
                        colors = ["secondary", "stop"]
                        interactive = [True, False, True, False]
                    progress(0.75, desc="Cleaning Up")
                    time.sleep(0.5)
                return {
                    start_local_server: gr.update(value=out[0], variant=colors[0], interactive=interactive[0]),
                    stop_local_server: gr.update(value=out[1], variant=colors[1], interactive=interactive[1]),
                    msg: gr.update(interactive=True,
                                   placeholder=("Enter text and press SUBMIT" if interactive[2] else "[NOT READY] Start the Local Inference Server OR Select a Different Inference Mode.")),
                    submit_btn: gr.update(value="Submit" if interactive[2] else "[NOT READY] Submit", interactive=interactive[2]),
                    download_model: gr.update(interactive=interactive[3]),
                }

        start_local_server.click(_toggle_local_server,
                                 [start_local_server, local_model_id, local_model_quantize, download_model],
                                 [start_local_server, stop_local_server, msg, submit_btn, download_model])
        stop_local_server.click(_toggle_local_server,
                                 [stop_local_server, local_model_id, local_model_quantize, download_model],
                                 [start_local_server, stop_local_server, msg, submit_btn, download_model])

        def _lock_tabs(btn: str,
                       start_local_server: str,
                       which_nim_tab: int,
                       nvcf_model_family: str,
                       progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to lock settings options with the user selected inference mode. """
            if btn == "Local System":
                if start_local_server == "Server Started":
                    interactive = True
                else:
                    interactive = False
                return {
                    tabs: gr.update(selected=0),
                    msg: gr.update(interactive=True,
                                   placeholder=("Enter text and press SUBMIT" if interactive else "[NOT READY] Start the Local Inference Server OR Select a Different Inference Mode.")),
                    inference_mode: gr.update(info="To use your LOCAL GPU for inference, start the Local Inference Server before making a query."),
                    submit_btn: gr.update(value="Submit" if interactive else "[NOT READY] Submit", interactive=interactive),
                }
            elif btn == "Cloud Endpoint":
                if nvcf_model_family == "Select":
                    interactive = False
                else:
                    interactive = True
                return {
                    tabs: gr.update(selected=1),
                    msg: gr.update(interactive=True, placeholder=("Enter text and press SUBMIT" if interactive else "[NOT READY] Select a model OR Select a Different Inference Mode.")),
                    inference_mode: gr.update(info="To use a CLOUD endpoint for inference, select the desired model before making a query."),
                    submit_btn: gr.update(value="Submit" if interactive else "[NOT READY] Submit", interactive=interactive),
                }
            elif btn == "Self-Hosted Microservice":
                return {
                    tabs: gr.update(selected=2),
                    msg: gr.update(interactive=True, placeholder="Enter text and press SUBMIT" if (which_nim_tab == 0) else "[NOT READY] Start the Local Microservice OR Select a Different Inference Mode."),
                    inference_mode: gr.update(info="To use a MICROSERVICE for inference, input the endpoint (and/or model) before making a query."),
                    submit_btn: gr.update(value="Submit" if (which_nim_tab == 0) else "[NOT READY] Submit",
                                          interactive=True if (which_nim_tab == 0) else False),
                }

        inference_mode.change(_lock_tabs, [inference_mode, start_local_server, which_nim_tab, nvcf_model_family], [tabs, msg, inference_mode, submit_btn])

        def _toggle_kb(btn: str, docs_uploaded, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to clear the vector database of all documents. """
            if btn == "Clear Database":
                progress(0.25, desc="Initializing Task")
                update_docs_uploaded = docs_uploaded
                time.sleep(0.25)
                progress(0.5, desc="Clearing Vector Database")
                success = utils.clear_knowledge_base()
                if success:
                    out = ["Clear Database"]
                    colors = ["secondary"]
                    interactive = [True]
                    progress(0.75, desc="Success!")
                    for key, value in update_docs_uploaded.items():
                        update_docs_uploaded.update({str(key): "Deleted"})
                    time.sleep(0.5)
                else:
                    gr.Warning("Your files may still be present in the database. Try again.")
                    out = ["Error Clearing Vector Database"]
                    colors = ["stop"]
                    interactive = [True]
                    progress(0.75, desc="Error, try again.")
                    for key, value in update_docs_uploaded.items():
                        update_docs_uploaded.update({str(key): "Unknown"})
                    time.sleep(0.5)
            else:
                out = ["Clear Database"]
                colors = ["secondary"]
                interactive = [True]
            return {
                file_output: gr.update(value=None,
                                       interactive=True,
                                       show_label=False,
                                       file_types=["text",
                                                   ".pdf",
                                                   ".html",
                                                   ".doc",
                                                   ".docx",
                                                   ".txt",
                                                   ".odt",
                                                   ".rtf",
                                                   ".tex"],
                                       file_count="multiple"),
                clear_docs: gr.update(value=out[0], variant=colors[0], interactive=interactive[0]),
                kb_checkbox: gr.update(value=None),
                docs: gr.update(value=update_docs_uploaded),
                docs_history: update_docs_uploaded,
            }

        clear_docs.click(_toggle_kb, [clear_docs, docs_history], [clear_docs, file_output, kb_checkbox, msg, docs, docs_history])

        def _vdb_select(inf_mode: str, start_local: str, vdb_active: bool, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to select the vector database settings top-level tab. """
            progress(0.25, desc="Initializing Task")
            time.sleep(0.25)
            progress(0.5, desc="Awaiting Vector DB Readiness")
            rc = subprocess.call("/bin/bash /project/code/scripts/check-database.sh ", shell=True)
            if rc == 0:
                if not vdb_active:
                    gr.Info("The Vector Database is now ready for file upload. ")
                interactive = True
            else:
                gr.Warning("The Vector Database has timed out. Check Output > Chat on AI Workbench for the full logs. ")
                interactive = False
            progress(0.75, desc="Cleaning Up")
            time.sleep(0.25)
            return [True if rc == 0 else False,
                    gr.update(interactive=interactive),
                    gr.update(interactive=interactive)]

        def _document_upload(files, docs_uploaded, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to upload documents to the vector database. """
            progress(0.25, desc="Initializing Task")
            update_docs_uploaded = docs_uploaded
            time.sleep(0.25)
            progress(0.5, desc="Polling Vector DB Status")
            rc = subprocess.call("/bin/bash /project/code/scripts/check-database.sh ", shell=True)
            if rc == 0:
                progress(0.75, desc="Pushing uploaded files to DB...")
                file_paths = utils.upload_file(files, client)
                success = True
                for file in file_paths:
                    update_docs_uploaded.update({str(file.split('/')[-1]): "Uploaded Successfully"})
            else:
                gr.Warning("Hang Tight! The Vector DB may be temporarily busy. Give it a moment, and then try again. ")
                file_paths = None
                success = False
                file_names = [file.name for file in files]
                for file in file_names:
                    update_docs_uploaded.update({str(file.split('/')[-1]): "Failed to Upload"})

            # Track uploaded documents in the current session
            try:
                active_sid = _get_current_session_id()
                if active_sid and success and file_paths:
                    session = _load_session_data(active_sid) or {}
                    existing_docs = session.get("documents", [])
                    new_names = [fp.split('/')[-1] for fp in file_paths]
                    all_docs = list(dict.fromkeys(existing_docs + new_names))
                    _save_session(active_sid, session.get("name", "Chat"),
                                  session.get("history", []), session.get("metrics", {}), all_docs)
                    new_docs_text = "\n".join(f"• {d}" for d in all_docs)
                else:
                    new_docs_text = gr.update()
            except Exception:
                new_docs_text = gr.update()

            return {
                file_output: gr.update(value=file_paths),
                kb_checkbox: gr.update(value="Toggle to use Vector Database" if success else None),
                docs: gr.update(value=update_docs_uploaded),
                docs_history: update_docs_uploaded,
                session_docs_md: new_docs_text,
            }

        file_output.upload(_document_upload, [file_output, docs_history], [file_output, kb_checkbox, docs, docs_history, session_docs_md])

        def _toggle_rag_start(btn: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to initialize the RAG backend API server. """
            progress(0.25, desc="Initializing Task")
            time.sleep(0.25)
            progress(0.5, desc="Setting Up RAG Backend (one-time process, may take a few moments)")
            rc = subprocess.call("/bin/bash /project/code/scripts/rag-consolidated.sh ", shell=True)
            progress(0.75, desc="Cleaning Up")
            time.sleep(0.25)
            if rc in (0, 2):
                if rc == 2:
                    gr.Info("Chain server is ready. The Vector DB may still be warming up — your first query may be slow.")
                return {
                    setup_group: gr.update(visible=False),
                    inference_mode: gr.update(visible=True),
                    tabs: gr.update(visible=True),
                    submit_btn: gr.update(value="Submit", interactive=True),
                    msg: gr.update(interactive=True, placeholder="Enter text and press SUBMIT"),
                }
            else:
                gr.Warning("RAG backend failed to start. Ensure the NIM container is running, then try again.")
                return {
                    setup_group: gr.update(visible=True),
                    inference_mode: gr.update(visible=False),
                    submit_btn: gr.update(value="[NOT READY] Submit", interactive=False),
                    msg: gr.update(interactive=False, placeholder="[NOT READY] Ensure the NIM container is running, then click Set Up RAG Backend."),
                }

        rag_start_button.click(_toggle_rag_start, [rag_start_button], [setup_group, inference_mode, tabs, submit_btn, msg])

        # ── Session management ────────────────────────────────────────────────
        def _new_chat_cb():
            sid = _new_session_id()
            name = f"Chat {datetime.now().strftime('%b %d %H:%M')}"
            _save_session(sid, name, [], {}, [])
            _set_current_session_id(sid)
            choices = _session_choices()
            return (
                gr.update(value=[]),
                {},
                gr.update(value=None),
                gr.update(choices=choices, value=sid),
                sid,
                "*(none yet)*",
                "✏️ Prompt: **0 tokens**",
            )

        def _switch_session_cb(session_id):
            if not session_id:
                return gr.update(), {}, gr.update(), session_id, "*(none yet)*"
            _set_current_session_id(session_id)
            session = _load_session_data(session_id)
            saved_docs = session.get("documents", [])
            docs_text = "\n".join(f"• {d}" for d in saved_docs) if saved_docs else "*(none yet)*"
            return (
                gr.update(value=session.get("history", [])),
                session.get("metrics", {}),
                gr.update(value=None),
                session_id,
                docs_text,
            )

        def _delete_session_cb(session_id):
            if session_id:
                _delete_session_file(session_id)
            sessions = _list_sessions()
            choices = _session_choices()
            if sessions:
                new_id = sessions[0]["id"]
                _set_current_session_id(new_id)
                session = sessions[0]
                saved_docs = session.get("documents", [])
                docs_text = "\n".join(f"• {d}" for d in saved_docs) if saved_docs else "*(none yet)*"
                return (
                    gr.update(value=session.get("history", [])),
                    session.get("metrics", {}),
                    gr.update(value=None),
                    gr.update(choices=choices, value=new_id),
                    new_id,
                    docs_text,
                )
            sid = _new_session_id()
            name = f"Chat {datetime.now().strftime('%b %d %H:%M')}"
            _save_session(sid, name, [], {}, [])
            _set_current_session_id(sid)
            choices = _session_choices()
            return (
                gr.update(value=[]),
                {},
                gr.update(value=None),
                gr.update(choices=choices, value=sid),
                sid,
                "*(none yet)*",
            )

        new_chat_btn.click(
            _new_chat_cb, inputs=[],
            outputs=[chatbot, metrics_history, metrics, session_radio, current_session_id, session_docs_md, token_counter_md],
        )
        session_radio.change(
            _switch_session_cb, inputs=[session_radio],
            outputs=[chatbot, metrics_history, metrics, current_session_id, session_docs_md],
        )
        delete_session_btn.click(
            _delete_session_cb, inputs=[current_session_id],
            outputs=[chatbot, metrics_history, metrics, session_radio, current_session_id, session_docs_md],
        )

        # ── Hamburger sidebar toggle ───────────────────────────────────────────
        def _toggle_sidebar(is_visible: bool):
            new_state = not is_visible
            return gr.update(visible=new_state), new_state

        hamburger_btn.click(_toggle_sidebar, inputs=[sidebar_visible], outputs=[sidebar_col, sidebar_visible])

        # ── Clear history ─────────────────────────────────────────────────────
        def _clear_history_cb(session_id):
            try:
                if os.path.exists(_HISTORY_FILE):
                    os.remove(_HISTORY_FILE)
            except Exception:
                pass
            try:
                if session_id:
                    session = _load_session_data(session_id) or {}
                    _save_session(session_id, session.get("name", "Chat"),
                                  [], {}, session.get("documents", []))
            except Exception:
                pass
            return (
                gr.update(value=""),
                gr.update(value=[]),
                gr.update(value=None),
                {},
            )

        clear_btn.click(
            _clear_history_cb,
            inputs=[current_session_id],
            outputs=[msg, chatbot, metrics, metrics_history],
        )

        # form actions
        _my_build_stream = functools.partial(_stream_predict, client)
        msg.submit(
            _my_build_stream, [kb_checkbox,
                               inference_mode,
                               nvcf_model_id,
                               nim_model_ip,
                               nim_model_port,
                               nim_model_id,
                               is_local_nim,
                               num_token_slider,
                               temp_slider,
                               top_p_slider,
                               freq_pen_slider,
                               pres_pen_slider,
                               start_local_server,
                               local_model_id,
                               msg,
                               metrics_history,
                               chatbot], [msg, chatbot, context, metrics, metrics_history, token_counter_md]
        )
        submit_btn.click(
            _my_build_stream, [kb_checkbox,
                               inference_mode,
                               nvcf_model_id,
                               nim_model_ip,
                               nim_model_port,
                               nim_model_id,
                               is_local_nim,
                               num_token_slider,
                               temp_slider,
                               top_p_slider,
                               freq_pen_slider,
                               pres_pen_slider,
                               start_local_server,
                               local_model_id,
                               msg,
                               metrics_history,
                               chatbot], [msg, chatbot, context, metrics, metrics_history, token_counter_md]
        )

        def _auto_start(progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ On page load: start RAG backend and restore session. """
            progress(0.25, desc="Starting RAG Backend...")
            time.sleep(0.25)
            progress(0.5, desc="Waiting for Chain Server and Vector DB...")
            rc = subprocess.call("/bin/bash /project/code/scripts/rag-consolidated.sh", shell=True)
            progress(0.75, desc="Configuring UI...")
            time.sleep(0.25)

            active_sid = _migrate_legacy_history()
            if not active_sid:
                active_sid = _get_current_session_id()
            session = _load_session_data(active_sid) if active_sid else {}
            saved_history = session.get("history", [])
            saved_metrics = session.get("metrics", {})
            saved_docs = session.get("documents", [])
            choices = _session_choices()
            docs_text = "\n".join(f"• {d}" for d in saved_docs) if saved_docs else "*(none yet)*"
            fetched_ip, fetched_port, fetched_model, fetch_status = _fetch_models()

            common = {
                chatbot: gr.update(value=saved_history),
                metrics_history: saved_metrics,
                nim_model_ip: fetched_ip,
                nim_model_port: fetched_port,
                nim_model_id: fetched_model,
                fetch_models_status: gr.update(value=fetch_status),
                session_radio: gr.update(choices=choices, value=active_sid),
                current_session_id: active_sid,
                session_docs_md: gr.update(value=docs_text),
            }
            if rc in (0, 2):
                return {
                    setup_group: gr.update(visible=False),
                    inference_mode: gr.update(visible=True),
                    tabs: gr.update(visible=True),
                    submit_btn: gr.update(value="Submit", interactive=True),
                    msg: gr.update(interactive=True, placeholder="Enter text and press SUBMIT"),
                    **common,
                }
            else:
                return {
                    setup_group: gr.update(visible=True),
                    inference_mode: gr.update(visible=False),
                    submit_btn: gr.update(value="[NOT READY] Submit", interactive=False),
                    msg: gr.update(interactive=False, placeholder="[NOT READY] Start the NIM container, then click Set Up RAG Backend."),
                    **common,
                }

        page.load(_auto_start, None, [setup_group, inference_mode, tabs, submit_btn, msg, chatbot, metrics_history, nim_model_ip, nim_model_port, nim_model_id, fetch_models_status, session_radio, current_session_id, session_docs_md])

        # ── Live token counter (updates as the user types) ───────────────────
        def _update_token_count(text: str) -> str:
            if not text or not text.strip():
                return "✏️ Prompt: **0 tokens**"
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                count = len(enc.encode(text))
                return f"✏️ Prompt: **{count} token{'s' if count != 1 else ''}**"
            except Exception:
                return "✏️ Prompt: counting…"

        msg.change(_update_token_count, inputs=[msg], outputs=[token_counter_md])

        # ── Local Model Launcher ──────────────────────────────────────────────
        _PROFILE_MAP = {
            "ollama — Ollama  (any model — manage below)":        ("ollama", "local-ollama",   "8000", None),
            "llama  — NIM     (meta/llama-3.2-3b-instruct)":     ("llama",  "local-nim-llama", "8000", "meta/llama-3.2-3b-instruct"),
            "hf     — vLLM    (HuggingFace — pick model below)": ("hf",     "local-hf",        "8000", None),
        }

        def _on_profile_change(profile_label):
            """Show the right Model Manager panel for the selected profile."""
            profile = _PROFILE_MAP.get(profile_label, ("",))[0]
            return (
                gr.update(visible=profile == "ollama"),   # ollama_group
                gr.update(visible=profile == "llama"),    # nim_group
                gr.update(visible=profile == "hf"),       # hf_group
            )

        def _launch_model(profile_label, hf_model_id):
            """Start the selected model container and poll until it is serving requests."""
            profile, host, port, fixed_model = _PROFILE_MAP[profile_label]
            model = hf_model_id.strip() if profile == "hf" else fixed_model
            no_change = gr.update(), gr.update(), gr.update(), gr.update()

            yield gr.update(value=f"⏳ Starting {host}…"), *no_change

            cmd = ["/bin/bash", "/project/code/scripts/launch-model.sh", profile]
            if profile == "hf" and model:
                cmd.append(model)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                err = (result.stderr or result.stdout or "unknown error")[:300]
                yield gr.update(value=f"❌ {err}"), *no_change
                return

            try:
                with open("/project/.model-profile", "w") as _pf:
                    _pf.write(profile)
            except Exception:
                pass

            # Poll until the container's /v1/models endpoint responds
            url = f"http://{host}:{port}/v1/models"
            start = time.time()
            max_wait = 360
            poll_interval = 5
            while True:
                elapsed = int(time.time() - start)
                try:
                    resp = requests.get(url, timeout=2)
                    if resp.ok:
                        ids = [m["id"] for m in resp.json().get("data", [])]
                        display_model = ids[0] if ids else (model or "unknown")
                        status = f"✅ {host} ready ({elapsed}s) — model: {display_model}"
                        ollama_dd = gr.update()
                        if profile == "ollama":
                            ollama_dd, ollama_msg = _refresh_ollama_models()
                            status += f"\n{ollama_msg}"
                        elif profile == "hf":
                            status += "\n⏳ First run downloads the model — check Container Logs for progress."
                        yield (
                            gr.update(value=status),
                            gr.update(value=host),
                            gr.update(value=port),
                            gr.update(value=display_model),
                            ollama_dd,
                        )
                        return
                except Exception:
                    pass

                if elapsed >= max_wait:
                    yield gr.update(value=f"⚠️ {host} started but not responding after {max_wait}s — check Container Logs"), *no_change
                    return

                yield gr.update(value=f"⏳ Waiting for {host}… {elapsed}s  (NIM may take 2-3 min to load TensorRT engine)"), *no_change
                time.sleep(poll_interval)

        def _stop_model():
            result = subprocess.run(
                ["/bin/bash", "/project/code/scripts/launch-model.sh", "stop"],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                return gr.update(value="⏹ All local model containers stopped.")
            err = (result.stderr or result.stdout or "unknown error")[:200]
            return gr.update(value=f"❌ {err}")

        def _refresh_logs(container_name):
            """Tail the last 60 lines from the selected model container."""
            result = subprocess.run(
                ["sudo", "docker", "logs", "--tail", "60", container_name],
                capture_output=True, text=True, timeout=15,
            )
            output = (result.stdout or "") + (result.stderr or "")
            if not output.strip():
                return gr.update(value=f"(no output yet from '{container_name}' — is it running?)")
            return gr.update(value=output.strip())

        launch_profile_dd.change(
            _on_profile_change,
            inputs=[launch_profile_dd],
            outputs=[ollama_group, nim_group, hf_group],
        )
        launch_model_btn.click(
            _launch_model,
            inputs=[launch_profile_dd, hf_model_input],
            outputs=[launch_status, nim_model_ip, nim_model_port, nim_model_id, ollama_model_dd],
        )
        stop_model_btn.click(
            _stop_model,
            inputs=[],
            outputs=[launch_status],
        )
        refresh_logs_btn.click(
            _refresh_logs,
            inputs=[log_container_dd],
            outputs=[log_output],
        )

        # ── Ollama Model Manager ──────────────────────────────────────────────
        def _ollama_running():
            """Return True if local-ollama is running."""
            chk = subprocess.run(
                ["sudo", "docker", "inspect", "local-ollama", "--format", "{{.State.Running}}"],
                capture_output=True, text=True, timeout=5,
            )
            return chk.returncode == 0 and chk.stdout.strip() == "true"

        def _refresh_ollama_models():
            """List all models downloaded in the running Ollama container."""
            if not _ollama_running():
                return gr.update(choices=[], value=None), "❌ No Ollama container running — launch the ollama profile first"
            lst = subprocess.run(
                ["sudo", "docker", "exec", "local-ollama", "ollama", "list"],
                capture_output=True, text=True, timeout=10,
            )
            if lst.returncode == 0:
                lines = lst.stdout.strip().split("\n")
                models = [l.split()[0] for l in lines[1:] if l.strip()]
                if models:
                    return gr.update(choices=models, value=models[0]), f"✅ {len(models)} model(s) available"
                return gr.update(choices=[], value=None), "⚠️ Container running but no models yet — pull one below"
            return gr.update(choices=[], value=None), "❌ Could not query Ollama"

        def _use_ollama_model(model_name):
            """Update Inference Settings to point at the selected Ollama model."""
            if not model_name:
                return gr.update(), gr.update(), gr.update(), "❌ No model selected"
            if not _ollama_running():
                return gr.update(), gr.update(), gr.update(), "❌ No Ollama container running"
            return (
                gr.update(value="local-ollama"),
                gr.update(value="8000"),
                gr.update(value=model_name),
                f"✅ Inference settings updated → local-ollama:8000  /  {model_name}",
            )

        def _pull_ollama_model(model_name):
            """Pull a new model into the running Ollama container (runs in background)."""
            if not model_name.strip():
                return gr.update(value="❌ Enter a model name to pull")
            if not _ollama_running():
                return gr.update(value="❌ No Ollama container running — launch the ollama profile first")
            subprocess.Popen(["sudo", "docker", "exec", "local-ollama", "ollama", "pull", model_name.strip()])
            return gr.update(
                value=f"⏳ Pulling '{model_name}' — open Container Logs and click 🔄 Refresh to track progress. "
                      f"Click 🔄 Refresh above when done to add it to the model list."
            )

        # ── HuggingFace Cache Browser ─────────────────────────────────────────
        def _refresh_hf_cache():
            """List models already downloaded in the vLLM HF cache volume."""
            chk = subprocess.run(
                ["sudo", "docker", "inspect", "local-hf", "--format", "{{.State.Running}}"],
                capture_output=True, text=True, timeout=5,
            )
            if chk.returncode != 0 or chk.stdout.strip() != "true":
                return "", "❌ local-hf container not running — launch the hf profile first"
            ls = subprocess.run(
                ["sudo", "docker", "exec", "local-hf", "ls", "/root/.cache/huggingface/hub/"],
                capture_output=True, text=True, timeout=10,
            )
            if ls.returncode != 0:
                return "", "❌ Could not read HF cache"
            entries = [e for e in ls.stdout.strip().split("\n") if e.startswith("models--")]
            if not entries:
                return "", "✅ HF cache is empty — launch the hf profile and it will download on first use"
            # models--google--gemma-3-4b-it  →  google/gemma-3-4b-it
            readable = []
            for e in entries:
                parts = e.replace("models--", "").split("--")
                readable.append("/".join(parts) if len(parts) >= 2 else e)
            return "\n".join(readable), f"✅ {len(readable)} model(s) cached"

        refresh_ollama_btn.click(_refresh_ollama_models, inputs=[], outputs=[ollama_model_dd, launch_status])
        use_ollama_model_btn.click(_use_ollama_model, inputs=[ollama_model_dd], outputs=[nim_model_ip, nim_model_port, nim_model_id, launch_status])
        pull_model_btn.click(_pull_ollama_model, inputs=[pull_model_input], outputs=[launch_status])
        refresh_hf_btn.click(_refresh_hf_cache, inputs=[], outputs=[hf_cache_output, launch_status])

        # ── Fetch Models from running microservice ────────────────────────────
        def _fetch_models():
            """Auto-detect whichever local model container is running and populate host, port, and model."""
            candidates = [
                (profile, host, port)
                for profile, host, port, _ in _PROFILE_MAP.values()
            ]
            seen = set()
            unique_candidates = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    unique_candidates.append(c)

            starting_up = []
            for _profile, host, port in unique_candidates:
                # First check if the container is actually running
                chk = subprocess.run(
                    ["sudo", "docker", "inspect", host, "--format", "{{.State.Running}}"],
                    capture_output=True, text=True, timeout=5,
                )
                container_up = chk.returncode == 0 and chk.stdout.strip() == "true"

                url = f"http://{host}:{port}/v1/models"
                try:
                    resp = requests.get(url, timeout=3)
                    resp.raise_for_status()
                    ids = [m["id"] for m in resp.json().get("data", [])]
                    if not ids:
                        continue
                    return (
                        gr.update(value=host),
                        gr.update(value=port),
                        gr.update(choices=ids, value=ids[0]),
                        f"✅ {len(ids)} model(s) found at {host}:{port}",
                    )
                except Exception:
                    if container_up:
                        starting_up.append(host)

            if starting_up:
                names = ", ".join(starting_up)
                return (
                    gr.update(), gr.update(), gr.update(),
                    f"⏳ {names} is running but still loading — click 🔍 Fetch Models again in a moment.",
                )
            return (
                gr.update(), gr.update(), gr.update(),
                "❌ No local model containers running — use the Local Model Launcher tab to start one.",
            )

        _fetch_outputs = [nim_model_ip, nim_model_port, nim_model_id, fetch_models_status]

        fetch_models_btn.click(_fetch_models, inputs=[], outputs=_fetch_outputs)

    page.queue()
    return page

def _stream_predict(
    client: chat_client.ChatClient,
    use_knowledge_base: List[str],
    inference_mode: str,
    nvcf_model_id: str,
    nim_model_ip: str,
    nim_model_port: str,
    nim_model_id: str,
    is_local_nim: bool,
    num_token_slider: float,
    temp_slider: float,
    top_p_slider: float,
    freq_pen_slider: float,
    pres_pen_slider: float,
    start_local_server: str,
    local_model_id: str,
    question: str,
    metrics_history: dict,
    chat_history: List[Tuple[str, str]],
) -> Any:
    """
    Make a prediction of the response to the prompt.

    Parameters:
        client (chat_client.ChatClient): The chat client running the application.
        use_knowledge_base (List[str]): Whether or not the vector db should be invoked for this query
        inference_mode (str): The inference mode selected for this query
        nvcf_model_id (str): The cloud endpoint selected for this query
        nim_model_ip (str): The ip address running the remote nim selected for this query
        nim_model_port (str): The port for the remote nim selected for this query
        nim_model_id (str): The model name for remote nim selected for this query
        is_local_nim (bool): Whether to run the query as local or remote nim
        num_token_slider (float): max number of tokens to generate
        temp_slider (float): temperature selected for this query
        top_p_slider (float): top_p selected for this query
        freq_pen_slider (float): frequency penalty selected for this query
        pres_pen_slider (float): presence penalty selected for this query
        start_local_server (str): local TGI server status
        local_model_id (str): model name selected for local TGI inference of this query
        question (str): user prompt
        metrics_history (dict): current list of generated metrics
        chat_history (List[Tuple[str, str]]): current history of chatbot messages

    Returns:
        (Dict[gr.component, Dict[Any, Any]]): Gradio components to update.
    """
    chunks = ""

    # Count prompt tokens upfront so we can display them after the response
    try:
        _prompt_tokens = len(tiktoken.get_encoding("cl100k_base").encode(question))
    except Exception:
        _prompt_tokens = 0

    _LOGGER.info(
        "processing inference request - %s",
        str({"prompt": question, "use_knowledge_base": False if len(use_knowledge_base) == 0 else True}),
    )

    # Input validation for remote microservice settings
    if (utils.inference_to_config(inference_mode) == "microservice" and
        (len(nim_model_ip) == 0) and
        is_local_nim == False):
        yield "", chat_history + [[question, "*** ERR: Unable to process query. ***\n\nMessage: Hostname/IP field cannot be empty. "]], None, gr.update(value=metrics_history), metrics_history, gr.update()

    # Inputs are validated, can proceed with generating a response to the user query.
    else:

        # Try to send a request for the query
        try:
            documents: Union[None, List[Dict[str, Union[str, float]]]] = None
            response_num = len(metrics_history.keys())
            retrieval_ftime = ""
            chunks = ""
            e2e_stime = time.time()
            use_vdb = len(use_knowledge_base) != 0
            if use_vdb:
                retrieval_stime = time.time()
                try:
                    documents = client.search(question)
                except Exception as vdb_err:
                    # Vector DB search failed (e.g. empty collection, embedding error).
                    # Warn the user and continue answering without context.
                    gr.Warning(
                        "⚠️ Vector DB search failed — answering without retrieved context. "
                        "Ensure you have uploaded documents before enabling the Vector Database. "
                        f"({type(vdb_err).__name__})"
                    )
                    use_vdb = False
                    documents = None
                retrieval_ftime = str((time.time() - retrieval_stime) * 1000).split('.', 1)[0]

            # Generate the output
            chunk_num = 0
            ttft = "0"
            for chunk in client.predict(question,
                                        utils.inference_to_config(inference_mode),
                                        local_model_id,
                                        utils.cloud_to_config(nvcf_model_id),
                                        nim_model_ip,
                                        nim_model_port,
                                        nim_model_id,
                                        temp_slider,
                                        top_p_slider,
                                        freq_pen_slider,
                                        pres_pen_slider,
                                        use_vdb,
                                        int(num_token_slider)):

                # The first chunk returned will always be the time to first token. Let's process that first.
                if chunk_num == 0:
                    chunk_num += 1
                    ttft = chunk
                    updated_metrics_history = utils.get_initial_metrics(metrics_history, response_num, inference_mode, nvcf_model_id, local_model_id,
                                                                        is_local_nim, nim_model_id, retrieval_ftime, ttft)
                    yield "", chat_history, documents, gr.update(value=updated_metrics_history), updated_metrics_history, gr.update()

                # Every next chunk will be the generated response. Let's append to the output and render it in real time.
                else:
                    chunks += chunk
                    chunk_num += 1
                yield "", chat_history + [[question, chunks]], documents, gr.update(value=metrics_history), metrics_history, gr.update()

            # With final output generated, run some final calculations and display them as metrics to the user
            gen_time, e2e_ftime, tokens, tokens_sec, itl = utils.get_final_metrics(time.time(), e2e_stime, ttft, retrieval_ftime, chunks)
            metrics_history.get(str(response_num)).update({"Generation Time": gen_time + "ms",
                                                           "End to End Time (E2E)": e2e_ftime + "ms",
                                                           "Tokens (est.)": tokens + " tokens",
                                                           "Tokens/Second (est.)": tokens_sec + " tokens/sec",
                                                           "Inter-Token Latency (est.)": itl + " ms"})

            # Warn the user if the response looks truncated (reached ≥95 % of the token limit)
            try:
                if int(tokens) >= int(num_token_slider) * 0.95:
                    chunks += (
                        f"\n\n---\n⚠️ *Response may be incomplete — the model reached the "
                        f"**{int(num_token_slider)}-token limit**. "
                        f"Open **⚙ Parameters** in the sidebar and increase the Max Tokens slider, "
                        f"then ask again.*"
                    )
            except Exception:
                pass

            token_info = (
                f"✏️ Prompt: **{_prompt_tokens}** tokens | "
                f"📤 Response: **{tokens}** tokens"
            )
            final_history = chat_history + [[question, chunks]]
            _save_history(final_history, metrics_history)
            try:
                active_sid = _get_current_session_id()
                if active_sid:
                    session = _load_session_data(active_sid) or {}
                    _save_session(active_sid, session.get("name", "Chat"),
                                  final_history, metrics_history,
                                  session.get("documents", []))
            except Exception:
                pass
            yield "", gr.update(value=final_history), documents, gr.update(value=metrics_history), metrics_history, gr.update(value=token_info)

        # Catch any exceptions and direct the user to the logs/output.
        except Exception as e:
            err_history = chat_history + [[question, "*** ERR: Unable to process query. ***\n\nMessage: " + traceback.format_exc()]]
            _save_history(err_history, metrics_history)
            yield "", err_history, None, gr.update(value=metrics_history), metrics_history, gr.update()
