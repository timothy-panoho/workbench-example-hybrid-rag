# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import os
import requests
from PIL import Image
import gradio as gr

NVIDIA_GENAI_BASE = "https://ai.api.nvidia.com/v1/genai"

# ---------------------------------------------------------------------------
# Model catalogue
# Each entry describes how to call that model's endpoint.
#   endpoint  – full URL (overrides NVIDIA_GENAI_BASE/{id} if set)
#   style     – "flux" | "sdxl"  (controls request payload shape)
#   sizes     – valid size strings shown in the UI
#   steps     – default inference steps
# ---------------------------------------------------------------------------
MODELS = {
    "FLUX.1 Schnell (fast, 1-4 steps)": {
        "id":    "black-forest-labs/flux.1-schnell",
        "style": "flux",
        "sizes": ["1024x1024", "768x1344", "1344x768", "896x1152", "1152x896", "832x1216", "1216x832"],
        "default_size": "1024x1024",
        "steps": 4,
    },
    "FLUX.1 Dev (quality, 50 steps)": {
        "id":    "black-forest-labs/flux.1-dev",
        "style": "flux",
        "sizes": ["1024x1024", "768x1344", "1344x768", "896x1152", "1152x896", "832x1216", "1216x832"],
        "default_size": "1024x1024",
        "steps": 50,
    },
    "Stable Diffusion XL": {
        "id":    "stabilityai/stable-diffusion-xl",
        "style": "sdxl",
        "sizes": ["1024x1024"],
        "default_size": "1024x1024",
        "steps": 25,
    },
}


def _get_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise gr.Error("NVIDIA_API_KEY is not set. Add it in Workbench → Environment → Secrets.")
    return key


def _build_payload(cfg: dict, prompt: str, negative_prompt: str,
                   width: int, height: int, seed: int) -> dict:
    style = cfg["style"]
    steps = cfg.get("steps", 25)

    if style == "flux":
        payload: dict = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "steps": steps,
            "seed": seed,
            "cfg_scale": 0,
            "samples": 1,
        }
        # FLUX doesn't support negative_prompt; skip silently

    elif style == "sdxl":
        text_prompts = [{"text": prompt, "weight": 1}]
        if negative_prompt.strip():
            text_prompts.append({"text": negative_prompt, "weight": -1})
        payload = {
            "text_prompts": text_prompts,
            "height": height,
            "width": width,
            "cfg_scale": 5,
            "steps": steps,
            "seed": seed,
            "samples": 1,
        }

    else:
        raise gr.Error(f"Unknown model style: {style}")

    return payload


def _generate(cfg: dict, prompt: str, negative_prompt: str,
              width: int, height: int, seed: int) -> Image.Image:
    payload = _build_payload(cfg, prompt, negative_prompt, width, height, seed)

    url = f"{NVIDIA_GENAI_BASE}/{cfg['id']}"
    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers, timeout=120)

    if response.status_code != 200:
        raise gr.Error(f"API error {response.status_code}: {response.text[:400]}")

    data = response.json()

    # NVIDIA genai response: {"artifacts": [{"base64": "...", ...}]}
    artifacts = data.get("artifacts", [])
    if not artifacts:
        raise gr.Error("No image returned from API.")

    img_bytes = base64.b64decode(artifacts[0]["base64"])
    return Image.open(io.BytesIO(img_bytes))


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------

def _on_model_change(model_label: str):
    cfg = MODELS[model_label]
    sizes = cfg["sizes"]
    default = cfg["default_size"]
    w, h = map(int, default.split("x"))
    show_neg = cfg["style"] != "flux"
    return (
        gr.update(choices=sizes, value=default),
        gr.update(value=w),
        gr.update(value=h),
        gr.update(visible=show_neg),
    )


def _on_size_change(size_str: str):
    w, h = map(int, size_str.split("x"))
    return gr.update(value=w), gr.update(value=h)


def _run_generation(model_label, prompt, negative_prompt, size_str, seed,
                    progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    cfg = MODELS[model_label]
    w, h = map(int, size_str.split("x"))
    progress(0, desc=f"Sending request to NVIDIA API ({cfg['id']})…")
    img = _generate(cfg, prompt, negative_prompt, w, h, int(seed))
    progress(1, desc="Done.")
    return img


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    default_model = "FLUX.1 Schnell (fast, 1-4 steps)"
    default_cfg = MODELS[default_model]

    with gr.Blocks(title="Text-to-Image — NVIDIA API", theme=gr.themes.Base()) as demo:
        gr.Markdown(
            "# Text-to-Image\n"
            "Generate images via NVIDIA cloud API (`ai.api.nvidia.com`). "
            "Requires `NVIDIA_API_KEY` set in Workbench → Environment → Secrets."
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value=default_model,
                    label="Model",
                )
                prompt_tb = gr.Textbox(
                    label="Prompt",
                    placeholder="A photorealistic astronaut riding a horse on Mars, golden hour",
                    lines=3,
                )
                neg_prompt_tb = gr.Textbox(
                    label="Negative prompt (SDXL only)",
                    placeholder="blurry, low quality, watermark",
                    lines=2,
                    visible=False,   # hidden for FLUX; shown for SDXL
                )
                size_dd = gr.Dropdown(
                    choices=default_cfg["sizes"],
                    value=default_cfg["default_size"],
                    label="Size",
                )
                with gr.Row():
                    width_num = gr.Number(
                        value=int(default_cfg["default_size"].split("x")[0]),
                        label="Width", precision=0, interactive=False,
                    )
                    height_num = gr.Number(
                        value=int(default_cfg["default_size"].split("x")[1]),
                        label="Height", precision=0, interactive=False,
                    )
                seed_num = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                image_out = gr.Image(label="Generated image", type="pil")

        model_dd.change(
            _on_model_change,
            inputs=model_dd,
            outputs=[size_dd, width_num, height_num, neg_prompt_tb],
        )
        size_dd.change(_on_size_change, inputs=size_dd, outputs=[width_num, height_num])
        generate_btn.click(
            _run_generation,
            inputs=[model_dd, prompt_tb, neg_prompt_tb, size_dd, seed_num],
            outputs=image_out,
        )

    return demo
