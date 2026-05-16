# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import os
import requests
from PIL import Image
import gradio as gr

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

MODELS = {
    "SDXL Turbo (fast)": {
        "id": "stabilityai/sdxl-turbo",
        "sizes": ["512x512", "768x768"],
        "default_size": "512x512",
    },
    "FLUX.1 Schnell (fast, 1024px)": {
        "id": "black-forest-labs/flux-schnell",
        "sizes": ["1024x1024"],
        "default_size": "1024x1024",
    },
    "FLUX.1 Dev (quality, 1024px)": {
        "id": "black-forest-labs/flux-dev",
        "sizes": ["1024x1024"],
        "default_size": "1024x1024",
    },
    "Stable Diffusion 3.5 Large (quality, 1024px)": {
        "id": "stabilityai/stable-diffusion-3-5-large",
        "sizes": ["1024x1024"],
        "default_size": "1024x1024",
    },
}


def _get_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise gr.Error("NVIDIA_API_KEY is not set. Add it in Workbench → Environment → Secrets.")
    return key


def _generate(model_id: str, prompt: str, negative_prompt: str, width: int, height: int, seed: int) -> Image.Image:
    payload = {
        "model": model_id,
        "prompt": prompt,
        "n": 1,
        "size": f"{width}x{height}",
        "response_format": "b64_json",
    }
    if negative_prompt.strip():
        payload["negative_prompt"] = negative_prompt
    if seed != 0:
        payload["seed"] = int(seed)

    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    response = requests.post(
        f"{NVIDIA_API_BASE}/images/generations",
        json=payload,
        headers=headers,
        timeout=120,
    )

    if response.status_code != 200:
        raise gr.Error(f"API error {response.status_code}: {response.text[:400]}")

    data = response.json()
    # OpenAI-compatible response: data.data[0].b64_json
    items = data.get("data", [])
    if not items:
        raise gr.Error("No image returned from API.")

    img_bytes = base64.b64decode(items[0]["b64_json"])
    return Image.open(io.BytesIO(img_bytes))


def _on_model_change(model_label: str):
    cfg = MODELS[model_label]
    sizes = cfg["sizes"]
    default = cfg["default_size"]
    w, h = map(int, default.split("x"))
    return gr.update(choices=sizes, value=default), gr.update(value=w), gr.update(value=h)


def _on_size_change(size_str: str):
    w, h = map(int, size_str.split("x"))
    return gr.update(value=w), gr.update(value=h)


def _run_generation(model_label, prompt, negative_prompt, size_str, seed, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    cfg = MODELS[model_label]
    w, h = map(int, size_str.split("x"))
    progress(0, desc=f"Sending request to NVIDIA API ({cfg['id']})...")
    img = _generate(cfg["id"], prompt, negative_prompt, w, h, seed)
    progress(1, desc="Done.")
    return img


def build_app() -> gr.Blocks:
    default_model = "SDXL Turbo (fast)"
    default_cfg = MODELS[default_model]

    with gr.Blocks(title="Text-to-Image — NVIDIA API", theme=gr.themes.Base()) as demo:
        gr.Markdown(
            "# Text-to-Image\n"
            "Generate images via NVIDIA cloud API (`integrate.api.nvidia.com`). "
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
                    label="Negative prompt (optional)",
                    placeholder="blurry, low quality, watermark",
                    lines=2,
                )
                size_dd = gr.Dropdown(
                    choices=default_cfg["sizes"],
                    value=default_cfg["default_size"],
                    label="Size",
                )
                with gr.Row():
                    width_num = gr.Number(
                        value=int(default_cfg["default_size"].split("x")[0]),
                        label="Width",
                        precision=0,
                        interactive=False,
                    )
                    height_num = gr.Number(
                        value=int(default_cfg["default_size"].split("x")[1]),
                        label="Height",
                        precision=0,
                        interactive=False,
                    )
                seed_num = gr.Number(value=0, label="Seed (0 = API default)", precision=0)
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                image_out = gr.Image(label="Generated image", type="pil")

        model_dd.change(_on_model_change, inputs=model_dd, outputs=[size_dd, width_num, height_num])
        size_dd.change(_on_size_change, inputs=size_dd, outputs=[width_num, height_num])
        generate_btn.click(
            _run_generation,
            inputs=[model_dd, prompt_tb, neg_prompt_tb, size_dd, seed_num],
            outputs=image_out,
        )

    return demo
