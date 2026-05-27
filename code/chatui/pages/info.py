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

### Markdown used to render certain documentation on the gradio application. ###

setup = """
Welcome to the Hybrid RAG example project for NVIDIA AI Workbench! \n\nTo get started, click the following button to set up the backend API server and vector database. This is a one-time process and may take a few moments to complete.
"""

update_kb_info = """
<br> 
Upload your text files here. This will embed them in the vector database, and they will persist as potential context for the model until you clear the database. Careful, clearing the database is irreversible!
"""

inf_mode_info = "To use a CLOUD endpoint for inference, select the desired model before making a query."

local_info = """
Select the desired model and quantization level. You can optionally filter the list by gated vs ungated models. Then load the model. This will either download it or load it from cache. The download may take a few minutes depending on your network. 

Once weights are loaded, start the Inference Server (~40s warmup in most cases). Ensure enough GPU VRAM for local hosting or you may encounter OOM errors when starting the server. When the server is running, you can chat with the model.
"""

local_prereqs = """
* A ``HUGGING_FACE_HUB_TOKEN`` variable is required for gated models. See **Tutorial 1** of the README for details. 
* If using any of the following gated models, verify ``You have been granted access to this model`` appears on the model card(s):
    * [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
    * [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
    * [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
    * [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
"""

local_trouble = """
* Ensure you have stopped any local processes also running on the system GPU(s). Otherwise, you may run into OOM errors running on the local inference server. 
* Your Hugging Face key may be missing and/or lack permissions for certain models. Ensure you see a ``You have been granted access to this model`` for each page: 
    * [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
    * [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
    * [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
    * [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
"""

cloud_info = """
This method uses NVIDIA-hosted API Endpoints from the NVIDIA API Catalog. Select a desired model family and model from the dropdown. You may then query the model using the text input on the left.
"""

cloud_prereqs = """
* A ``NVIDIA_API_KEY`` variable is required. See the **Quickstart** of the README for details. 
    * Generate the key [here](https://build.nvidia.com/meta/llama-3_1-8b-instruct) by clicking "Get API Key". Log in with [NGC credentials](https://ngc.nvidia.com/signin).
"""

cloud_trouble = """
* Ensure your NVIDIA API Key is correct and configured properly in the AI Workbench. 
"""

nim_info = """
This method uses an [LLM NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/meta/containers/llama-3.1-8b-instruct/tags) that you can self-host on your own infra via the Compose feature in AI Workbench. Check out the NIM [docs](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html) for details. Users can also try 3rd party services supporting the [OpenAI API](https://github.com/ollama/ollama/blob/main/docs/openai.md) like [Ollama](https://github.com/ollama/ollama/blob/main/README.md#building). 

Input the desired microservice name if running locally or IP/hostname if running remotely, optional port number, and model name. Then, start conversing using the text input on the left.
"""

nim_prereqs = """
* (Remote) Set up a NIM running on another system ([docs](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html)). Alternatively, you may set up a 3rd party supporting the [OpenAI API](https://github.com/ollama/ollama/blob/main/docs/openai.md) like [Ollama](https://github.com/ollama/ollama/blob/main/README.md#building). Ensure your service is running and reachable. See **Tutorial 2** of the README for details. 
* (Local) Start the Compose service in the AI Workbench window. Wait a few minutes to ensure your service is running and reachable. See **Tutorial 3** of the README for details. 
"""

nim_trouble = """
* Send a curl request to your microservice to ensure it is running and reachable. NIM docs [here](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html).
* If any other processes are running on the local GPU(s), you may run into memory issues when also running the NIM locally. Stop the other processes. 
"""

num_token_label = """Max tokens in the reply (1 token ≈ ¾ of an English word, ~100 tokens ≈ 75 words).
• Too low → response gets cut off mid-sentence. Raise it if answers feel incomplete.
• Recommended starting point: 1024. For long-form answers push to 2048–4096.
• Very high values use more GPU memory and take longer — there is no benefit above the model's own context limit.
"""

temp_label = """Controls how creative / random the reply is. Range: 0 (fully deterministic) → 1 (very creative).
• 0.1 – 0.3 → focused, factual, consistent. Best for Q&A, code, and summarisation.
• 0.5 – 0.7 → balanced and natural. Good default for conversation.
• 0.8 – 1.0 → creative and varied, but may drift off-topic.
Tip: tune Temperature OR Top P — not both at the same time.
"""

top_p_label = """Nucleus sampling — restricts the pool of tokens to those whose cumulative probability reaches P.
• 0.95 – 0.999 → wide pool, natural output (recommended default).
• 0.5 → conservative / more predictable.
• 1.0 → no restriction (identical to temperature-only sampling).
Leave at 0.999 and adjust Temperature instead in most cases.
"""

freq_pen_label = """Penalises tokens that have already appeared often in the reply.
• 0 (default) → no penalty; model may repeat itself.
• 0.3 – 0.8 → noticeably reduces repetition. Good for long responses.
• 2.0 → very strong penalty; output may become unnatural.
Use this when the model keeps looping on the same phrases.
"""

pres_pen_label = """Penalises any token that has appeared at all in the reply (regardless of frequency).
• 0 (default) → no penalty.
• 0.3 – 0.8 → encourages the model to introduce new words / topics.
• 2.0 → strong push for novelty; may cause the answer to wander.
Use this when you want a broader, more exploratory answer.
"""

metrics_guide = """
| Metric | What it measures | Typical values & tips |
|---|---|---|
| **Retrieval time** | Time to search the vector database for relevant documents. Only measured when *Toggle to use Vector Database* is enabled. | < 500 ms = fast; > 2 s may mean the DB is still warming up |
| **Time to First Token (TTFT)** | Delay from clicking Submit until the first word appears. Covers prompt processing, KV-cache build, and network latency. | < 1 s on a local GPU; NIM on first cold-start can be 5–15 s while TensorRT loads |
| **Generation Time** | Time to produce the full response after the first token. Longer responses naturally take more time. | Roughly proportional to response length and inversely to GPU speed |
| **End to End (E2E)** | Total wall-clock time = Retrieval + TTFT + Generation. | Sum of the above rows |
| **Tokens (est.)** | Estimated number of tokens in the generated reply. ≈ ¾ of a word per token. | ⚠️ If this equals your *Max Tokens* setting the response was **cut off** — increase the slider and retry |
| **Tokens / Second (est.)** | Generation speed. | 30 – 80 tok/s is typical for a 4 B model on an RTX 4090 |
| **Inter-Token Latency (est.)** | Average time between consecutive tokens (= 1000 / tokens-per-sec). Lower is better. | < 20 ms = smooth streaming; > 100 ms will feel sluggish |
"""