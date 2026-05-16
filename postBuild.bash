#!/bin/bash
set -e

# Install deps to run the API in a seperate venv to isolate different components
conda create --name api-env -y python=3.10 pip
$HOME/.conda/envs/api-env/bin/pip install setuptools fastapi==0.109.2 uvicorn[standard]==0.27.0.post1 python-multipart==0.0.7 langchain==0.0.335 langchain-community==0.0.19 openai==1.55.3 httpx==0.27.2 unstructured[all-docs]==0.17.2 sentence-transformers==2.7.0 llama-index==0.9.44 dataclass-wizard==0.22.3 pymilvus==2.3.1 opencv-python==4.8.0.76 hf_transfer==0.1.5 text_generation==0.6.1 transformers==4.50.0 nltk==3.8.1 torch==2.5.0 

# Install deps to run the UI in a seperate venv to isolate different components
conda create --name ui-env -y python=3.10 pip
$HOME/.conda/envs/ui-env/bin/pip install dataclass_wizard==0.22.2 gradio==4.15.0 jinja2==3.1.2 numpy==1.25.2 protobuf==3.20.3 PyYAML==6.0 uvicorn==0.22.0 torch==2.1.1 tiktoken==0.7.0 regex==2024.5.15 fastapi==0.112.2 Pillow requests
$HOME/.conda/envs/ui-env/bin/pip install "huggingface_hub==0.23.4"

sudo -E /opt/conda/bin/pip install anyio==4.3.0 pymilvus==2.3.1 transformers==4.40.0 marshmallow==3.20.1

sudo -E mkdir -p /mnt/milvus
sudo -E mkdir -p /data
sudo -E chown $NVWB_UID:$NVWB_GID /mnt/milvus
sudo -E chown $NVWB_UID:$NVWB_GID /data

sudo -E curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo -E bash
sudo -E apt-get install git-lfs
$HOME/.conda/envs/api-env/bin/pip install 'setuptools==69.5.1' --quiet

# Install Docker CLI (client only — daemon runs on the host, socket is shared by Workbench)
# Installed to user home so no sudo is required
DOCKER_VERSION="27.5.1"
mkdir -p "$HOME/.local/bin"
curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz" \
  | tar -xz --strip-components=1 -C "$HOME/.local/bin" docker/docker

# Install Docker Compose plugin (v2) to user directory
COMPOSE_VERSION="2.27.0"
mkdir -p "$HOME/.docker/cli-plugins"
curl -SL "https://github.com/docker/compose/releases/download/v${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
  -o "$HOME/.docker/cli-plugins/docker-compose"
chmod +x "$HOME/.docker/cli-plugins/docker-compose"

# Ensure ~/.local/bin is on PATH
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" \
  || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"

# Make launch-model.sh executable
chmod +x /project/code/scripts/launch-model.sh