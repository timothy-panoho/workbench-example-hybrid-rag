#!/bin/bash
# postBuild.sh
#
# Configures passwordless sudo for the workbench user so Docker commands
# work from within the project container without a TTY or password prompt.
#
# Apply immediately (no rebuild needed):
#   wsl -d NVIDIA-Workbench -u root -e docker exec --user root project-hybrid-rag bash /project/postBuild.sh
#
# Also runs automatically on the next environment rebuild.

set -e

echo "[postBuild] running as: $(whoami)"

# Grant workbench full passwordless sudo AND preserve PATH so that
# docker at /home/workbench/.local/bin/docker is found by sudo.
SUDOERS_FILE=/etc/sudoers.d/workbench-nopasswd
printf 'Defaults env_keep += "PATH"\nworkbench ALL=(ALL) NOPASSWD: ALL\n' > "$SUDOERS_FILE"
chmod 440 "$SUDOERS_FILE"
echo "[postBuild] wrote $SUDOERS_FILE"

# Belt-and-braces: add workbench to the docker group in case the host
# socket GID matches at runtime.
groupadd docker 2>/dev/null || true
usermod -aG docker workbench 2>/dev/null || true
echo "[postBuild] workbench added to docker group"

echo "[postBuild] done"
