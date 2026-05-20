#!/bin/bash
# postBuild.sh
#
# Configures passwordless sudo for the workbench user so Docker commands
# work from within the project container without a TTY or password prompt.
#
# Run manually to apply immediately:
#   sudo bash /project/postBuild.sh
#
# This is also picked up automatically on the next environment rebuild.

set -e

echo "[postBuild] running as: $(whoami)"

# Grant workbench full passwordless sudo.
# NOPASSWD: ALL is standard for development containers and is required
# because the Docker socket GID on the host may not match any group
# inside the container, making sudo the only reliable access path.
SUDOERS_FILE=/etc/sudoers.d/workbench-nopasswd
echo 'workbench ALL=(ALL) NOPASSWD: ALL' > "$SUDOERS_FILE"
chmod 440 "$SUDOERS_FILE"
echo "[postBuild] wrote $SUDOERS_FILE"

# Also add workbench to the docker group (belt-and-braces: works if
# the host docker socket GID happens to match at runtime).
groupadd docker 2>/dev/null || true
usermod -aG docker workbench 2>/dev/null || true
echo "[postBuild] workbench added to docker group"

echo "[postBuild] done — restart the chat app to apply"
