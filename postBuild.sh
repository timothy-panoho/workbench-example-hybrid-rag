#!/bin/bash
# postBuild.sh — executed as ROOT during AI Workbench environment build.
#
# Configures passwordless sudo for docker so the workbench user can manage
# containers from within the project container without needing a TTY or
# password (the socket is bind-mounted but owned by root:docker).

set -e

# Allow workbench to run docker (and docker compose plugin) without password.
# /usr/bin/docker covers both "docker <cmd>" and "docker compose <cmd>"
# because the compose plugin is invoked through the main docker binary.
SUDOERS_FILE=/etc/sudoers.d/workbench-docker
echo 'workbench ALL=(root) NOPASSWD: /usr/bin/docker' > "$SUDOERS_FILE"
chmod 440 "$SUDOERS_FILE"
# Validate the file is syntactically correct before leaving it in place
visudo -c -f "$SUDOERS_FILE" || { rm -f "$SUDOERS_FILE"; echo "ERROR: sudoers syntax check failed"; exit 1; }

echo "✅ passwordless sudo for /usr/bin/docker granted to workbench"
