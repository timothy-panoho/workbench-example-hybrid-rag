#!/bin/bash
# Wait for Milvus to be reachable.
# Uses Python urllib instead of curl to avoid the conda/libcurl conflict.

PYTHON="$HOME/.conda/envs/ui-env/bin/python3"
ATTEMPTS=0
MAX_ATTEMPTS=30

http_status() {
    $PYTHON - "$1" <<'PYEOF' 2>/dev/null
import sys, urllib.request
try:
    print(urllib.request.urlopen(sys.argv[1], timeout=3).getcode())
except Exception:
    print("000")
PYEOF
}

while [ "$(http_status http://localhost:19530/v1/vector/collections)" != "200" ]; do
    ATTEMPTS=$((ATTEMPTS+1))
    if [ "$ATTEMPTS" -eq "$MAX_ATTEMPTS" ]; then
        echo "Max attempts reached: $MAX_ATTEMPTS. Server may have timed out. Stop the container and try again."
        exit 1
    fi
    echo "Polling Milvus. Awaiting status 200; trying again in 10s."
    sleep 10
done
