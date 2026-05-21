import os
import uvicorn
from modelui.app import create_app

proxy_prefix = os.environ.get("PROXY_PREFIX", "")
app = create_app(proxy_prefix)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")
