# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

if __name__ == "__main__":
    proxy_prefix = os.environ.get("PROXY_PREFIX", "")
    from imageui.app import build_app
    demo = build_app()
    demo.queue(max_size=5)
    demo.launch(server_name="0.0.0.0", server_port=8081, root_path=proxy_prefix)
