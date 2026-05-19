"""
main.py - PNP Entrypoint

Start the local persistent process and API server.
"""

import logging
import os

import uvicorn
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

if __name__ == "__main__":
    config_path = os.getenv("PNP_CONFIG_PATH", "config/default.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    print("""
PNP - Persistent Neural Process
Starting local persistent API...
    """)

    uvicorn.run(
        "api.server:app",
        host=config.get("api_host", "127.0.0.1"),
        port=config.get("api_port", 8000),
        log_level=config.get("api_log_level", "info"),
        reload=False,
    )
