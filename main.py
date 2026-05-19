"""
main.py - PNP Entrypoint

Starts the persistent process and API server together.

Usage:
  python main.py
  PNP_CONFIG=custom.yaml python main.py
  PNP_INFERENCE_PROVIDER=ollama PNP_MODEL_ID=qwen2.5-coder:7b python main.py
"""

import logging
import os
import sys

import uvicorn
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
)

BANNER = """
================================================
    PNP - Persistent Neural Process
    v1.0.0
================================================
"""


def main():
    config_path = os.environ.get("PNP_CONFIG_PATH") or os.environ.get("PNP_CONFIG", "config/default.yaml")
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    data_dir = config.get("data_dir", "./data")
    os.makedirs(data_dir, exist_ok=True)

    provider = os.environ.get("PNP_INFERENCE_PROVIDER") or config.get("inference_provider", "mock")
    model = (
        os.environ.get("PNP_MODEL_ID")
        or os.environ.get("PNP_MODEL")
        or config.get("model_id")
        or config.get("base_model", "local-model")
    )
    host = config.get("api_host", "127.0.0.1")
    port = config.get("api_port", 8000)

    print(BANNER)
    print(f"  Config:    {config_path}")
    print(f"  Provider:  {provider}")
    print(f"  Model:     {model}")
    print(f"  API:       http://{host}:{port}")
    print(f"  Docs:      http://localhost:{port}/docs")
    print()

    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        log_level=config.get("api_log_level", "info"),
        reload=False,
        access_log=True,
    )


if __name__ == "__main__":
    main()
