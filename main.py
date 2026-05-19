"""
main.py — PNP Entrypoint

Starts the persistent process + API server together.
The API server provides the external interface; the persistent core
runs as a background task within the same event loop.

Usage:
  python main.py                          # Default config
  PNP_CONFIG=custom.yaml python main.py  # Custom config
  ANTHROPIC_API_KEY=sk-... python main.py  # With live inference
"""

import asyncio
import logging
import os
import sys
import yaml
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
)

BANNER = """
╔══════════════════════════════════════════╗
║    PNP — Persistent Neural Process       ║
║    v1.0.0                                ║
╚══════════════════════════════════════════╝
"""


def main():
    config_path = os.environ.get("PNP_CONFIG", "config/default.yaml")
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    os.makedirs("./data", exist_ok=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    inference_status = "LIVE" if api_key else "MOCK (set ANTHROPIC_API_KEY for live)"

    print(BANNER)
    print(f"  Config:    {config_path}")
    print(f"  Model:     {config.get('base_model', 'claude-haiku-4-5-20251001')}")
    print(f"  Inference: {inference_status}")
    print(f"  API:       http://{config.get('api_host', '0.0.0.0')}:{config.get('api_port', 8000)}")
    print(f"  Docs:      http://localhost:{config.get('api_port', 8000)}/docs")
    print()

    uvicorn.run(
        "api.server:app",
        host=config.get("api_host", "0.0.0.0"),
        port=config.get("api_port", 8000),
        log_level=config.get("api_log_level", "info"),
        reload=False,
        access_log=True,
    )


if __name__ == "__main__":
    main()
