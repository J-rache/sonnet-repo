"""
main.py — PNP Entrypoint

Start the persistent process + API server.
"""

import asyncio
import logging
import uvicorn
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)

if __name__ == "__main__":
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    print("""
╔═══════════════════════════════════════╗
║   PNP — Persistent Neural Process     ║
║   Starting continuous existence...    ║
╚═══════════════════════════════════════╝
    """)

    uvicorn.run(
        "api.server:app",
        host=config.get("api_host", "0.0.0.0"),
        port=config.get("api_port", 8000),
        log_level=config.get("api_log_level", "info"),
        reload=False,
    )
