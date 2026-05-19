"""
daemon/heartbeat.py

Process management for the PNP daemon — runs the persistent core as a
long-lived background process with signal handling, PID file management,
and graceful shutdown.

Usage:
  python -m daemon.heartbeat start    # Start daemon in foreground
  python -m daemon.heartbeat status   # Check if running
  python -m daemon.heartbeat stop     # Send SIGTERM to running daemon
"""

import asyncio
import signal
import sys
import os
import json
import time
import logging
import argparse
import yaml

logger = logging.getLogger(__name__)

PID_FILE = "./data/pnp.pid"
LOG_FILE = "./data/pnp.log"


class HeartbeatDaemon:
    """
    Manages the persistent core as a long-lived process.

    Handles:
    - Signal-based graceful shutdown (SIGTERM, SIGINT)
    - PID file for process tracking
    - State restoration across restarts
    - Periodic state persistence (every 60s)
    """

    def __init__(self, config: dict):
        self.config = config
        self._core = None
        self._shutdown_event = asyncio.Event()
        self._state_save_interval = 60  # seconds

    async def run(self):
        """Main entry point — starts the persistent core and runs until signal."""
        from core.process import PersistentCore

        self._core = PersistentCore(self.config)
        self._write_pid()

        # Register signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._signal_handler)

        logger.info(f"PNP daemon started. PID: {os.getpid()}")
        print(f"PNP daemon running (PID {os.getpid()}). Ctrl+C to stop.")

        try:
            await asyncio.gather(
                self._core.start(),
                self._periodic_save_loop(),
                self._shutdown_watch(),
            )
        except asyncio.CancelledError:
            pass
        finally:
            await self._core.stop()
            self._remove_pid()
            logger.info("PNP daemon stopped cleanly.")

    def _signal_handler(self):
        logger.info("Shutdown signal received.")
        self._shutdown_event.set()

    async def _shutdown_watch(self):
        """Watch for shutdown signal and cancel all tasks."""
        await self._shutdown_event.wait()
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()

    async def _periodic_save_loop(self):
        """Periodically persist core state to disk."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(self._state_save_interval)
            if self._core and not self._shutdown_event.is_set():
                try:
                    await self._core._save_state()
                    logger.debug("Periodic state save complete.")
                except Exception as e:
                    logger.warning(f"Periodic save failed: {e}")

    def _write_pid(self):
        os.makedirs(os.path.dirname(os.path.abspath(PID_FILE)), exist_ok=True)
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

    def _remove_pid(self):
        try:
            os.remove(PID_FILE)
        except FileNotFoundError:
            pass


def get_running_pid() -> int | None:
    """Return PID of running daemon, or None if not running."""
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # Check if process exists (raises if not)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def cmd_status():
    pid = get_running_pid()
    if pid:
        print(f"PNP daemon is RUNNING (PID {pid})")
        # Try to read last state
        if os.path.exists("./data/core_state.json"):
            try:
                with open("./data/core_state.json") as f:
                    state = json.load(f)
                uptime_h = round(state.get("uptime_seconds", 0) / 3600, 2)
                interactions = state.get("total_interactions", 0)
                print(f"  Uptime: {uptime_h}h | Interactions: {interactions}")
            except Exception:
                pass
    else:
        print("PNP daemon is NOT running")
        # Clean stale PID file
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)


def cmd_stop():
    pid = get_running_pid()
    if not pid:
        print("PNP daemon is not running.")
        return
    import signal as sig
    try:
        os.kill(pid, sig.SIGTERM)
        print(f"Sent SIGTERM to PID {pid}. Daemon shutting down...")
        # Wait for process to exit
        for _ in range(30):
            time.sleep(0.5)
            if get_running_pid() is None:
                print("Daemon stopped.")
                return
        print("Daemon did not stop within 15s. Try kill -9.")
    except ProcessLookupError:
        print("Process not found — already stopped.")


def cmd_start(config: dict):
    pid = get_running_pid()
    if pid:
        print(f"PNP daemon already running (PID {pid}). Use 'stop' first.")
        return

    os.makedirs("./data", exist_ok=True)
    daemon = HeartbeatDaemon(config)
    asyncio.run(daemon.run())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE) if os.path.exists("./data") or True else logging.NullHandler(),
        ]
    )

    parser = argparse.ArgumentParser(description="PNP Daemon")
    parser.add_argument("command", choices=["start", "stop", "status"],
                        help="Daemon command")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Config file path")
    args = parser.parse_args()

    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    if args.command == "start":
        cmd_start(config)
    elif args.command == "stop":
        cmd_stop()
    elif args.command == "status":
        cmd_status()
