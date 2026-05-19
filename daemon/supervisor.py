"""
daemon/supervisor.py

Small local watchdog for running the PNP API as a managed child process. It can
restart a crashed child and optionally restart after repeated health failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import os
import subprocess
import sys
import time
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass
class SupervisorConfig:
    command: list[str]
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    health_url: Optional[str] = None
    check_interval_seconds: float = 2.0
    restart_backoff_seconds: float = 1.0
    max_restarts: int = 10
    max_health_failures: int = 3
    stop_timeout_seconds: float = 5.0


@dataclass
class SupervisorStats:
    starts: int = 0
    restarts: int = 0
    health_failures: int = 0
    last_exit_code: Optional[int] = None
    last_restart_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "starts": self.starts,
            "restarts": self.restarts,
            "health_failures": self.health_failures,
            "last_exit_code": self.last_exit_code,
            "last_restart_reason": self.last_restart_reason,
        }


class ProcessSupervisor:
    """Supervise one child process with restart and health-check behavior."""

    def __init__(self, config: SupervisorConfig):
        self.config = config
        self.stats = SupervisorStats()
        self.process: Optional[subprocess.Popen] = None
        self._consecutive_health_failures = 0

    def start(self):
        env = os.environ.copy()
        env.update(self.config.env)
        self.process = subprocess.Popen(
            self.config.command,
            cwd=self.config.cwd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.stats.starts += 1

    def stop(self):
        if not self.process or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=self.config.stop_timeout_seconds)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=self.config.stop_timeout_seconds)

    def run_cycle(self):
        if self.process is None:
            self.start()
            return

        exit_code = self.process.poll()
        if exit_code is not None:
            self.stats.last_exit_code = exit_code
            self._restart("process_exited")
            return

        if self.config.health_url and not self._healthy():
            self._consecutive_health_failures += 1
            self.stats.health_failures += 1
            if self._consecutive_health_failures >= self.config.max_health_failures:
                self._restart("health_check_failed")
        else:
            self._consecutive_health_failures = 0

    def supervise(self, max_cycles: Optional[int] = None) -> SupervisorStats:
        cycles = 0
        try:
            while max_cycles is None or cycles < max_cycles:
                self.run_cycle()
                cycles += 1
                if self.stats.restarts >= self.config.max_restarts:
                    break
                time.sleep(self.config.check_interval_seconds)
        finally:
            if max_cycles is not None:
                self.stop()
        return self.stats

    def _restart(self, reason: str):
        if self.stats.restarts >= self.config.max_restarts:
            return
        self.stats.restarts += 1
        self.stats.last_restart_reason = reason
        self.stop()
        time.sleep(self.config.restart_backoff_seconds)
        self.start()

    def _healthy(self) -> bool:
        try:
            request = Request(self.config.health_url or "", method="GET")
            with urlopen(request, timeout=2.0) as response:
                return 200 <= response.status < 500
        except (OSError, URLError, TimeoutError):
            return False


def build_default_config(repo_root: str, host: str = "127.0.0.1", port: int = 8000) -> SupervisorConfig:
    return SupervisorConfig(
        command=[sys.executable, "main.py"],
        cwd=repo_root,
        health_url=f"http://{host}:{port}/",
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the PNP local API under a watchdog supervisor.")
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-cycles", type=int, default=None)
    args = parser.parse_args(argv)

    supervisor = ProcessSupervisor(build_default_config(args.repo_root, args.host, args.port))
    stats = supervisor.supervise(max_cycles=args.max_cycles)
    print(stats.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
