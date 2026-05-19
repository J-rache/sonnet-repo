from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from daemon.supervisor import ProcessSupervisor, SupervisorConfig


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="pnp-supervisor-smoke-") as tmp:
        supervisor = ProcessSupervisor(
            SupervisorConfig(
                command=[sys.executable, "-c", "import sys; sys.exit(7)"],
                cwd=tmp,
                check_interval_seconds=0.05,
                restart_backoff_seconds=0.01,
                max_restarts=2,
            )
        )
        stats = supervisor.supervise(max_cycles=20)

    result = {
        "ok": stats.restarts == 2 and stats.last_exit_code == 7,
        "stats": stats.to_dict(),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
