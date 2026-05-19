from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODEL = os.environ.get("PNP_SMOKE_OLLAMA_MODEL", "qwen2.5-coder:7b")
TOKEN = "smoke-local-token"


def _ollama_has_model(model: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return False, "ollama executable not found"
    except subprocess.TimeoutExpired:
        return False, "ollama list timed out"

    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0 and model in output, output


def main() -> int:
    artifact_dir = REPO_ROOT / ".smoke" / "ollama-qwen-smoke"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    runtime_dir = artifact_dir / "runtime"
    runtime_dir.mkdir(parents=True)

    has_model, model_check = _ollama_has_model(MODEL)
    result: dict = {
        "ok": False,
        "model": MODEL,
        "artifact_dir": str(artifact_dir),
        "model_check": model_check,
    }
    if not has_model:
        result["error"] = f"Ollama model is not available: {MODEL}"
        result_path = artifact_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 1

    config = {
        "model_id": MODEL,
        "base_model": MODEL,
        "data_dir": str(runtime_dir),
        "working_memory_capacity": 2048,
        "episodic_db_path": str(runtime_dir / "episodic.db"),
        "semantic_db_path": str(runtime_dir / "semantic.db"),
        "adapter_path": str(runtime_dir / "adapter"),
        "core_state_path": str(runtime_dir / "core_state.json"),
        "journal_path": str(runtime_dir / "events.jsonl"),
        "api_host": "127.0.0.1",
        "api_port": 8000,
        "api_log_level": "warning",
        "local_api_token": TOKEN,
        "inference_provider": "ollama",
        "providers": {
            "ollama": {
                "api_base": os.environ.get("PNP_OLLAMA_BASE", "http://127.0.0.1:11434"),
            }
        },
        "adapter_training_epochs": 4,
    }
    config_path = artifact_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    os.environ["PNP_CONFIG_PATH"] = str(config_path)
    os.environ["PNP_INFERENCE_PROVIDER"] = "ollama"
    os.environ["PNP_MODEL_ID"] = MODEL

    from fastapi.testclient import TestClient
    from api import server

    headers = {"X-PNP-Token": TOKEN}
    with TestClient(server.app) as client:
        root = client.get("/")
        result["root_status"] = root.status_code
        result["root"] = root.json()

        chat = client.post(
            "/chat",
            json={
                "message": "Reply with a short confirmation that the local PNP Ollama smoke ran.",
                "concepts": ["ollama", "local_model"],
                "max_tokens": 48,
            },
            headers=headers,
            timeout=180,
        )
        result["chat_status"] = chat.status_code
        result["chat"] = chat.json()

    response_text = result.get("chat", {}).get("response", "")
    result["ok"] = (
        result["root_status"] == 200
        and result["chat_status"] == 200
        and result.get("chat", {}).get("provider") == "ollama"
        and bool(response_text.strip())
    )
    result["result_path"] = str(artifact_dir / "result.json")
    (artifact_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
