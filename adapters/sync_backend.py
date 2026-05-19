"""
Provider-neutral sync-model adapter backend.

The sync pack is the durable bridge between PNP's learned state and whichever
model/provider is active on startup. It always persists a context pack that can
be injected into inference. When a local trainable runtime is configured, it can
also train and save a real PEFT LoRA adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Optional


@dataclass
class SyncTrainingResult:
    backend: str
    status: str
    trained: bool
    adapter_path: Optional[str] = None
    sample_count: int = 0
    train_steps: int = 0
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _delta_text(delta: Any) -> str:
    return str(getattr(delta, "content", ""))


def _positive_deltas(deltas: Iterable[Any]) -> list[Any]:
    return [
        delta for delta in deltas
        if float(getattr(delta, "feedback", 0.0)) > 0.1
        and _delta_text(delta).strip()
    ]


def _stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class SyncModelAdapterBackend:
    """Writes the durable sync pack and delegates optional weight training."""

    def __init__(self, adapter_path: str, config: dict, base_model_id: str):
        self.adapter_path = Path(adapter_path)
        self.config = config
        self.base_model_id = base_model_id
        self.pack_path = self.adapter_path / "sync_model_pack.json"
        self.context_path = self.adapter_path / "sync_model_context.md"

    def train(
        self,
        deltas: list[Any],
        domain_weights: dict[str, float],
        low_rank_metrics: dict[str, Any],
        epochs: Optional[int] = None,
    ) -> dict[str, Any]:
        positive = _positive_deltas(deltas)
        context = self._build_context(positive, domain_weights, low_rank_metrics)
        self.context_path.write_text(context, encoding="utf-8")

        peft_result = self._maybe_train_peft(positive, epochs=epochs)
        pack = {
            "schema_version": 1,
            "created_at": time.time(),
            "base_model_id": self.base_model_id,
            "provider": os.getenv("PNP_INFERENCE_PROVIDER") or self.config.get("inference_provider", "mock"),
            "model_id": os.getenv("PNP_MODEL_ID") or os.getenv("PNP_MODEL") or self.config.get("model_id") or self.base_model_id,
            "delta_count": len(deltas),
            "positive_delta_count": len(positive),
            "domain_weights": domain_weights,
            "low_rank_metrics": low_rank_metrics,
            "context_path": str(self.context_path),
            "context_sha256": _stable_digest(context),
            "peft_lora": peft_result.to_dict(),
        }
        self.pack_path.write_text(json.dumps(pack, indent=2, sort_keys=True), encoding="utf-8")
        return pack

    def load_pack(self) -> dict[str, Any]:
        if not self.pack_path.exists():
            return {}
        try:
            return json.loads(self.pack_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def load_context(self) -> str:
        if not self.context_path.exists():
            return ""
        try:
            return self.context_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _build_context(
        self,
        positive_deltas: list[Any],
        domain_weights: dict[str, float],
        low_rank_metrics: dict[str, Any],
    ) -> str:
        lines = [
            "=== SYNC MODEL ADAPTER STATE ===",
            f"Base model id: {self.base_model_id}",
            f"Configured provider: {os.getenv('PNP_INFERENCE_PROVIDER') or self.config.get('inference_provider', 'mock')}",
            f"Configured model: {os.getenv('PNP_MODEL_ID') or os.getenv('PNP_MODEL') or self.config.get('model_id') or self.base_model_id}",
            f"Positive learned deltas: {len(positive_deltas)}",
            f"Low-rank train steps: {low_rank_metrics.get('train_steps', low_rank_metrics.get('sample_count', 0))}",
        ]
        if domain_weights:
            ranked = sorted(domain_weights.items(), key=lambda item: item[1], reverse=True)[:8]
            lines.append("Domain weights: " + ", ".join(f"{name}={weight:.3f}" for name, weight in ranked))
        if positive_deltas:
            lines.append("Learned patterns:")
            for delta in positive_deltas[-12:]:
                domain = getattr(delta, "domain", "general")
                feedback = float(getattr(delta, "feedback", 0.0))
                confidence = float(getattr(delta, "confidence", 0.0))
                text = _delta_text(delta).replace("\n", " ").strip()
                lines.append(f"- [{domain} feedback={feedback:+.2f} confidence={confidence:.2f}] {text[:500]}")
        lines.append("=== END SYNC MODEL ADAPTER STATE ===")
        return "\n".join(lines)

    def _maybe_train_peft(self, positive_deltas: list[Any], epochs: Optional[int] = None) -> SyncTrainingResult:
        backend = str(self.config.get("sync_model_adapter_backend", "context_pack")).lower()
        if backend not in {"peft_lora", "auto"}:
            return SyncTrainingResult(
                backend="context_pack",
                status="context_only",
                trained=False,
                sample_count=len(positive_deltas),
                adapter_path=str(self.context_path),
                message="Provider-neutral sync context written; PEFT backend not requested.",
            )

        trainer = PeftLoRABackend(self.adapter_path, self.config)
        if not trainer.available():
            return SyncTrainingResult(
                backend="peft_lora",
                status="unavailable",
                trained=False,
                sample_count=len(positive_deltas),
                message=trainer.unavailable_reason(),
            )
        return trainer.train(positive_deltas, epochs=epochs)


class PeftLoRABackend:
    """Optional real LoRA adapter training for local Hugging Face runtimes."""

    def __init__(self, adapter_path: Path, config: dict):
        self.adapter_path = adapter_path / "peft_lora"
        self.config = config
        self._reason = ""

    def available(self) -> bool:
        model_name = self._base_model_name()
        if not model_name:
            self._reason = "Set sync_model_base_model or PNP_SYNC_MODEL_BASE to a local/Hugging Face model path."
            return False
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            import peft  # noqa: F401
        except ImportError as exc:
            self._reason = f"Missing optional training dependency: {exc.name}. Install with .\\install.ps1 -InstallTrainingDeps."
            return False
        return True

    def unavailable_reason(self) -> str:
        if not self._reason:
            self.available()
        return self._reason

    def train(self, deltas: list[Any], epochs: Optional[int] = None) -> SyncTrainingResult:
        if not deltas:
            return SyncTrainingResult(
                backend="peft_lora",
                status="no_samples",
                trained=False,
                adapter_path=str(self.adapter_path),
                sample_count=0,
                message="No positive deltas available for local LoRA training.",
            )

        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self._base_model_name()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        existing_adapter = self.adapter_path / "adapter_config.json"
        if existing_adapter.exists():
            model = PeftModel.from_pretrained(base_model, str(self.adapter_path), is_trainable=True)
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(self.config.get("peft_lora_rank", self.config.get("adapter_rank", 8))),
                lora_alpha=float(self.config.get("peft_lora_alpha", self.config.get("adapter_alpha", 16))),
                lora_dropout=float(self.config.get("peft_lora_dropout", 0.05)),
                target_modules=self.config.get("peft_lora_target_modules"),
            )
            model = get_peft_model(base_model, lora_config)
        model.to(device)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model.train()

        texts = [self._training_text(delta) for delta in deltas]
        max_length = int(self.config.get("peft_max_length", 512))
        learning_rate = float(self.config.get("peft_learning_rate", 2e-4))
        batch_size = int(self.config.get("peft_batch_size", 1))
        train_epochs = int(epochs or self.config.get("peft_training_epochs", 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        train_steps = 0
        last_loss = 0.0
        for _ in range(train_epochs):
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start:start + batch_size]
                encoded = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                labels = encoded["input_ids"].clone()
                if "attention_mask" in encoded:
                    labels[encoded["attention_mask"] == 0] = -100
                outputs = model(**encoded, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                last_loss = float(loss.detach().cpu())
                train_steps += 1

        self.adapter_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(self.adapter_path))
        tokenizer.save_pretrained(str(self.adapter_path))
        return SyncTrainingResult(
            backend="peft_lora",
            status="trained",
            trained=True,
            adapter_path=str(self.adapter_path),
            sample_count=len(texts),
            train_steps=train_steps,
            metadata={"device": device, "last_loss": round(last_loss, 6), "base_model": model_name},
        )

    def _base_model_name(self) -> str:
        return (
            os.getenv("PNP_SYNC_MODEL_BASE")
            or self.config.get("sync_model_base_model")
            or self.config.get("peft_base_model")
            or ""
        )

    def _training_text(self, delta: Any) -> str:
        domain = getattr(delta, "domain", "general")
        feedback = float(getattr(delta, "feedback", 0.0))
        confidence = float(getattr(delta, "confidence", 0.0))
        content = _delta_text(delta).strip()
        return (
            "Persistent Neural Process adaptation example\n"
            f"Domain: {domain}\n"
            f"Feedback: {feedback:+.2f}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Pattern: {content}\n"
        )
