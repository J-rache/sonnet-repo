"""
Provider adapters for PNP inference.

The adapter boundary is intentionally model/vendor agnostic. PNP builds one
context packet, then sends it to whichever configured provider is selected.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Optional

import httpx

from inference.engine import (
    InferenceRequest,
    InferenceResult,
    build_full_context,
    build_system_prompt,
    estimate_valence,
    extract_memory_deltas,
)


@dataclass
class ProviderResponse:
    content: str
    tokens_used: int
    metadata: dict


class InferenceProvider:
    name = "base"

    async def generate(self, request: InferenceRequest, system_prompt: str, messages: list[dict]) -> ProviderResponse:
        raise NotImplementedError


class MockProvider(InferenceProvider):
    name = "mock"

    async def generate(self, request: InferenceRequest, system_prompt: str, messages: list[dict]) -> ProviderResponse:
        content = f"Mock inference response: received '{request.user_input[:120]}'"
        return ProviderResponse(
            content=content,
            tokens_used=max(1, (len(content) + sum(len(m["content"]) for m in messages)) // 4),
            metadata={"provider": self.name, "model": request.model},
        )


class AnthropicProvider(InferenceProvider):
    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, api_base: str = "https://api.anthropic.com"):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")

    async def generate(self, request: InferenceRequest, system_prompt: str, messages: list[dict]) -> ProviderResponse:
        if not self.api_key:
            raise RuntimeError("Anthropic provider requires an API key.")

        payload = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{self.api_base}/v1/messages", headers=headers, json=payload)
            response.raise_for_status()
            raw = response.json()

        content = raw["content"][0]["text"]
        usage = raw.get("usage") or {}
        tokens_used = int(usage.get("input_tokens", 0) + usage.get("output_tokens", 0)) or max(1, len(content) // 4)
        return ProviderResponse(
            content=content,
            tokens_used=tokens_used,
            metadata={"provider": self.name, "model": request.model, "api_base": self.api_base},
        )


class OpenAICompatibleChatProvider(InferenceProvider):
    name = "openai_compatible"

    def __init__(self, api_base: str, api_key: Optional[str] = None):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key

    async def generate(self, request: InferenceRequest, system_prompt: str, messages: list[dict]) -> ProviderResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            raw = response.json()

        content = raw["choices"][0]["message"]["content"]
        usage = raw.get("usage") or {}
        tokens_used = int(usage.get("total_tokens") or max(1, len(content) // 4))
        return ProviderResponse(
            content=content,
            tokens_used=tokens_used,
            metadata={"provider": self.name, "model": request.model, "api_base": self.api_base},
        )


class OllamaProvider(InferenceProvider):
    name = "ollama"

    def __init__(self, api_base: str):
        self.api_base = api_base.rstrip("/")

    async def generate(self, request: InferenceRequest, system_prompt: str, messages: list[dict]) -> ProviderResponse:
        payload = {
            "model": request.model,
            "stream": False,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "options": {"num_predict": request.max_tokens},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{self.api_base}/api/chat", json=payload)
            response.raise_for_status()
            raw = response.json()

        content = raw.get("message", {}).get("content", "")
        eval_count = raw.get("eval_count") or 0
        prompt_eval_count = raw.get("prompt_eval_count") or 0
        tokens_used = int(eval_count + prompt_eval_count) or max(1, len(content) // 4)
        return ProviderResponse(
            content=content,
            tokens_used=tokens_used,
            metadata={"provider": self.name, "model": request.model, "api_base": self.api_base},
        )


def provider_from_config(config: dict) -> InferenceProvider:
    provider_name = os.getenv("PNP_INFERENCE_PROVIDER") or config.get("inference_provider", "mock")
    provider_name = provider_name.lower()

    providers = config.get("providers", {})
    provider_config = providers.get(provider_name, {})

    if provider_name == "mock":
        return MockProvider()
    if provider_name == "anthropic":
        api_key_env = provider_config.get("api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.getenv(api_key_env) or provider_config.get("api_key")
        api_base = provider_config.get("api_base", "https://api.anthropic.com")
        return AnthropicProvider(api_key=api_key, api_base=api_base)
    if provider_name in {"openai", "openai_compatible", "vllm", "lmstudio"}:
        api_base = (
            os.getenv("PNP_OPENAI_COMPATIBLE_BASE")
            or provider_config.get("api_base")
            or "http://127.0.0.1:8001/v1"
        )
        api_key_env = provider_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env) or provider_config.get("api_key")
        return OpenAICompatibleChatProvider(api_base=api_base, api_key=api_key)
    if provider_name == "ollama":
        api_base = os.getenv("PNP_OLLAMA_BASE") or provider_config.get("api_base") or "http://127.0.0.1:11434"
        return OllamaProvider(api_base=api_base)

    raise ValueError(f"Unsupported inference provider: {provider_name}")


async def run_provider_inference(request: InferenceRequest, config: dict) -> InferenceResult:
    t_start = time.monotonic()
    system_prompt = request.system_prompt_override or build_system_prompt(request.core_state)
    messages = build_full_context(request)
    provider = provider_from_config(config)
    provider_response = await provider.generate(request, system_prompt, messages)
    latency_ms = (time.monotonic() - t_start) * 1000

    memory_deltas = extract_memory_deltas(request.user_input, provider_response.content)
    valence = estimate_valence(provider_response.content)

    metadata = dict(provider_response.metadata)
    metadata["system_prompt_tokens"] = len(system_prompt) // 4

    return InferenceResult(
        content=provider_response.content,
        tokens_used=provider_response.tokens_used,
        latency_ms=latency_ms,
        memory_deltas=memory_deltas,
        suggested_goals=[],
        valence=valence,
        metadata=metadata,
    )
