import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from inference.engine import InferenceRequest
from inference.providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAICompatibleChatProvider,
    provider_from_config,
)


class ProviderHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)

        if self.path == "/v1/chat/completions":
            self._send({
                "choices": [{"message": {"content": "openai-compatible ok"}}],
                "usage": {"total_tokens": 7},
            })
            return

        if self.path == "/api/chat":
            self._send({
                "message": {"content": "ollama ok"},
                "prompt_eval_count": 3,
                "eval_count": 4,
            })
            return

        if self.path == "/v1/messages":
            self._send({
                "content": [{"type": "text", "text": "anthropic-compatible ok"}],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            })
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return

    def _send(self, body: dict):
        raw = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), ProviderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def request() -> InferenceRequest:
    return InferenceRequest(
        user_input="hello",
        working_memory_context="",
        episodic_context="",
        semantic_context="",
        adaptation_context="",
        core_state={},
        model="test-model",
    )


def test_provider_selection_is_model_agnostic():
    assert provider_from_config({"inference_provider": "mock"}).name == "mock"
    assert provider_from_config({"inference_provider": "ollama"}).name == "ollama"
    assert provider_from_config({"inference_provider": "openai_compatible"}).name == "openai_compatible"


def test_http_provider_adapters_against_local_server():
    server = serve()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        openai_provider = OpenAICompatibleChatProvider(f"{base}/v1", api_key="test")
        openai_result = asyncio.run(openai_provider.generate(request(), "system", [{"role": "user", "content": "hi"}]))
        assert openai_result.content == "openai-compatible ok"
        assert openai_result.tokens_used == 7

        ollama_provider = OllamaProvider(base)
        ollama_result = asyncio.run(ollama_provider.generate(request(), "system", [{"role": "user", "content": "hi"}]))
        assert ollama_result.content == "ollama ok"
        assert ollama_result.tokens_used == 7

        anthropic_provider = AnthropicProvider(api_key="test", api_base=base)
        anthropic_result = asyncio.run(
            anthropic_provider.generate(request(), "system", [{"role": "user", "content": "hi"}])
        )
        assert anthropic_result.content == "anthropic-compatible ok"
        assert anthropic_result.tokens_used == 7
    finally:
        server.shutdown()
