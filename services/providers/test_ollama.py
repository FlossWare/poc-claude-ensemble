"""Tests for local Ollama routing through the provider registry / api_client."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from arbitration.api_client import MultiModelClient
from providers.ollama import OllamaProvider
from providers.registry import ProviderRegistry


class FakeOllama:
    """Deterministic loopback stand-in for Ollama's HTTP API."""

    def __init__(self, *, content="LOCAL_OK", thinking=None, model="qwen2.5-coder:3b",
                 status=200, payload=None):
        self.content = content
        self.thinking = thinking
        self.model = model
        self.status = status
        self.payload = payload
        self.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def _handler(fake):  # noqa: N805 - closure factory
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append({"path": self.path, "body": body})
                if fake.payload is not None:
                    raw = json.dumps(fake.payload).encode("utf-8")
                else:
                    message = {"role": "assistant", "content": fake.content}
                    if fake.thinking is not None:
                        message["thinking"] = fake.thinking
                    raw = json.dumps({
                        "model": fake.model,
                        "message": message,
                        "done": True,
                        "prompt_eval_count": 11,
                        "eval_count": 3,
                    }).encode("utf-8")
                self.send_response(fake.status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                raw = json.dumps({"models": [
                    {"name": "qwen2.5-coder:3b"},
                    {"name": "deepseek-r1:7b"},
                ]}).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        return Handler

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@pytest.fixture
def fake_ollama():
    servers = []

    def build(**kwargs):
        server = FakeOllama(**kwargs)
        servers.append(server)
        return server

    yield build
    for server in servers:
        server.close()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ---------------------------------------------------------------------------
# Label routing
# ---------------------------------------------------------------------------

def test_registry_strips_ollama_prefix_and_selects_local_provider():
    registry = ProviderRegistry()
    assert registry.resolve_model("ollama/qwen2.5-coder:3b") == (
        "ollama", "qwen2.5-coder:3b")
    assert isinstance(registry.resolve("ollama/qwen2.5-coder:3b"), OllamaProvider)


def test_registry_still_rejects_unknown_labels():
    registry = ProviderRegistry()
    with pytest.raises(ValueError, match="Unsupported model label"):
        registry.resolve_model("ollama-local")
    with pytest.raises(ValueError, match="must include a model name"):
        registry.resolve_model("ollama/")


def test_ollama_runtime_can_be_injected_but_others_cannot():
    provider = OllamaProvider(base_url="http://127.0.0.1:1")
    registry = ProviderRegistry({"ollama": provider})
    assert registry.resolve("ollama/llama3.2:3b") is provider
    with pytest.raises(ValueError, match="Only the 'claude-code' runtime"):
        ProviderRegistry({"google": provider})


# ---------------------------------------------------------------------------
# Request format / response parsing
# ---------------------------------------------------------------------------

def test_chat_request_format_and_response_parsing(fake_ollama):
    server = fake_ollama()
    client = MultiModelClient(registry=ProviderRegistry(
        {"ollama": OllamaProvider(base_url=server.url)}))

    response = client.call_model_response(
        model="ollama/qwen2.5-coder:3b", prompt="say hello", temperature=0.1,
        max_tokens=64, system="be terse",
    )

    assert response.provider == "ollama"
    assert response.model == "qwen2.5-coder:3b"
    assert response.text == "LOCAL_OK"
    assert response.input_tokens == 11
    assert response.output_tokens == 3
    assert response.cost_usd == 0.0

    request = server.requests[-1]
    assert request["path"] == "/api/chat"
    body = request["body"]
    assert body["model"] == "qwen2.5-coder:3b"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.1, "num_predict": 64}
    assert body["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "say hello"},
    ]


def test_direct_provider_omits_num_predict_until_explicit():
    from providers.model_provider import ModelRequest

    provider = OllamaProvider(base_url="http://127.0.0.1:1")
    payload = provider.build_payload(ModelRequest(prompt="hi", model="qwen3:4b"))
    assert payload["options"] == {"temperature": 0.7}
    payload = provider.build_payload(
        ModelRequest(prompt="hi", model="qwen3:4b", max_tokens=32))
    assert payload["options"]["num_predict"] == 32


def test_prior_messages_use_assistant_role(fake_ollama):
    from providers.model_provider import ModelRequest

    server = fake_ollama()
    provider = OllamaProvider(base_url=server.url)
    provider.generate(ModelRequest(
        prompt="and then?",
        model="qwen3:4b",
        messages=({"role": "user", "content": "first"}, {"role": "model", "content": "ok"}),
    ))
    assert server.requests[-1]["body"]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "and then?"},
    ]


def test_think_flag_is_forwarded_only_when_configured(fake_ollama):
    server = fake_ollama()
    from providers.model_provider import ModelRequest

    payload = OllamaProvider(base_url=server.url).build_payload(
        ModelRequest(prompt="hi", model="deepseek-r1:7b"))
    assert "think" not in payload

    payload = OllamaProvider(base_url=server.url, think=False).build_payload(
        ModelRequest(prompt="hi", model="deepseek-r1:7b"))
    assert payload["think"] is False


def test_num_ctx_is_forwarded_when_configured(monkeypatch):
    from providers.model_provider import ModelRequest
    from providers.ollama import OLLAMA_NUM_CTX_ENV

    payload = OllamaProvider(num_ctx=8192).build_payload(
        ModelRequest(prompt="hi", model="qwen3:4b"))
    assert payload["options"]["num_ctx"] == 8192

    monkeypatch.setenv(OLLAMA_NUM_CTX_ENV, "16384")
    assert OllamaProvider().num_ctx == 16384
    monkeypatch.setenv(OLLAMA_NUM_CTX_ENV, "not-a-number")
    assert OllamaProvider().num_ctx is None

    with pytest.raises(ValueError, match="num_ctx"):
        OllamaProvider(num_ctx=0)


def test_think_env_override(monkeypatch):
    from providers.ollama import OLLAMA_THINK_ENV

    monkeypatch.setenv(OLLAMA_THINK_ENV, "0")
    assert OllamaProvider().think is False
    monkeypatch.setenv(OLLAMA_THINK_ENV, "1")
    assert OllamaProvider().think is True
    monkeypatch.setenv(OLLAMA_THINK_ENV, "auto")
    assert OllamaProvider().think is None


def test_thinking_is_used_when_content_is_empty(fake_ollama):
    server = fake_ollama(content="", thinking="SELECTED: Review 2")
    client = MultiModelClient(registry=ProviderRegistry(
        {"ollama": OllamaProvider(base_url=server.url)}))
    response = client.call_model_response(model="ollama/deepseek-r1:7b", prompt="pick")
    assert response.text == "SELECTED: Review 2"


def test_malformed_or_incomplete_response_is_an_error(fake_ollama):
    server = fake_ollama(payload={"model": "qwen3:4b"})
    client = MultiModelClient(registry=ProviderRegistry(
        {"ollama": OllamaProvider(base_url=server.url)}))
    with pytest.raises(RuntimeError, match="message"):
        client.call_model_response(model="ollama/qwen3:4b", prompt="hi")


def test_http_error_is_reported_with_status(fake_ollama):
    server = fake_ollama(status=500, payload={"error": "model not found"})
    client = MultiModelClient(registry=ProviderRegistry(
        {"ollama": OllamaProvider(base_url=server.url)}))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        client.call_model_response(model="ollama/missing:1b", prompt="hi")


# ---------------------------------------------------------------------------
# Ollama down
# ---------------------------------------------------------------------------

def test_connection_refused_reports_actionable_error():
    url = f"http://127.0.0.1:{free_port()}"
    client = MultiModelClient(registry=ProviderRegistry(
        {"ollama": OllamaProvider(base_url=url)}))

    with pytest.raises(RuntimeError) as excinfo:
        client.call_model_response(model="ollama/qwen2.5-coder:3b", prompt="hi", timeout=1.0)

    message = str(excinfo.value)
    assert "not reachable" in message
    assert "ollama serve" in message


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_list_models_reads_tags(fake_ollama):
    server = fake_ollama()
    provider = OllamaProvider(base_url=server.url)
    assert provider.list_models() == ["qwen2.5-coder:3b", "deepseek-r1:7b"]
