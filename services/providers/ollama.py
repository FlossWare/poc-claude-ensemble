"""Local Ollama provider.

Ollama serves an OpenAI-compatible chat API on the operator's machine.  This
adapter keeps local models inside the same provider-neutral contract as the
Claude Code runtime, so ``MultiModelClient`` (and therefore the review crew,
the arbiter, and the Thompson outcome sync) can call a local model exactly the
way it calls a hosted one.

Model labels use the ``ollama/`` prefix:
``ollama/qwen2.5-coder:3b`` is sent to Ollama as ``qwen2.5-coder:3b``.

Cost is always ``$0.00`` -- the model runs on local hardware.  Token counts
come from Ollama's own ``prompt_eval_count`` / ``eval_count`` fields when the
server reports them.

If Ollama is not running, ``generate`` raises ``RuntimeError`` with an
actionable message; callers (for example the crew tool) treat that like any
other failed worker instead of hanging or silently substituting a model.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

from .model_capabilities import ModelCapabilities
from .model_provider import ModelProvider, ModelRequest, ModelResponse

DEFAULT_OLLAMA_URL = "http://localhost:11434"
# Environment overrides, consistent with the other POC_CLAUDE_ENSEMBLE_* knobs.
OLLAMA_URL_ENV = "POC_CLAUDE_ENSEMBLE_OLLAMA_URL"
# "0"/"1" force reasoning off/on for every local call; unset/other omits the
# field so Ollama uses the model default.  Reasoning models (deepseek-r1) can
# spend the whole output budget thinking on CPU, so callers that need a concise
# answer set this to 0.
OLLAMA_THINK_ENV = "POC_CLAUDE_ENSEMBLE_OLLAMA_THINK"
# Optional context window. Ollama defaults to 4096 for many models; long review
# or replay prompts need more or the beginning (including instructions) is
# truncated. Unset/0 omits the field and keeps the model default.
OLLAMA_NUM_CTX_ENV = "POC_CLAUDE_ENSEMBLE_OLLAMA_NUM_CTX"


class OllamaProvider(ModelProvider):
    """Invoke a local model through Ollama's ``/api/chat`` endpoint."""

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        think: bool | None = None,
        num_ctx: int | None = None,
    ) -> None:
        configured = base_url or os.environ.get(OLLAMA_URL_ENV) or DEFAULT_OLLAMA_URL
        self.base_url = configured.rstrip("/")
        self.default_timeout = timeout
        if think is None:
            raw = os.environ.get(OLLAMA_THINK_ENV, "").strip().lower()
            if raw in {"0", "false", "off", "no"}:
                think = False
            elif raw in {"1", "true", "on", "yes"}:
                think = True
        self.think = think
        if num_ctx is None:
            raw_ctx = os.environ.get(OLLAMA_NUM_CTX_ENV, "").strip()
            if raw_ctx.isdigit():
                num_ctx = int(raw_ctx) or None
        if num_ctx is not None and num_ctx < 1:
            raise ValueError("num_ctx must be a positive integer")
        self.num_ctx = num_ctx

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            provider=self.name,
            supports_system_prompt=True,
            supports_messages=True,
            supports_temperature_override=True,
            supports_max_tokens_override=True,
            supports_timeout_override=True,
        )

    # -- request/response -------------------------------------------------

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.capabilities.validate(request)
        model = request.model
        if not isinstance(model, str) or not model.strip():
            raise RuntimeError("ollama model name must be a non-empty string")

        payload = self.build_payload(request)
        started = time.monotonic()
        data = self._post_json("/api/chat", payload, timeout=request.timeout)
        latency_ms = (time.monotonic() - started) * 1000
        return self.parse_response(data, requested_model=model, latency_ms=latency_ms)

    def build_payload(self, request: ModelRequest) -> dict[str, Any]:
        """The exact JSON body sent to Ollama (exported for tests)."""
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        for message in request.conversation():
            role = message.get("role")
            messages.append({
                "role": "assistant" if role == "model" else role,
                "content": message.get("content", ""),
            })
        options: dict[str, Any] = {"temperature": request.temperature}
        # Ollama calls the output cap num_predict; -1 means unlimited. Only send
        # it when the caller explicitly chose a max_tokens value, matching the
        # capability contract (which advertises override support).
        if "max_tokens" in request.explicit_fields:
            options["num_predict"] = request.max_tokens
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if self.think is not None:
            payload["think"] = self.think
        return payload

    @staticmethod
    def parse_response(
        data: Any,
        *,
        requested_model: str,
        latency_ms: float | None = None,
    ) -> ModelResponse:
        """Normalize an Ollama ``/api/chat`` response."""
        if not isinstance(data, Mapping):
            raise RuntimeError("Ollama returned a non-object JSON response")
        message = data.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("Ollama response does not contain a 'message' object")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Ollama response does not contain string message content")
        thinking = message.get("thinking")
        text = content
        if not text.strip() and isinstance(thinking, str):
            # Reasoning models may put the answer in `thinking` when they did
            # not emit a separate final channel. Preserve the text rather than
            # turning a usable local call into an empty response.
            text = thinking

        return ModelResponse(
            provider="ollama",
            model=str(data.get("model") or requested_model),
            text=text,
            input_tokens=_nonnegative_int(data.get("prompt_eval_count")),
            output_tokens=_nonnegative_int(data.get("eval_count")),
            cost_usd=0.0,
            latency_ms=latency_ms,
            raw_metadata=dict(data),
        )

    # -- transport --------------------------------------------------------

    def _post_json(self, path: str, payload: Mapping[str, Any], *, timeout: float) -> Any:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method="POST",
            headers={"content-type": "application/json", "accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:2000]
            except Exception:  # noqa: BLE001 - error reporting is best effort
                pass
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code} for model {payload.get('model')!r}"
                + (f": {detail}" if detail else "")
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(self._connection_error(exc)) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                f"Ollama call timed out after {timeout:.1f}s at {self.base_url}; "
                "increase the timeout for large local models"
            ) from exc
        except OSError as exc:
            # Connection reset / broken pipe while Ollama is stopping or
            # restarting: still an actionable local-runtime failure.
            raise RuntimeError(
                f"Ollama request to {self.base_url} failed: {exc}"
            ) from exc

        try:
            return json.loads(raw or b"null")
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned malformed JSON") from exc

    def _connection_error(self, exc: urllib.error.URLError) -> str:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (ConnectionRefusedError, ConnectionResetError)) or (
            isinstance(reason, OSError) and getattr(reason, "errno", None) == 111
        ):
            return (
                f"Ollama is not reachable at {self.base_url} (connection refused); "
                "start it with `ollama serve` or set "
                f"{OLLAMA_URL_ENV}"
            )
        return f"Ollama request to {self.base_url} failed: {reason or exc}"

    def list_models(self, *, timeout: float = 5.0) -> list[str]:
        """Names of locally available Ollama models (``GET /api/tags``)."""
        request = urllib.request.Request(
            self.base_url + "/api/tags", method="GET",
            headers={"accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read() or b"{}")
        except urllib.error.URLError as exc:
            raise RuntimeError(self._connection_error(exc)) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(f"Ollama /api/tags timed out after {timeout:.1f}s") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama /api/tags returned malformed JSON") from exc
        models = data.get("models") if isinstance(data, Mapping) else None
        if not isinstance(models, list):
            return []
        return [
            str(item.get("name"))
            for item in models
            if isinstance(item, Mapping) and item.get("name")
        ]


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))
