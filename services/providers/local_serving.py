"""Local OpenVINO serving-layer provider (Arc iGPU MoE models).

The ThinkPad P1 Gen 8 serving layer (``thinkpad-p1-gen8/serving/server.py
--device GPU``) exposes an **Anthropic-shaped** ``POST /v1/messages`` endpoint
for one model per process.  This adapter lets the ensemble call those models
exactly the way it calls Ollama or the hosted runtime, so the MoE models become
ordinary Thompson arms.

Model labels use the ``local/`` prefix and the *served* model id:
``local/qwen3-coder-30b-a3b``.  The label maps to a loopback base URL through
:data:`DEFAULT_ENDPOINTS` (override with
``POC_CLAUDE_ENSEMBLE_LOCAL_SERVING_URLS`` as a JSON object).

Cost is always ``$0.00`` -- the model runs on local hardware.  Token counts
come from the shim's ``usage`` block.

The shim is text-only: it rejects ``tools`` explicitly, so this provider never
sends them.  If the serving layer is not running, ``generate`` raises
``RuntimeError`` with the exact command to start it; callers treat that like
any other failed worker instead of substituting a model.
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

LOCAL_RUNTIME = "local"
LOCAL_PREFIX = "local/"

#: Served model id -> loopback base URL of its serving-layer process.
DEFAULT_ENDPOINTS: dict[str, str] = {
    "qwen3-coder-30b-a3b": "http://127.0.0.1:8931",
    "qwen3.6-35b-a3b": "http://127.0.0.1:8932",
}

ENDPOINTS_ENV = "POC_CLAUDE_ENSEMBLE_LOCAL_SERVING_URLS"

#: Only one large MoE fits on the Arc iGPU at a time (the second compile
#: segfaults with the first resident).  When a request arrives for a model
#: whose server is not running, the provider can stop the other MoE server,
#: start the requested one, and retry -- "swap" instead of "fail".  Registry:
#: served model id -> OpenVINO IR directory on this host.
DEFAULT_MODEL_IRS: dict[str, str] = {
    "qwen3-coder-30b-a3b": "~/.FlossWare/models/llm/qwen3-coder-30b-a3b-instruct-int4-ov",
    "qwen3.6-35b-a3b": "~/.FlossWare/models/llm/qwen3.6-35b-a3b-int4-ov",
}
MODELS_ENV = "POC_CLAUDE_ENSEMBLE_LOCAL_SERVING_MODELS"
AUTOSWAP_ENV = "POC_CLAUDE_ENSEMBLE_LOCAL_SERVING_AUTOSWAP"
SERVING_REPO = "~/Development/github/FlossWare/thinkpad-p1-gen8"
SWAP_LOCK_PATH = "/tmp/poc-claude-ensemble-moe-swap.lock"
SWAP_STARTUP_TIMEOUT_S = 180.0


def _model_ir_map() -> dict[str, str]:
    """Default IR registry plus any JSON override from the environment."""
    irs = {k: os.path.expanduser(v) for k, v in DEFAULT_MODEL_IRS.items()}
    raw = os.environ.get(MODELS_ENV, "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{MODELS_ENV} must be a JSON object: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{MODELS_ENV} must be a JSON object")
        for key, value in parsed.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(f"{MODELS_ENV} keys and values must be strings")
            if not key or not value.strip():
                raise ValueError(f"{MODELS_ENV} keys and values must be non-empty")
            irs[key] = os.path.expanduser(value)
    return irs


def _autoswap_enabled() -> bool:
    return os.environ.get(AUTOSWAP_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def _serving_env(ir: str) -> dict[str, str]:
    """Environment for the serving process, mirroring thinkpad env.sh.

    The serving layer must run against the pinned OpenVINO 2026.5 source
    build (``~/AI``), not the system wheel: the 35B MoE segfaults during
    pipeline compilation with the wrong runtime.  The pinned paths are
    deliberately placed FIRST in PYTHONPATH/LD_LIBRARY_PATH so they take
    precedence over any inherited values -- operator overrides for these
    two variables do not win, by design; ``FLOSSWARE_AI_ROOT`` is the
    supported relocation knob.
    """
    ai_root = os.environ.get("FLOSSWARE_AI_ROOT", os.path.expanduser("~/AI"))
    genai_py = f"{ai_root}/install/openvino-genai-2026.5/python"
    ov_py = f"{ai_root}/install/openvino-2026.5-python"
    genai_lib = f"{ai_root}/install/openvino-genai-2026.5/runtime/lib/intel64"
    env = dict(os.environ)
    env["NPU_MODEL_DIR"] = ir
    env["PYTHONPATH"] = f"{genai_py}:{ov_py}" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    env["LD_LIBRARY_PATH"] = f"{ov_py}/openvino/libs:{genai_lib}" + (
        f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""
    )
    env.setdefault("IE_NPU_COMPILER_TYPE", "PLUGIN")
    return env


class _ServingUnreachable(RuntimeError):
    """The serving process for a model is not listening (swap candidate)."""

    def __init__(self, base: str, model: Any, exc: BaseException):
        super().__init__(str(exc))
        self.base = base
        self.model = model


#: Anthropic shim limit (server.py rejects max_tokens outside [1, 4096]).
MAX_OUTPUT_TOKENS = 4096


def endpoint_map() -> dict[str, str]:
    """Default endpoints plus any JSON override from the environment."""
    endpoints = dict(DEFAULT_ENDPOINTS)
    raw = os.environ.get(ENDPOINTS_ENV, "").strip()
    if not raw:
        return endpoints
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{ENDPOINTS_ENV} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{ENDPOINTS_ENV} must be a JSON object")
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{ENDPOINTS_ENV} keys and values must be strings")
        if not key or not value.strip():
            raise ValueError(f"{ENDPOINTS_ENV} keys and values must be non-empty")
        endpoints[key] = value.rstrip("/")
    return endpoints


class LocalServingProvider(ModelProvider):
    """Invoke a local OpenVINO-served model through the Anthropic shim."""

    name = LOCAL_RUNTIME

    def __init__(
        self,
        *,
        endpoints: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        self.endpoints = dict(endpoints) if endpoints is not None else endpoint_map()
        self.default_timeout = timeout

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            provider=self.name,
            supports_system_prompt=True,
            supports_messages=True,
            # The shim generates greedily (do_sample=False) and ignores
            # temperature; advertising support would be a lie.
            supports_temperature_override=False,
            supports_max_tokens_override=True,
            supports_timeout_override=True,
        )

    # -- request/response -------------------------------------------------

    def base_url(self, model: str) -> str:
        url = self.endpoints.get(model)
        if not url:
            known = ", ".join(sorted(self.endpoints)) or "none configured"
            raise RuntimeError(
                f"no local serving endpoint for model {model!r} (known: {known}); "
                f"set {ENDPOINTS_ENV} to add one"
            )
        return url

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.capabilities.validate(request)
        model = request.model
        if not isinstance(model, str) or not model.strip():
            raise RuntimeError("local serving model name must be a non-empty string")

        payload = self.build_payload(request)
        base = self.base_url(model)
        timeout = request.timeout if self.default_timeout is None else min(
            request.timeout, self.default_timeout
        )
        started = time.monotonic()
        try:
            data = self._post_json(base, "/v1/messages", payload, timeout=timeout)
        except _ServingUnreachable as exc:
            hint = self._connection_error(base, model, exc)
            if not _autoswap_enabled():
                raise RuntimeError(hint) from exc
            # One large MoE fits on the iGPU: stop whatever is resident,
            # start the requested model, retry once.  Swap time is dominated
            # by pipeline compilation (~20-90 s).
            try:
                self._swap_to(model, base)
            except Exception as swap_exc:  # noqa: BLE001 - reported with the hint
                raise RuntimeError(
                    f"auto-swap to {model!r} failed: {swap_exc}. {hint}"
                ) from exc
            try:
                data = self._post_json(base, "/v1/messages", payload, timeout=timeout)
            except _ServingUnreachable as retry_exc:
                raise RuntimeError(
                    f"local serving layer still unreachable for {model!r} after auto-swap "
                    f"({retry_exc}). {hint}"
                ) from retry_exc
        latency_ms = (time.monotonic() - started) * 1000
        return self.parse_response(data, requested_model=model, latency_ms=latency_ms)

    def build_payload(self, request: ModelRequest) -> dict[str, Any]:
        """The exact JSON body sent to the Anthropic shim (exported for tests)."""
        max_tokens = (
            request.max_tokens
            if "max_tokens" in request.explicit_fields
            else MAX_OUTPUT_TOKENS
        )
        max_tokens = max(1, min(int(max_tokens), MAX_OUTPUT_TOKENS))
        messages: list[dict[str, str]] = []
        for message in request.conversation():
            role = message.get("role")
            messages.append({
                "role": "assistant" if role == "model" else role,
                "content": message.get("content", ""),
            })
        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt
        return payload

    @staticmethod
    def parse_response(
        data: Any,
        *,
        requested_model: str,
        latency_ms: float | None = None,
    ) -> ModelResponse:
        """Normalize an Anthropic Messages-API response."""
        if not isinstance(data, Mapping):
            raise RuntimeError("local serving layer returned a non-object JSON response")
        content = data.get("content")
        if not isinstance(content, list):
            raise RuntimeError(
                "local serving layer response does not contain a 'content' list"
            )
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "text"
        ]
        text = "\n".join(part for part in parts if part)
        usage = data.get("usage") if isinstance(data.get("usage"), Mapping) else {}
        return ModelResponse(
            provider=LOCAL_RUNTIME,
            model=str(data.get("model") or requested_model),
            text=text,
            input_tokens=_nonnegative_int(usage.get("input_tokens")),
            output_tokens=_nonnegative_int(usage.get("output_tokens")),
            cost_usd=0.0,
            request_id=str(data.get("id")) if data.get("id") else None,
            latency_ms=latency_ms,
            raw_metadata=dict(data),
        )

    # -- transport --------------------------------------------------------

    def _swap_to(self, model: str, base: str) -> None:
        """Stop the resident MoE server and start ``model`` (exclusive GPU).

        Serialized by an flock so concurrent requests cannot interleave
        swaps.  Failure at any step raises; the caller reports the manual
        start command as a fallback.
        """
        import fcntl
        import subprocess

        irs = _model_ir_map()
        ir = irs.get(model)
        if not ir:
            raise RuntimeError(f"no IR directory registered for {model!r}; set {MODELS_ENV}")
        if not os.path.isdir(ir):
            raise RuntimeError(f"registered IR directory does not exist: {ir}")
        repo = os.path.expanduser(SERVING_REPO)
        if not os.path.isfile(os.path.join(repo, "serving", "server.py")):
            raise RuntimeError(f"serving repo not found at {repo}")
        port = base.rsplit(":", 1)[-1]
        log_dir = os.path.expanduser("~/.FlossWare/logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"local-serving-{model}.log")

        # Anchored so only real server processes match -- a shell whose
        # command line merely *contains* the pattern (an editor, a monitor,
        # a verification loop) must never be killed.
        STOP_PATTERN = r"^python3 serving/server\.py --device GPU"
        with open(SWAP_LOCK_PATH, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            subprocess.run(["pkill", "-f", STOP_PATTERN], check=False)
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                probe = subprocess.run(
                    ["pgrep", "-f", STOP_PATTERN], capture_output=True, check=False
                )
                if probe.returncode != 0:
                    break
                time.sleep(0.5)
            # Give the driver a moment to release the iGPU after process
            # exit; compiling the next MoE too early segfaults.
            time.sleep(3.0)
            env = _serving_env(ir)
            with open(log_path, "ab") as log:
                subprocess.Popen(
                    ["python3", "serving/server.py", "--device", "GPU", "--port", port],
                    cwd=repo,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            # Wait for the new server to answer before retrying the request.
            deadline = time.monotonic() + SWAP_STARTUP_TIMEOUT_S
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(base + "/health", timeout=5) as response:
                        json.loads(response.read())
                        return
                except urllib.error.HTTPError:
                    return  # listening and answering; that is enough to retry
                except (urllib.error.URLError, OSError, ValueError):
                    time.sleep(2.0)
        raise RuntimeError(
            f"serving process for {model!r} did not become healthy within "
            f"{SWAP_STARTUP_TIMEOUT_S:.0f}s (log: {log_path})"
        )

    def health(self, model: str) -> Mapping[str, Any]:
        """Return the serving process's ``/health`` payload."""
        base = self.base_url(model)
        request = urllib.request.Request(base + "/health", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise RuntimeError(self._connection_error(base, model, exc)) from exc

    def _post_json(
        self, base: str, path: str, payload: Mapping[str, Any], *, timeout: float
    ) -> Any:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            base + path,
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
                f"local serving layer returned HTTP {exc.code} for model "
                f"{payload.get('model')!r}" + (f": {detail}" if detail else "")
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                f"local serving call timed out after {timeout:.1f}s at {base}; "
                "MoE decode on the Arc iGPU is ~15-30 tok/s, so long answers "
                "need a large --timeout"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise _ServingUnreachable(base, payload.get("model"), exc) from exc
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise RuntimeError(
                f"local serving layer returned invalid JSON from {base}{path}"
            ) from exc

    @staticmethod
    def _connection_error(base: str, model: Any, exc: BaseException) -> str:
        return (
            f"local serving layer at {base} is unreachable for model {model!r} "
            f"({exc}). Start it with: "
            "cd ~/Development/github/FlossWare/thinkpad-p1-gen8 && "
            f"NPU_MODEL_DIR=<IR dir for {model}> python3 serving/server.py "
            f"--device GPU --port {base.rsplit(':', 1)[-1]}"
        )


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)
