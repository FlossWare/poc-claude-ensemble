"""Model routing for Claude Ensemble.

Hosted model work runs through the installed Claude Code CLI.  Local models run
through Ollama (``ollama/<name>`` labels) or the OpenVINO serving layer on the
Arc iGPU (``local/<served-model>`` labels).  Direct Anthropic and Google API
adapters have been removed from this branch; the prior implementation is
preserved on the archive branch.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from .claude_code import ClaudeCodeProvider
from .local_serving import LOCAL_PREFIX, LOCAL_RUNTIME, LocalServingProvider
from .model_provider import ModelProvider
from .ollama import OllamaProvider

CLAUDE_CODE_RUNTIME = "claude-code"
OLLAMA_RUNTIME = "ollama"
OLLAMA_PREFIX = "ollama/"
SUPPORTED_RUNTIMES = frozenset({CLAUDE_CODE_RUNTIME, OLLAMA_RUNTIME, LOCAL_RUNTIME})


class ProviderRegistry:
    """Routes Claude labels to Claude Code, ``ollama/*`` to Ollama, and
    ``local/*`` to the local OpenVINO serving layer."""

    def __init__(
        self,
        providers: Mapping[str, ModelProvider] | None = None,
        credentials: object | None = None,
    ) -> None:
        # Keep the old constructor shape for callers, but reject alternative
        # runtimes instead of silently allowing multi-provider dispatch.
        self.providers: dict[str, ModelProvider] = {}
        for name, provider in (providers or {}).items():
            if name not in SUPPORTED_RUNTIMES:
                raise ValueError(
                    f"Only the 'claude-code' runtime is supported (plus local "
                    f"'ollama' and 'local'); got {name!r}"
                )
            self.providers[name] = provider
        self._configured_provider = os.environ.get(
            "POC_CLAUDE_ENSEMBLE_CLAUDE_CODE_EXPECTED_PROVIDER", "vertex"
        ).strip().lower()
        if not self._configured_provider:
            raise ValueError(
                "POC_CLAUDE_ENSEMBLE_CLAUDE_CODE_EXPECTED_PROVIDER must not be blank"
            )

    def _provider(self) -> ModelProvider:
        if CLAUDE_CODE_RUNTIME not in self.providers:
            self.providers[CLAUDE_CODE_RUNTIME] = ClaudeCodeProvider(
                expected_provider=self._configured_provider
            )
        return self.providers[CLAUDE_CODE_RUNTIME]

    def _ollama_provider(self) -> ModelProvider:
        if OLLAMA_RUNTIME not in self.providers:
            self.providers[OLLAMA_RUNTIME] = OllamaProvider()
        return self.providers[OLLAMA_RUNTIME]

    def _local_provider(self) -> ModelProvider:
        if LOCAL_RUNTIME not in self.providers:
            self.providers[LOCAL_RUNTIME] = LocalServingProvider()
        return self.providers[LOCAL_RUNTIME]

    def register(self, name: str, provider: ModelProvider) -> None:
        if name not in SUPPORTED_RUNTIMES:
            raise ValueError(
                f"Only the 'claude-code' runtime is supported (plus local "
                f"'ollama' and 'local'); got {name!r}"
            )
        self.providers[name] = provider

    def resolve(self, model: str) -> ModelProvider:
        runtime, _ = self.resolve_model(model)
        if runtime == OLLAMA_RUNTIME:
            return self._ollama_provider()
        if runtime == LOCAL_RUNTIME:
            return self._local_provider()
        return self._provider()

    def resolve_model(self, model: str) -> tuple[str, str]:
        """Map a model label to ``(runtime, canonical model name)``.

        ``ollama/<name>`` strips the prefix and returns ``("ollama", "<name>")``;
        ``local/<name>`` returns ``("local", "<name>")`` for the OpenVINO
        serving layer.  Claude labels are preserved and rejected provider
        labels raise.  Never silently substitute another model for an
        unsupported request.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        normalized = model.strip()
        lowered = normalized.lower()

        if lowered == OLLAMA_RUNTIME or lowered.startswith(OLLAMA_PREFIX):
            ollama_model = normalized[len(OLLAMA_PREFIX):].strip()
            if not ollama_model:
                raise ValueError(
                    "ollama model label must include a model name, e.g. "
                    "'ollama/qwen2.5-coder:3b'"
                )
            self._ollama_provider()
            return OLLAMA_RUNTIME, ollama_model

        if lowered == LOCAL_RUNTIME or lowered.startswith(LOCAL_PREFIX):
            local_model = normalized[len(LOCAL_PREFIX):].strip()
            if not local_model:
                raise ValueError(
                    "local serving model label must include a model name, e.g. "
                    "'local/qwen3-coder-30b-a3b'"
                )
            provider = self._local_provider()
            # Fail at resolve time when the label has no configured endpoint,
            # rather than at the first (expensive) generation call.
            if isinstance(provider, LocalServingProvider):
                provider.base_url(local_model)
            return LOCAL_RUNTIME, local_model

        aliases = {"haiku", "sonnet", "opus"}
        if lowered in aliases:
            canonical = lowered
        elif lowered.startswith(("claude-", "claude_", "rits/")):
            canonical = normalized
        else:
            raise ValueError(f"Unsupported model label: {normalized!r}")

        self._provider()
        return CLAUDE_CODE_RUNTIME, canonical

    def credential_status(self, provider: str) -> list[dict[str, object]]:
        """Return no API-key accounts; Claude Code owns authentication."""
        if provider in {"claude-code", "anthropic", "google"}:
            return []
        return []
