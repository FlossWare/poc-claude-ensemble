# Model Runtime

Claude Ensemble has three active model runtimes: the installed Claude Code CLI
for hosted Claude models, a local Ollama server for `ollama/<name>` labels, and
the local OpenVINO GPU serving layer for `local/<model>` labels.
Memory, Learning, Thompson, GA, Graph, Alert, Messenger, arbitration, and
collaboration remain independent systems; they are not provider implementations
and are not removed by this simplification.

## Runtime contract

`MultiModelClient` and `ModelRequest` / `ModelResponse` remain as compatibility
interfaces so existing orchestration, metrics, and learning code does not need
to be rewritten. `ProviderRegistry` sends Claude labels through
`ClaudeCodeProvider`, `ollama/<name>` labels through `OllamaProvider`, and
`local/<model>` labels through `LocalServingProvider`. It does not dispatch to
direct Anthropic, Google, Cursor, or OpenAI APIs.

Claude model aliases `haiku`, `sonnet`, and `opus` pass through to Claude Code.
Local models use the `ollama/` prefix; the prefix is stripped and the remainder
is sent to Ollama's `/api/chat` endpoint (default
`http://localhost:11434`, override with `POC_CLAUDE_ENSEMBLE_OLLAMA_URL`).
Calls are recorded at `$0.00` cost with Ollama's token counts. If Ollama is not
running, the call fails with an actionable error instead of hanging or
substituting another model.

Unsupported labels such as `gemini`, `gpt-4`, `cursor`, and `ollama-local`
are rejected. CE never silently substitutes a different model for an
unsupported request. Gemini routing, when available on a later explicit-routing
branch, must remain opt-in and policy-gated.

Reasoning models can spend the entire output budget thinking on CPU. Set
`POC_CLAUDE_ENSEMBLE_OLLAMA_THINK=0` (or construct `OllamaProvider(think=False)`)
to send `"think": false`; leave it unset to use the model's default.

### Local OpenVINO serving layer (`local/` — Arc iGPU MoE models)

`local/<served-model>` labels go to `LocalServingProvider`, which POSTs the
Anthropic-shaped `POST /v1/messages` endpoint of the ThinkPad P1 Gen 8 serving
layer (`thinkpad-p1-gen8/serving/server.py --device GPU`). One process serves
one model, so one arm == one process == one loopback port:

| Arm / model id | Port | OpenVINO IR |
|---|---|---|
| `local/qwen3-coder-30b-a3b` | 8931 | `~/.FlossWare/models/llm/qwen3-coder-30b-a3b-instruct-int4-ov` |
| `local/qwen3.6-35b-a3b` | 8932 | `~/.FlossWare/models/llm/qwen3.6-35b-a3b-int4-ov` |

Start a model (the serving layer is not managed by a systemd unit):

```bash
cd ~/Development/github/FlossWare/thinkpad-p1-gen8
NPU_MODEL_DIR=$HOME/.FlossWare/models/llm/qwen3-coder-30b-a3b-instruct-int4-ov \
  python3 serving/server.py --device GPU --port 8931
```

`qwen3.6-35b-a3b` is a multimodal `Qwen3_5MoeForConditionalGeneration` export.
The OpenVINO GenAI 2026.2 wheel segfaults on it and the 2026.5 `LLMPipeline`
loads it but cannot infer (`Port for tensor name input_ids was not found`), so
**that arm is registered but not servable through `/v1/messages` today**; it
runs only through GenAI 2026.5 `VLMPipeline` (≈25 tok/s on the Arc iGPU). Keep
it excluded from crews (`--exclude local/qwen3.6-35b-a3b`) until a VLM shim
exists.

Overrides: `POC_CLAUDE_ENSEMBLE_LOCAL_SERVING_URLS` is a JSON object of
`{model_id: base_url}`. Calls cost `$0.00`. An unreachable port raises a
`RuntimeError` naming the exact start command; the model is never substituted.
The shim is text-only — it rejects `tools`, and it generates greedily, so
`supports_temperature_override` is `False`.

## Authentication and data policy

Claude Code owns authentication. CE does not read or copy credentials from the
Claude configuration directory. The CLI inherits the service process
environment unless an explicit provider environment is supplied.

Before sending a prompt, the adapter runs `claude auth status` and requires
`apiProvider` to match `POC_CLAUDE_ENSEMBLE_CLAUDE_CODE_EXPECTED_PROVIDER`. If status cannot
be read or does not match, the prompt is not sent. The adapter also checks the
provider metadata reported in the response. The default expectation is
`vertex`; set an explicit value only when the target data is authorized for
that upstream. **For Red Hat internal code, keep the expected provider set to
`vertex`; do not route that code through Anthropic or another upstream.**

Do not change the user's existing `~/.claude` authentication configuration as
part of installing CE. Profile and hook installation are separate from model
runtime selection.

## Smoke test

With Claude Code installed and authenticated to the permitted upstream:

```bash
python -c 'from arbitration.api_client import MultiModelClient; print(MultiModelClient().call_model("sonnet", "Reply with one word."))'
python -c 'from arbitration.api_client import MultiModelClient; print(MultiModelClient().call_model("ollama/qwen2.5-coder:3b", "Reply with one word."))'
```

The automated tests mock Claude Code and Ollama and do not require live
credentials or a running local server. Local model data never leaves the
machine; the Ollama adapter performs no external network calls beyond the
configured loopback URL.


## Claude Code adapter request limitations

The active CLI adapter accepts a self-contained prompt and optional system_prompt.
It does not currently implement multi-turn messages, temperature overrides, or
output-token limits. To avoid silently changing the meaning of a request, it
rejects non-empty messages, non-default temperature, and non-default max_tokens
before starting the CLI process. The ModelRequest default values for temperature
and token count are compatibility defaults, not a claim that the CLI enforces
those settings. Callers needing these controls must not assume they are applied
by this adapter.
