# Psionic MLX Text Serve Package

This document defines the bounded `psionic-mlx-serve` package that closes
`PMLX-703`.

## Scope

`psionic-mlx-serve` is the MLX-facing text-serving package above
`psionic-mlx-catalog` and `psionic-serve`.

It exists to make the already-landed Psionic-owned generic OpenAI-compatible
server usable through MLX-style references instead of raw artifact paths.

The package owns:

- MLX-style model-reference resolution for served text models
- package-owned bootstrap reports for `/v1/chat/completions` and
  `/v1/responses`
- explicit response-state storage selection for prompt-replay continuation
- bounded CLI entrypoints for planning and serving

The package does not own:

- multimodal or audio request handling
- Gradio demos, desktop pickers, or product UX
- fake hot-load or unload claims beyond what `psionic-serve` actually exposes

## Current Truth

`psionic-mlx-serve` resolves model references through `psionic-mlx-catalog`,
loads direct GGUF text runtimes through the shared MLX text lane, and then
boots the reusable `psionic-serve::OpenAiCompatServer`.

That means the real serving semantics come from the shared Psionic server path:

- `/v1/chat/completions`
- `/v1/responses`
- streaming and non-streaming text generation
- tool calling with explicit `none` / `auto` / `required` / named modes
- structured output via the shared fallback matcher
- logprobs and stop-sequence handling
- explicit prompt-replay response-state truth
- shared-prefix cache reuse truth

The package bootstrap report must remain explicit about the current lifecycle
posture:

- `load_status = loaded`
- `warm_control = not_implemented`
- `unload_control = not_implemented`
- `memory_pressure_reporting = not_implemented`

So this package closes the MLX-facing entrypoint gap without pretending the
generic server already supports live hot-load or unload control.

## Response State

`psionic-mlx-serve` supports the same prompt-replay response-state store modes
as the shared server:

- in-memory process-lifetime storage
- JSON-file-backed best-effort local durable storage

Those modes must be surfaced in the bootstrap report using the
`ResponseStateCapability` emitted by `psionic-router`.

## Reference Rules

The package accepts the same model references as `psionic-mlx-catalog`:

- direct local GGUF paths
- local Ollama model references
- local Hugging Face cache references

Only direct GGUF-resolvable references can be served directly today. If a
reference resolves to a conversion entrypoint or refused metadata posture
instead of one direct GGUF path, the package must fail explicitly rather than
hiding conversion behind an implicit side effect.

## CLI

The package CLI is:

- `psionic-mlx-serve plan`
- `psionic-mlx-serve serve`

`plan` emits the machine-readable bootstrap report without binding a server.
`serve` writes the same report optionally and then starts the shared
OpenAI-compatible router.

## Non-Goals

This package is not the place to duplicate serving logic already owned by
`psionic-serve`.

It should keep one runtime truth:

- resolve MLX-style references here
- serve through the shared Psionic server
- report current unsupported lifecycle controls honestly
