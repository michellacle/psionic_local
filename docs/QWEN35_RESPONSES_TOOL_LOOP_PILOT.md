# Qwen35 Responses Tool-Loop Pilot

> Status: `implemented_early` on 2026-03-28 after landing the runnable checker
> in `scripts/release/check-psionic-qwen35-responses-tool-loop-pilot.sh`.

This document records the bounded native `qwen35` `/v1/responses` tool-loop
pilot for Psionic.

The pilot proves the stateful part of the tool contract that still matters
after `/v1/chat/completions` tool calling is already working:

- a native `qwen35` `/v1/responses` turn can store a tool-call assistant turn
- replayed tool results remain machine-distinguishable as `role = tool` with a
  preserved tool name
- prompt-replay continuation can return a final assistant answer after the tool
  result is replayed

## Canonical Runner

Run the pilot from the repo root:

```bash
scripts/release/check-psionic-qwen35-responses-tool-loop-pilot.sh
```

## Workload Shape

The pilot uses synthetic tiny native `qwen35` fixtures under the real Psionic
CUDA serving path.

The checker runs three evidence layers:

1. prompt-conversion coverage for `/v1/responses`
2. one stored native `qwen35` tool-call turn on `/v1/responses`
3. one replayed tool-result continuation that reaches a final assistant answer

The continuation evidence uses a second deterministic native `qwen35` fixture
for the final assistant-answer turn. That is intentional. The tiny fixtures
emit fixed turns so the pilot separates the tool-call turn from the
post-tool-result answer turn while still keeping both turns inside the native
`qwen35` runtime and the real prompt-replay response-state surface.

## Pass Criteria

The pilot is green only if all of the following remain true:

- `ResponsesInput::Messages` preserves assistant and tool replay turns through
  qwen35 prompt conversion
- `role = tool` replay stays machine-distinguishable via
  `PromptMessageRole::Tool`
- tool replay preserves the tool name on the stored prompt message
- a native `qwen35` `/v1/responses` tool turn stores response state and surfaces
  `psionic_tool_calls`
- the stored assistant turn remains the raw machine-readable tool envelope
- a continuation over replayed qwen35 prompt history reaches a final assistant
  answer
- the continuation records `previous_response_id`, stable `conversation.id`,
  and nonzero `psionic_response_state.replayed_prompt_messages`

## Expected Signals

The current pilot should surface these signals:

- tool-turn response:
  - `psionic_tool_calls[0].name = "get_weather"`
  - `psionic_response_state.stored = true`
  - stored prompt history ends with the assistant tool envelope
- replayed continuation:
  - stored history contains one `role = tool` prompt message
  - that tool prompt message keeps `author_name = "get_weather"`
  - `output_text = "Tomorrow will also be sunny."`
  - `previous_response_id` points at the seeded replay state
  - `conversation.revision = 2`

## Current Limitations

This pilot is intentionally bounded:

- it uses deterministic tiny native qwen35 fixtures instead of the downloaded
  public artifact
- it proves prompt-replay semantics, not broad agent-benchmark quality
- it does not claim streaming `/v1/responses`
- it does not claim marketplace or external tool-provider production readiness

Those bounds are acceptable for this pilot because the point is to prove the
native qwen35 response-state and tool-replay contract, not to claim a finished
agent product surface.
