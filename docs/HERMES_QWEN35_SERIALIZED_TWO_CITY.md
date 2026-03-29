# Hermes Qwen3.5 Serialized Two-City Compatibility

> Status: `implemented_early` on 2026-03-28 for a retained one-consumer-GPU
> serialized Hermes tool loop on native Psionic `chat.completions`.

This document records the retained serialized two-city Hermes compatibility
proof for native Psionic `qwen35`.

This lane exists because the strict direct-compatibility matrix is still red on
same-turn parallel tool batching. That is a real boundary, but it is not the
same thing as proving Hermes cannot complete a practical two-tool workflow on
one consumer GPU. This checker isolates the overlooked local path:

- `parallel_tool_calls = false`
- two city tools with disjoint schemas and handlers
- one tool call turn for Paris
- one later tool call turn for Tokyo
- one no-tool synthesis turn after both tool results are replayed

## Canonical Checker

Run the retained checker from the repo root:

```bash
scripts/release/check-psionic-hermes-qwen35-serialized-two-city.sh
```

## Exact Retained Command

```bash
TMPDIR=/home/christopherdavid/scratch/tmp/hermes-serialized-two-city \
PSIONIC_HERMES_SKIP_BUILD=1 \
PSIONIC_HERMES_ROOT=/home/christopherdavid/scratch/hermes-agent-proof2 \
PSIONIC_HERMES_PYTHON=/home/christopherdavid/scratch/hermes-min/.venv/bin/python \
PSIONIC_HERMES_SERVER_BIN=/home/christopherdavid/.cache/psionic-hermes-target/debug/psionic-openai-server \
PSIONIC_HERMES_QWEN35_MODEL_PATH=/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf \
PSIONIC_HERMES_SERIALIZED_TWO_CITY_REPORT_PATH=/home/christopherdavid/scratch/psionic-hermes-serialized-two-city/fixtures/qwen35/hermes/hermes_qwen35_serialized_two_city_report_20260328_archlinux_2b.json \
PSIONIC_HERMES_SERVER_LOG_PATH=/home/christopherdavid/scratch/logs/hermes_qwen35_serialized_two_city_20260328_archlinux_2b.log \
scripts/release/check-psionic-hermes-qwen35-serialized-two-city.sh
```

## Retained Report

- `fixtures/qwen35/hermes/hermes_qwen35_serialized_two_city_report_20260328_archlinux_2b.json`

## Fixed Contract

The retained row keeps the following fixed:

- same host: `archlinux`
- same Hermes revision used by the direct-compatibility work
- native Psionic `chat.completions`
- local consumer GPU qwen row
- `temperature = 0`
- `seed = 0`
- `parallel_tool_calls = false`

What changes from the red strict-parallel row:

- the tool loop is allowed to span multiple assistant turns
- the request stays in `tool_choice = required` until both tool results have
  been replayed
- after a city result has been replayed, the checker prunes that completed tool
  from the next offered tool list instead of leaving Hermes stuck with a stale
  already-completed option
- after the tool loop is complete, the checker runs one no-tool synthesis turn
  to render the final answer from the retained tool-result text
- the success criterion is practical serialized workflow completion rather than
  one same-turn batched `tool_calls` array

## Acceptance Bar

This row passes only if:

- Hermes calls `get_paris_weather` exactly once
- Hermes later calls `get_tokyo_weather` exactly once
- both tool result turns are present and mapped back to the correct tool names
- the final assistant answer mentions both Paris and Tokyo

## Why This Matters

The retained direct-compatibility matrix still matters. Same-turn parallel tool
calling is still red and still blocks full strict parity. But local
consumer-GPU work should not collapse into a single binary claim that Hermes
either clears every strict row or is unusable.

This serialized row answers a narrower and more operational question:

- can native Psionic on one consumer GPU complete a real two-tool Hermes loop
  with honest tool replay and a truthful final answer?

That is the right next local proof before escalating to stronger rows or bigger
GPUs.
