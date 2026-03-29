# Hermes Qwen3.5 Compatibility

> Status: `retained_consumer_gpu_full_compatibility` on 2026-03-29 for a full
> repo-owned Hermes `chat.completions` proof on native Psionic `qwen35`.

This document records the canonical retained Hermes compatibility proof for
native Psionic `qwen35` on one consumer GPU.

The current honest boundary is:

- native Psionic now closes the full Hermes acceptance matrix `6/6` on the
  local `archlinux` `RTX 4080` lane
- the previously red same-turn `parallel_tool_turn` case is now green in the
  full checker on `2b`
- the same strict parallel row is also green in the separate attribution
  matrix on `2b`, `4b`, and `9b` against both Psionic and Ollama on the same
  host
- the older `5/6` retained reports remain useful historical evidence, but they
  are no longer the canonical current boundary

## Canonical Retained Revisions

- Psionic revision:
  `f4788f38cc04febf5d9e9eb526694de048ceabc2`
- Hermes revision:
  `e295a2215acd55f2ee930fc7a4cd2df1c5464234`
- Host:
  `archlinux`

## Canonical Checker

Run the retained checker from the repo root:

```bash
scripts/release/check-psionic-hermes-qwen35-compatibility.sh
```

Exact retained command:

```bash
TMPDIR=/home/christopherdavid/scratch/tmp/hermes-check-final \
PSIONIC_HERMES_SKIP_BUILD=1 \
PSIONIC_HERMES_ROOT=/home/christopherdavid/scratch/hermes-agent-proof2 \
PSIONIC_HERMES_PYTHON=/home/christopherdavid/scratch/hermes-min/.venv/bin/python \
PSIONIC_HERMES_SERVER_BIN=/home/christopherdavid/.cache/psionic-hermes-target/debug/psionic-openai-server \
PSIONIC_HERMES_QWEN35_MODEL_PATH=/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf \
PSIONIC_HERMES_REPORT_PATH=/home/christopherdavid/scratch/psionic-hermes-final/fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260329_archlinux_2b_f4788f38.json \
PSIONIC_HERMES_SERVER_LOG_PATH=/home/christopherdavid/scratch/logs/hermes_qwen35_psionic_compatibility_20260329_archlinux_2b_f4788f38.log \
scripts/release/check-psionic-hermes-qwen35-compatibility.sh
```

## Canonical Retained Reports

- full compatibility proof:
  `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260329_archlinux_2b_f4788f38.json`
- grounded rerun of the full checker with the stricter parallel-row acceptance:
  `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260329_archlinux_2b_grounded.json`
- strict same-turn parallel matrix:
  `fixtures/qwen35/hermes/hermes_qwen35_parallel_tool_attribution_matrix_20260329_archlinux.json`

Historical reports kept for comparison:

- `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_2b.json`
- `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_2b_b2277ed5.json`
- `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_2b_9f67d65b.json`
- `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_9b.json`
- `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_9b_9f67d65b.json`

## Acceptance Matrix

Canonical retained `2b` result on pushed `f4788f38`:

| Case | Result | Retained Summary |
| --- | --- | --- |
| `required_tool_turn` | `pass` | one required weather tool call emitted |
| `auto_plain_text_turn` | `pass` | plain text answer, no tool call |
| `multi_turn_tool_loop` | `pass` | tool replay reached final answer |
| `parallel_tool_turn` | `pass` | same-turn assistant response emitted both tools and grounded the final answer in their results |
| `invalid_argument_truthful_refusal` | `pass` | truthful unsupported-city refusal |
| `streamed_tool_turn` | `pass` | streamed tool-call turn preserved |

Overall:

- `overall_pass = true`
- `passing_case_count = 6`
- `total_case_count = 6`

## What Changed

The old red row was not resolved by pretending the local model family had
changed. The fix was in the native Psionic tool contract:

1. required/named tool turns now prefer a plain JSON-schema tool-call batch
   instead of relying on the older tagged wrapper shape
2. the required parallel contract now carries a real
   `minimum_required_tool_calls` floor
3. that floor is derived from backticked declared tool names in the request
   messages when `tool_choice = required` and `parallel_tool_calls = true`
4. the prompt contract now tells the model to emit at least that many tools on
   the required same-turn batch

That is why the canonical retained full checker now clears the formerly red
parallel row instead of stopping at `5/6`.

One additional correction matters here: the parallel row is no longer counted
green merely because the first assistant turn emitted both tools. The current
retained checker requires the final assistant message to land as the grounded
summary `Paris is sunny at 18C. Tokyo is rainy at 12C.` after those two tool
results are replayed.

## Relation To Other Hermes Docs

- strict same-turn parity receipts now live in
  `docs/HERMES_QWEN35_PARALLEL_ATTRIBUTION.md`
- same-host benchmark truth still lives in
  `docs/HERMES_BACKEND_BENCHMARK.md`
- serialized two-city proof still lives in
  `docs/HERMES_QWEN35_SERIALIZED_TWO_CITY.md`
- fast-path versus fallback runtime truth still lives in
  `docs/HERMES_QWEN35_FAST_PATH.md`
- repeated-loop warm-path truth still lives in
  `docs/HERMES_QWEN35_REUSE_BENCHMARK.md`

## Honest Bottom Line

It is now honest to say the repo retains a direct Hermes-on-Psionic
compatibility proof on one consumer GPU through the normal custom-provider
`chat.completions` path.

It is still not honest to say every Hermes question is solved:

- Ollama still wins wallclock on the older easy same-host benchmark rows
- `llama.cpp` is still not a runnable apples-to-apples comparator for this
  `qwen35` artifact contract on the current host
- raw-versus-registry artifact attribution on the local Ollama-managed `2b`
  blob is still permission-blocked

But the old direct compatibility blocker is no longer open. The retained
consumer-GPU Psionic lane is now green on the full Hermes acceptance matrix.
