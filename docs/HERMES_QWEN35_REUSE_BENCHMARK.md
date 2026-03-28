# Hermes Qwen3.5 Reuse Benchmark

> Status: `implemented_early` on 2026-03-28 for retained repeated-loop
> prefix-cache evidence on exact pushed Psionic.

This document records the retained repeated Hermes-equivalent reuse benchmark
for native Psionic `qwen35` on `/v1/chat/completions`.

The benchmark keeps the model and task fixed and compares:

- `psionic_prefix_cache = bypass`
- `psionic_prefix_cache = auto`

It uses the two repeated turn shapes that matter for Hermes-style tool loops:

- `required_tool_turn`
- `tool_result_continuation`

## Exact Revision

- Psionic revision:
  `a800435c4542336d011e36a0154d111228fdc5dd`
- Host:
  `archlinux`
- Model:
  `qwen3.5-2b-q8_0-registry.gguf`

## Canonical Runner

Run the benchmark from the repo root:

```bash
scripts/release/check-psionic-hermes-qwen35-reuse-benchmark.sh
```

The exact retained receipt used a copied clean tree on `archlinux`, so the
runner now accepts `PSIONIC_HERMES_PSIONIC_REVISION` for honest provenance in
ephemeral benchmark roots.

Exact retained command:

```bash
TMPDIR=/home/christopherdavid/scratch/tmp/hermes-reuse-a800435c \
CARGO_TARGET_DIR=/home/christopherdavid/scratch/psionic-hermes-reuse-a800435c/target \
PSIONIC_HERMES_SERVER_BIN=/home/christopherdavid/scratch/psionic-hermes-reuse-a800435c/target/debug/psionic-openai-server \
PSIONIC_HERMES_PSIONIC_REVISION=a800435c4542336d011e36a0154d111228fdc5dd \
PSIONIC_HERMES_QWEN35_MODEL_PATH=/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf \
PSIONIC_HERMES_REUSE_REPORT_PATH=/home/christopherdavid/scratch/psionic-hermes-reuse-a800435c/fixtures/qwen35/hermes/hermes_qwen35_reuse_benchmark_20260328_archlinux_2b.json \
scripts/release/check-psionic-hermes-qwen35-reuse-benchmark.sh
```

## Retained Report

- `fixtures/qwen35/hermes/hermes_qwen35_reuse_benchmark_20260328_archlinux_2b.json`

## Retained Result

The exact retained report is:

- `overall_pass = true`
- `improvement_status = improved`

Warm-path deltas:

| Case | Bypass warm wallclock | Auto warm wallclock | Improvement |
| --- | --- | --- | --- |
| `required_tool_turn` | `0.602263s` | `0.199582s` | `66.8613%` |
| `tool_result_continuation` | `0.852610s` | `0.201771s` | `76.3349%` |

Warm-path reuse:

| Case | Warm prefix tokens reused | Warm TTFT |
| --- | --- | --- |
| `required_tool_turn` | `135` | `0.000474ms` |
| `tool_result_continuation` | `215` | `0.000469ms` |

The important behavioral point is that the repeated exact-hit continuation now
stays on the assistant-message path across all retained iterations. The earlier
regression where a later exact-hit continuation collapsed back into a tool call
is gone on the retained `a800435c` row.

## What Changed

The retained green row required three concrete runtime changes:

- qwen35 exact-hit reuse now carries a retained first selected token for the
  repeated prompt, so the first decode step does not need to consult stale
  structured-output buffers
- the qwen35 shared-prefix store now deep-clones CUDA prompt state instead of
  shallow-cloning CUDA-backed KV buffers, so later decode steps cannot mutate
  the stored prompt snapshot in place
- the repo-owned reuse benchmark wrapper now accepts
  `PSIONIC_HERMES_PSIONIC_REVISION` so copied or ephemeral benchmark trees can
  retain honest revision provenance

The repo also now has a server-level regression test for the repeated
tool-result continuation path:

- `generic_server_native_qwen35_tool_result_prefix_cache_preserves_final_answer`

## Honest Boundary

This retained result is about repeated Hermes-equivalent prompt replay and
shared-prefix reuse on `chat.completions`.

It does **not** mean:

- native qwen35 now implements low-level `session_id` reuse
- full Hermes product loops have been benchmarked end to end
- same-turn parallel tool use is solved on the current local qwen35 row

Those remain separate questions. What this receipt does prove is that repeated
direct Hermes tool loops now have a real retained warm path on native Psionic,
with material wallclock improvement and no regression to the retained final
assistant-answer continuation lane.
