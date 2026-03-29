# Hermes Qwen3.5 Parallel Attribution

> Status: `implemented_early` on 2026-03-29 for retained strict same-turn
> parallel-tool attribution across reachable local consumer-GPU backends.

This document records the retained attribution matrix for the still-red Hermes
`parallel_tool_turn` case.

It corrects the earlier overstatement that the remaining red row was simply
"local qwen35 model behavior" on every backend. The retained cross-backend
matrix now shows a stronger result:

- on the same `archlinux` `RTX 4080` host, the same Hermes controller and same
  strict parallel task pass on Ollama `2b` and `4b`
- those same strict rows still fail on native Psionic `2b`, `4b`, and `9b`
- Ollama `9b` also fails, but with a different symptom than Psionic `9b`

So the honest interpretation is:

- there is a real native Psionic lane-specific failure present on `2b` and `4b`
- `9b` is still mixed, because the failure reproduces across both backends but
  not with the same behavior

## Retained Reports

- aggregate:
  `fixtures/qwen35/hermes/hermes_qwen35_parallel_tool_attribution_matrix_20260329_archlinux.json`
- row:
  `fixtures/qwen35/hermes/parallel_matrix_rows/hermes_psionic_parallel_tool_2b_archlinux.json`
- row:
  `fixtures/qwen35/hermes/parallel_matrix_rows/hermes_ollama_parallel_tool_2b_archlinux.json`
- row:
  `fixtures/qwen35/hermes/parallel_matrix_rows/hermes_psionic_parallel_tool_4b_archlinux.json`
- row:
  `fixtures/qwen35/hermes/parallel_matrix_rows/hermes_ollama_parallel_tool_4b_archlinux.json`
- row:
  `fixtures/qwen35/hermes/parallel_matrix_rows/hermes_psionic_parallel_tool_9b_archlinux.json`
- row:
  `fixtures/qwen35/hermes/parallel_matrix_rows/hermes_ollama_parallel_tool_9b_archlinux.json`

## Fixed Contract

Every retained row above keeps the following fixed:

- same host: `archlinux`
- same Hermes revision: `e295a2215acd55f2ee930fc7a4cd2df1c5464234`
- same task:
  emit exactly one assistant tool-call turn containing `get_paris_weather`
  then `get_tokyo_weather`
- same tool schemas and handlers
- `temperature = 0`
- `seed = 0`
- same `required_then_auto` tool policy

What changes between rows:

- backend
- backend-specific model/artifact

## Exact Matrix

| Model | Psionic | Ollama | Honest Verdict |
| --- | --- | --- | --- |
| `2b` | `fail` | `pass` | native Psionic lane-specific failure |
| `4b` | `fail` | `pass` | native Psionic lane-specific failure |
| `9b` | `fail` | `fail` | shared failure, but with different symptoms |

Per-row strict parallel result:

- Psionic `2b`:
  requested both tools, emitted only `get_paris_weather`
- Ollama `2b`:
  emitted `get_paris_weather`, then `get_tokyo_weather` in the same assistant
  tool-call turn
- Psionic `4b`:
  requested both tools, emitted only `get_paris_weather`
- Ollama `4b`:
  emitted `get_paris_weather`, then `get_tokyo_weather` in the same assistant
  tool-call turn
- Psionic `9b`:
  requested both tools, emitted only `get_paris_weather`
- Ollama `9b`:
  requested both tools, but retained no assistant tool-call turn at all on the
  strict parallel case

## Why This Matters

The retained request summaries are the same high-level contract on both
backends:

- `tool_choice = required`
- `parallel_tool_calls = true`
- both declared tools are present in the request:
  `get_paris_weather`, `get_tokyo_weather`

That means the `2b` and `4b` rows are no longer honestly attributable to
"Hermes never asked correctly" or "the local qwen35 family can never do this."

Those rows now show:

- Hermes asked correctly
- Ollama completed the strict same-turn two-tool turn
- native Psionic did not

So `2b` and `4b` are direct evidence of a native Psionic lane gap on this
strict same-turn parallel-tool lane. That is still not the same thing as
proving a pure runtime-only bug, because the Psionic and Ollama rows use
different artifact packaging for the same model family.

## Current Bottom Line

The current strict parallel attribution is:

- overall attribution:
  `native_psionic_lane_specific_failure_present`
- strongest evidence:
  `2b` and `4b` pass on Ollama and fail on Psionic under the same strict task
- remaining nuance:
  `9b` still fails on both backends, so the larger row remains mixed and does
  not reduce cleanly to "Psionic only" or "model only"

That is enough to change the program posture:

- do not keep describing the strict red row as purely local qwen35 model
  behavior
- treat `2b` and `4b` as native Psionic lane-gap evidence rather than a
  universal model-family limit
- treat `9b` as a separate mixed boundary that still needs deeper inspection

## Repo-Owned Runners

The repo now has:

- strict probe filtering in:
  `scripts/release/hermes_qwen35_compatibility_probe.py`
- row runner:
  `scripts/release/run-hermes-qwen35-parallel-attribution-matrix.sh`
- aggregate builder:
  `scripts/release/hermes_qwen35_parallel_matrix_aggregate.py`
