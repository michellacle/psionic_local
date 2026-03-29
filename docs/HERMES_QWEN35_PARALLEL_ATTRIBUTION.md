# Hermes Qwen3.5 Parallel Attribution

> Status: `retained_consumer_gpu_strict_parallel_parity` on 2026-03-29 for the
> strict same-turn Hermes parallel-tool row across reachable local backends.

This document records the retained strict same-turn parallel-tool matrix after
the native Psionic contract fix.

The earlier matrix was useful because it proved the remaining red row was not
honestly reducible to "Hermes asked wrong" or "local qwen35 can never do
this." The latest matrix moves beyond that boundary:

- on the same `archlinux` `RTX 4080` host, Psionic and Ollama now both pass
  the strict same-turn two-tool row on `2b`, `4b`, and `9b`
- the current aggregate attribution is now
  `strict_parallel_tool_turn_solved_on_all_reachable_rows`
- the repo now retains a real native Psionic same-turn parallel receipt rather
  than only a serialized workaround

## Canonical Retained Reports

- aggregate:
  `fixtures/qwen35/hermes/hermes_qwen35_parallel_tool_attribution_matrix_20260329_archlinux.json`
- Psionic `2b` row:
  `fixtures/qwen35/hermes/parallel_matrix_rows/hermes_psionic_parallel_tool_2b_archlinux.json`
- Ollama `2b` row:
  `fixtures/qwen35/hermes/parallel_matrix_rows/hermes_ollama_parallel_tool_2b_archlinux.json`
- Psionic `4b` row:
  `fixtures/qwen35/hermes/parallel_matrix_rows/hermes_psionic_parallel_tool_4b_archlinux.json`
- Ollama `4b` row:
  `fixtures/qwen35/hermes/parallel_matrix_rows/hermes_ollama_parallel_tool_4b_archlinux.json`
- Psionic `9b` row:
  `fixtures/qwen35/hermes/parallel_matrix_rows/hermes_psionic_parallel_tool_9b_archlinux.json`
- Ollama `9b` row:
  `fixtures/qwen35/hermes/parallel_matrix_rows/hermes_ollama_parallel_tool_9b_archlinux.json`

## Fixed Contract

Every retained row above keeps the following fixed:

- same host: `archlinux`
- same Hermes revision: `e295a2215acd55f2ee930fc7a4cd2df1c5464234`
- same strict task:
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
| `2b` | `pass` | `pass` | passes on both backends |
| `4b` | `pass` | `pass` | passes on both backends |
| `9b` | `pass` | `pass` | passes on both backends |

Per-row retained assistant tool order is now the same everywhere:

- `get_paris_weather`
- `get_tokyo_weather`

## What Actually Fixed The Row

The winning change was not a benchmark wrapper trick. It was a tighter native
Psionic same-turn tool contract:

1. required tool batches now prefer a plain JSON-schema batch at the output
   boundary
2. the runtime now accepts both the plain JSON batch and the older tagged
   structure during replay
3. required parallel turns now carry a concrete
   `minimum_required_tool_calls` floor
4. that floor is inferred from backticked declared tool-name mentions in the
   request messages when `tool_choice = required` and
   `parallel_tool_calls = true`

That last part mattered. The older contract let the model legally stop after
one tool call even when the strict Hermes task clearly named both tools in the
same turn.

## Why The Earlier Matrix Still Helped

The older failing matrix remains useful historical evidence because it narrowed
the search space:

- Hermes was already sending both declared tools
- Ollama could already complete the strict two-tool row on the same host
- so the remaining gap was small enough to keep attacking locally on one
  consumer GPU instead of escalating immediately to bigger hardware

That local diagnosis turned out to be correct. The current retained matrix is
the closeout proof.

## Honest Bottom Line

The strict same-turn Hermes parallel-tool row is now green on the reachable
consumer-GPU lanes:

- native Psionic `2b`
- native Psionic `4b`
- native Psionic `9b`
- Ollama `2b`
- Ollama `4b`
- Ollama `9b`

This closes the old local parallel parity blocker. Remaining Hermes work is
now about benchmark behavior, comparator availability, and broader product
readiness, not this same-turn tool row.
