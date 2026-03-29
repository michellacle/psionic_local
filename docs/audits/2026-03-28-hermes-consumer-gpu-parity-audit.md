# Hermes Consumer-GPU Parity Audit

> Status: `implemented_early` on 2026-03-28 for direct Hermes parity on one
> consumer GPU.

## Summary

The local Hermes consumer-GPU parity blocker is now closed on pushed
`f4788f38cc04febf5d9e9eb526694de048ceabc2`.

The repo now retains:

- one full native Psionic Hermes compatibility receipt that passes `6/6` on
  the normal custom-provider `chat.completions` path:
  `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260329_archlinux_2b_f4788f38.json`
- one strict same-turn parallel attribution matrix showing Psionic and Ollama
  both pass the same row on `2b`, `4b`, and `9b` on the same `archlinux`
  `RTX 4080` host:
  `fixtures/qwen35/hermes/hermes_qwen35_parallel_tool_attribution_matrix_20260329_archlinux.json`

## What Was Actually Wrong

The red same-turn parallel row was not resolved by changing hardware, adding a
special-case benchmark exemption, or reinterpreting Hermes behavior.

The missed runtime seams were:

1. the required-tool batch contract still leaned on the older tagged output
   shape instead of preferring the plain JSON-schema tool-call batch
2. the runtime had no concrete lower bound on how many tool calls had to
   appear in a required same-turn parallel batch

That second omission mattered because the strict Hermes prompt literally named
both declared tools in backticks. The old contract still allowed the model to
stop after a single tool call.

## What Changed

In `crates/psionic-serve/src/openai_http.rs`:

- required/named tool turns now prefer a plain JSON-schema batch
- replay accepts both plain JSON and the older tagged structure
- `ToolCallingContract` now carries `minimum_required_tool_calls`
- that minimum is inferred from backticked declared tool-name mentions in the
  request messages when `tool_choice = required` and
  `parallel_tool_calls = true`
- the prompt contract now states that the model must emit at least that many
  tools on the required parallel batch

## Retained Results

### Full compatibility proof

Native Psionic `2b` now passes all six retained Hermes acceptance cases:

- `required_tool_turn`
- `auto_plain_text_turn`
- `multi_turn_tool_loop`
- `parallel_tool_turn`
- `invalid_argument_truthful_refusal`
- `streamed_tool_turn`

Receipt:

- `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260329_archlinux_2b_f4788f38.json`

### Strict parallel parity

The separate strict same-turn matrix now closes on every reachable local row:

- Psionic `2b`: pass
- Ollama `2b`: pass
- Psionic `4b`: pass
- Ollama `4b`: pass
- Psionic `9b`: pass
- Ollama `9b`: pass

Aggregate receipt:

- `fixtures/qwen35/hermes/hermes_qwen35_parallel_tool_attribution_matrix_20260329_archlinux.json`

Overall attribution:

- `strict_parallel_tool_turn_solved_on_all_reachable_rows`

## Overlooked Opportunities That Turned Out Real

The earlier "consumer-GPU opportunities may be exhausted" posture was too
strong. The practical local opportunities that were still available were:

- tighten the required same-turn output contract instead of assuming the model
  family had reached a hard limit
- use the same-host Ollama pass rows as a diagnostic bound rather than as a
  reason to give up on native Psionic
- prove one honest full compatibility receipt after the row-level fix instead
  of stopping at split evidence

Those were the right next steps, and they were sufficient to clear the local
parity blocker without escalating to bigger GPUs.

## What Is Still Not Solved

This parity closeout does not mean every Hermes question is finished.

Real remaining gaps:

- Ollama still wins wallclock on the older easy same-host benchmark rows
- `llama.cpp` is still not a real apples-to-apples runtime comparator for this
  `qwen35` artifact contract on the current host
- raw-versus-registry artifact isolation remains permission-blocked because the
  Ollama-managed blob path is not readable to the current user

Those are benchmark and attribution follow-ons. They are not direct Hermes
compatibility blockers anymore.

## Honest Bottom Line

The repo now retains direct Hermes-on-Psionic parity on one consumer GPU.

The next honest program phase is no longer "keep proving Hermes can work at
all." It is:

- improve benchmark behavior
- improve comparator coverage
- decide whether broader product/demo readiness needs anything beyond the now
  green local compatibility lane
