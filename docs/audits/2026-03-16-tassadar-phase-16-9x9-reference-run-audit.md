# Tassadar Phase 16 9x9 Reference Run Audit

Date: 2026-03-16

Issue: `#8` "Tassadar Phase 16: persist and review first honest 9x9 run"

## Outcome

9x9 only partially fit and remains blocked.

That is the entire point of this phase landing honestly: the repo now has a
real artifact-backed 9x9 learned-lane bundle, and the artifacts say exactly
where the lane stops.

## What Landed

The canonical bundle now lives at:

- `fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0`

The canonical replay path is:

- `crates/psionic-train/examples/tassadar_sudoku_9x9_reference_run.rs`

The run now persists:

- `training_manifest.json`
- `training_report.json`
- `boundary_exactness_report.json`
- `trace_divergence_report.json`
- `sequence_fit_report.json`
- `postmortem.json`
- `next_run_plan.json`
- checkpoint and model artifacts needed to replay the selected bundle state

## What The Fit Report Says

`sequence_fit_report.json` makes the fit boundary machine-readable:

- model max sequence tokens: `524288`
- prompt length: `51536`
- full target length: `4839686` to `5283773`
- full total sequence length: `4891222` to `5335309`
- full-sequence overflow over current model context: `4366934` to `4811021`

That means a truthful full-sequence learned 9x9 run is not possible with the
current model/context contract.

The landed run therefore uses an explicit bounded long-trace posture:

- teacher-forced training strategy: `incremental_decode_window`
- curriculum stages: `1`, `8`, `32`, `128`, then `512` target tokens
- eval cap: first `512` target tokens

The fit report also records why this strategy is the honest one right now:

- estimated naive full forward buffer pressure on the longest full sequence:
  `11246829264` bytes
- estimated bounded forward buffer pressure at the landed `512`-token window:
  `109715076` bytes
- estimated incremental decode live-state pressure at that same bounded window:
  `1459452` bytes

So this issue does not end with "9x9 works."

It ends with "the repo now records the real fit cliff and the bounded strategy
it used instead of hand-waving past it."

## What The Run Achieved

The selected checkpoint is:

- `tassadar-executor-transformer-sudoku-9x9-reference-run-v0.checkpoint.epoch_0004`

Bounded validation metrics over the first `512` target tokens:

- first-target exactness: `10000` bps
- first-8 exactness: `7500` bps
- first-32 exactness: `5938` bps
- exact validation traces: `0/1`
- bounded validation aggregate target exactness: `8125` bps

The divergence artifact keeps the remaining failure explicit:

- all train/validation/test cases still first diverge at target token `1`
- the reference token there is `<step_index>`
- the learned prediction there is `<byte_00>`

So the learned lane is no longer failing at token `0`, but it is also nowhere
near a truthful exact 9x9 learned executor claim.

## Why This Closes Phase 16

Phase 16 was not "make 9x9 succeed."

It was:

1. persist the first honest 9x9 run
2. make the long-trace strategy explicit
3. make fit and failure evidence machine-readable
4. keep the audit honest about whether 9x9 worked

The repo now satisfies that bar.

It does not satisfy a stronger bar such as:

- full-trace 9x9 fit
- exact 9x9 learned traces
- exact 9x9 final outputs
- article parity

## Next Honest Move

`next_run_plan.json` is intentionally narrow:

- add later-window truth instead of over-reading the first bounded window
- either increase context or adopt explicitly wider long-trace windowing
- keep the lane in the "partial" state until those blockers are removed
