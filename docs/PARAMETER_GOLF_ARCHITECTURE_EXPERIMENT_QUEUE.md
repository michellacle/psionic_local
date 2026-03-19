# Psionic Parameter Golf Architecture Experiment Queue

> Status: canonical `PGOLF-610` / `#254` concrete post-parity architecture
> queue, with `PGOLF-611` / `#255`, `PGOLF-612` / `#256`, and
> `PGOLF-613` / `#257` now implemented as bounded research-only rows, updated
> 2026-03-19 after landing
> `crates/psionic-research/src/parameter_golf_architecture_experiment_queue.rs`
> and the dedicated restricted-attention report.

This document records the first concrete post-parity architecture queue for
Parameter Golf.

It exists to separate research exploration from baseline or record-track claim
posture while still preserving one concrete, machine-readable queue instead of
one generic "variant search" bucket.

## What Landed

`psionic-research` now exposes:

- `ParameterGolfArchitectureExperimentQueueReport`
- `build_parameter_golf_architecture_experiment_queue_report()`
- `write_parameter_golf_architecture_experiment_queue_report(...)`

The committed machine-readable report now lives at:

- `fixtures/parameter_golf/reports/parameter_golf_architecture_experiment_queue_report.json`

## Concrete Queue

The first queue now has three explicit rows:

- `PGOLF-611` / `#255`: implemented shared-depth decoder value-tying proxy
- `PGOLF-612` / `#256`: implemented mirrored block-pair tying proxy
- `PGOLF-613` / `#257`: implemented restricted-attention candidate backed by a
  dedicated `seq_len=1024` challenge-format slice report

The first two rows stay bound to the bounded local-reference harness. The
restricted-attention row keeps the same frozen model and counted code surface,
but it now points at the dedicated challenge-format slice evidence in
`fixtures/parameter_golf/reports/parameter_golf_restricted_attention_report.json`.

## Implemented Rows

The three implemented rows are intentionally narrow:

- the shared-depth row reuses one canonical decoder block by value across the
  decoder half
- the stronger parameter-tying row ties mirrored encoder-decoder block pairs by
  value
- the restricted-attention row limits every target token to the most recent
  `256` source positions on one committed `seq_len=1024` challenge-format
  validation slice

The first two rows are measured on the bounded local-reference validation path.
The restricted-attention row is measured in the dedicated report at
`docs/PARAMETER_GOLF_RESTRICTED_ATTENTION_REPORT.md`.

Across those reports, the queue now preserves:

- `val_loss`
- `val_bpb`
- compressed-model bytes
- total counted bytes under unchanged code bytes
- explicit runtime facts for the unchanged dense runtime path or the explicit
  restricted-attention proxy path

## Current Honest Boundary

This queue is still research-only.

The implemented rows do not yet claim:

- full retraining closure
- single-H100 trainer closure
- `8xH100` readiness
- record-track promotion

The restricted-attention result is currently negative evidence, not a promoted
new baseline: the committed `256`-token window cuts analytic attention-score
terms to about `43.7%` of dense, but it worsens `val_bpb` by about `0.05037`
on the committed challenge-format slice. That is still useful closure because
future locality work now has a concrete evidence floor instead of a hand-wavy
intuition bucket.
