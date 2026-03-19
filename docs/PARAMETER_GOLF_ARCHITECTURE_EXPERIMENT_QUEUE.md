# Psionic Parameter Golf Architecture Experiment Queue

> Status: canonical `PGOLF-610` / `#254` concrete post-parity architecture
> queue, with `PGOLF-611` / `#255` and `PGOLF-612` / `#256` implemented as
> bounded research-only proxy rows, updated 2026-03-19 after landing
> `crates/psionic-research/src/parameter_golf_architecture_experiment_queue.rs`.

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
- `PGOLF-613` / `#257`: still-open restricted-attention candidate with a
  benchmark plan and explicit missing-evidence note

All of them stay bound to the same bounded local-reference oracle, metric, and
counted non-record code surface already frozen by the earlier research harness.

## Implemented Rows

The two implemented rows are intentionally narrow:

- the shared-depth row reuses one canonical decoder block by value across the
  decoder half
- the stronger parameter-tying row ties mirrored encoder-decoder block pairs by
  value

Both rows are measured on the same bounded local-reference validation path and
on the same int8-plus-zlib export surface used by the non-record package. That
means the report now preserves:

- `val_loss`
- `val_bpb`
- compressed-model bytes
- total counted bytes under unchanged code bytes
- explicit runtime facts for the unchanged dense runtime path

## Current Honest Boundary

This queue is still research-only.

The implemented rows are post-train value-sharing probes over the frozen local
baseline control. They do not yet claim:

- full retraining closure
- single-H100 trainer closure
- `8xH100` readiness
- record-track promotion

The locality or restricted-attention issue also remains open intentionally.
The queue now records the benchmark plan for that row, but Psionic does not yet
have a public-safe restricted-attention eval path on real `seq_len=1024`
challenge-format windows. The repo keeps that gap explicit instead of closing
the issue on toy-window evidence.
