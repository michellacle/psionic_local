# Psion Run Observability

> Status: canonical `PSION-13` / `#369` run-observability contract, written
> 2026-03-22 after landing the explicit Psion pretrain stage.

This document freezes the first structured observability surface for `Psion`
pretraining runs.

It keeps cost, throughput, checkpoint size, hardware topology, and instability
markers in typed receipts instead of ad hoc operator notes.

## Canonical Artifacts

- `crates/psionic-train/src/psion_run_observability.rs` owns the typed
  pretrain-run observability receipts and the stage-level summary artifact.
- `crates/psionic-train/examples/psion_run_observability_fixtures.rs`
  regenerates the canonical observability fixtures.
- `fixtures/psion/observability/psion_pilot_pretrain_run_observability_receipt_v1.json`
  is the canonical pilot-run observability receipt.
- `fixtures/psion/observability/psion_broader_pretrain_run_observability_receipt_v1.json`
  is the canonical broader-pretraining observability receipt.
- `fixtures/psion/observability/psion_pretrain_stage_observability_summary_v1.json`
  is the canonical stage-level summary that compares the pilot and broader
  pretraining receipts directly.

The stable schema versions are:

- `psion.pretrain_run_observability_receipt.v1`
- `psion.pretrain_stage_observability_summary.v1`

## What The Receipt Freezes

The first observability receipt now binds:

- one explicit pretrain-stage receipt digest from `PSION-12`
- run-level cost breakdown in micro-USD
- token-rate, sequence-rate, step-latency, and checkpoint-write throughput
- promoted-checkpoint artifact size reporting
- realized delivered-execution and hardware-topology facts
- aggregated instability telemetry and structured policy markers when present

That gives later pilot-review and broader-pretraining decisions a repo-owned
artifact surface instead of screenshots or remembered logs.

## Mechanical Enforcement

`psionic-train` now validates that:

- every observability receipt still points to one exact pretrain-stage receipt
- cost components sum exactly to the declared total
- throughput fields stay non-zero and internally coherent
- checkpoint artifact identity still matches the promoted checkpoint from the
  stage receipt
- multi-device runs surface an explicit topology instead of hiding device shape
- instability markers map back to surfaced telemetry when a stability verdict
  exists
- the stage-level summary contains both the `pilot` and
  `broader_pretraining` rows and each row still matches the underlying run
  receipt
