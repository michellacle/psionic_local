# `TAS-171` Transformer Direct-Proof Closure Audit

## Scope

Move the existing bounded direct no-tool proof family off the historical
fixture model and onto the committed trained trace-bound Transformer-backed
reference-linear route without widening the proof family beyond the three
named canonical workloads.

## What Changed

- `psionic-runtime` now requires an explicit model-lineage contract ref and
  digest on every direct model-weight proof receipt
- `psionic-router` now exposes one dedicated reference-linear direct-proof
  route rebinding helper for the Transformer-backed model id
- `psionic-serve` now commits the rebased direct-proof report at
  `fixtures/tassadar/reports/tassadar_direct_model_weight_execution_proof_report.json`
- `psionic-provider` now mirrors the stronger report shape in its
  provider-facing direct-proof receipt

## Frozen Facts

- the canonical direct-proof family still covers exactly 3 workloads:
  `long_loop_kernel`, `sudoku_v0_test_a`, and `hungarian_matching`
- each receipt now binds the trained trace-bound Transformer model id,
  descriptor digest, weight-bundle digest, and lineage-contract digest
- the proof route stays direct, fallback-free, and zero-external-call on every
  committed case
- the Transformer rebinding is justified only through the committed
  fixture-to-Transformer parity certificate and does not skip that dependency
- the historical fixture model remains visible only as baseline provenance, not
  as the proving model for the committed report

## Verdict

The repo now has one machine-readable bounded direct-proof report whose proving
model is the committed trained trace-bound Transformer route rather than the
historical fixture lane.

It does not yet close full declared-workload reference-linear exactness,
anti-memorization, contamination independence, fast-route closure, benchmark
parity, single-run closure, or the final clean-room interpreter-in-weights
verdict.
