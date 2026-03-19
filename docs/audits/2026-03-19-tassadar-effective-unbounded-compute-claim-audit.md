# Tassadar Effective-Unbounded Compute Claim Audit

## Scope

This audit evaluates whether the public `psionic` repo may honestly claim
`effective unbounded computation under envelopes`.

## Current verdict

Current verdict: `suppressed`.

The repo does have bounded execution slices plus resumable continuation and
state objects under explicit envelopes. It does **not** yet have the
combination of broad publication, portability, and specialization-safety facts
required for a broader public claim.

## What is true now

- resumable continuation objects are real:
  `tassadar_process_object_report.json`,
  `tassadar_spill_tape_store_report.json`,
  `tassadar_installed_process_lifecycle_report.json`
- effect receipts and typed refusal boundaries are real:
  `tassadar_effectful_replay_audit_report.json`,
  `tassadar_hybrid_process_controller_report.json`
- hybrid process control exists in bounded form with verifier-positive-delta
  cases and explicit unsupported-transition refusal:
  `tassadar_hybrid_process_controller_report.json`

## What still blocks the claim

- `portable_publication_envelope`
  Only the narrow article-closeout portability anchor is green enough for
  promotion. That is not sufficient for an `effective unbounded` public claim.
- `broad_publication_gate`
  The broad internal-compute gate is not green. Suppression remains explicit.
- `specialization_safety_gate`
  Broad-family specialization is not fully safe to promote because some
  families are benchmark-only and at least one current family is explicitly
  non-decompilable.

## What the allowed statement means

The strongest honest statement today is:

Psionic/Tassadar has bounded execution slices plus resumable continuation and
state objects under explicit envelopes.

That statement does **not** imply:

- arbitrary Wasm execution
- broad served internal compute
- arbitrary effectful process execution
- Turing-complete support

## Source artifacts

- `fixtures/tassadar/reports/tassadar_effective_unbounded_compute_claim_report.json`
- `fixtures/tassadar/reports/tassadar_effective_unbounded_compute_claim_summary.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_acceptance_gate.json`
- `fixtures/tassadar/reports/tassadar_broad_family_specialization_report.json`
- `fixtures/tassadar/reports/tassadar_hybrid_process_controller_report.json`
