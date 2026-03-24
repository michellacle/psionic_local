# Psionic Parameter Golf Device-Resident Validation Runner Audit

> Status: written 2026-03-24 after landing the device-resident validation
> runner in
> `crates/psionic-train/src/parameter_golf_single_h100_training.rs` and
> recording
> `fixtures/parameter_golf/reports/parameter_golf_validation_runtime_comparison.json`.

## Summary

The single-H100 validation path no longer rebuilds the full host parameter map
and reuploads the stable model surface for every validation batch.

The validation runner now:

- allocates one resident CUDA buffer per stable Parameter Golf graph input
  parameter for the admitted batch shape
- reuses one mutable input-token buffer and one mutable target-token buffer
  across all batches in the pass
- precomputes per-batch byte-accounting once before the loop
- records a machine-readable runtime receipt inside each validation summary

This does not solve the main validation-speed blocker. The dominant forward
cost is still the decode-style attention path tracked separately in `#482`.

It does remove the structural host churn that was still making validation
materially different from the upstream model-resident eval posture.

## What Changed

The new validation path lives in:

- `evaluate_validation_on_cuda(...)`
- `ParameterGolfCudaValidationSession`
- `ParameterGolfSingleH100ValidationRuntimeReceipt`

The runner now creates one session per admitted validation batch shape and
keeps these objects resident for the duration of the pass:

- the lowered training graph for that batch size
- the stable parameter-input CUDA buffers
- reusable dense `i32` token-id and target-id buffers
- reusable host staging vectors for those mutable token buffers

The old host-rebind path still exists as
`evaluate_validation_on_cuda_legacy(...)` only so the repo can produce one
same-node comparison receipt against the old behavior.

## Same-Node Evidence

The committed receipt at
`fixtures/parameter_golf/reports/parameter_golf_validation_runtime_comparison.json`
was generated on the current local RTX 4080 node with:

- `batch_sequences=8`
- `sequence_length=1024`
- `batch_limit=2`

Observed result:

- legacy average batch time: `7574.50 ms`
- device-resident average batch time: `7392.00 ms`

That is a bounded same-node win of about `2.4%` while keeping model outputs
numerically aligned for the measured slice.

The resident-path runtime receipt in that same file also records:

- `persistent_parameter_buffer_count=92`
- `persistent_parameter_value_count=17059912`
- `per_batch_stable_parameter_buffer_allocations=0`
- `resident_parameter_upload_us=59863`
- `total_input_token_write_us=61`
- `total_target_token_write_us=52`
- `total_byte_accounting_us=23`

That is the important substrate proof for this issue: the stable parameter
surface is now uploaded once per validation session instead of once per batch.

## Honest Boundary

This issue does not close the main validation-speed problem.

The bounded local receipt improved because host churn was removed. Validation is
still far too slow overall because the admitted forward attention path is still
a token-by-token decode-style loop.

That remaining blocker is tracked by `#482`.
