# Psionic Parameter Golf Eval Graph Split Audit

> Status: written 2026-03-24 after landing the eval-specific Parameter Golf
> graph surface in `crates/psionic-train/src/parameter_golf_baseline_graph.rs`
> and routing the device-resident validation runner in
> `crates/psionic-train/src/parameter_golf_single_h100_training.rs` through
> that surface.

## Summary

Validation no longer routes through the Parameter Golf training-graph surface
by default.

The device-resident validation runner now consumes an explicit
`parameter_golf_baseline_eval_graph_v1` surface that:

- keeps persistent resident weights
- keeps reusable mutable token-id and target-id buffers
- emits only the scalar loss needed for validation reporting
- records the eval graph surface explicitly in the runtime receipt

This is the structural split the validation-runtime audit called for.

## What Changed

The repo now owns `ParameterGolfBaselineEvalGraph` alongside the existing
baseline graph and training graph.

That eval graph:

- is built with `AutodiffContext::evaluation()`
- binds the same stable parameter surface
- accepts integer `input_token_ids` and `target_ids`
- emits one scalar projection loss
- avoids the training-graph surface that carried backward-oriented structure
  and assumptions

`evaluate_validation_on_cuda(...)` now caches and reuses that eval graph
instead of caching `ParameterGolfBaselineTrainingGraph`.

The device-resident validation receipt now records:

- `path=device_resident_cuda_eval_graph_v1`
- `graph_surface=parameter_golf_baseline_eval_graph_v1`

## Local Evidence

The refreshed canonical comparison receipt in
`fixtures/parameter_golf/reports/parameter_golf_validation_runtime_comparison.json`
was generated on the local RTX 4080 node after the eval split.

It recorded:

- legacy average batch time: `3431.00 ms`
- device-resident average batch time: `3241.50 ms`
- runtime receipt path: `device_resident_cuda_eval_graph_v1`
- runtime receipt graph surface: `parameter_golf_baseline_eval_graph_v1`

The eval split is not a massive speed event by itself. That is expected.

The main value here is structural:

- validation has its own explicit graph surface
- receipts can now name that surface directly
- later single-H100 and distributed validation work can target eval-specific
  runtime code instead of inheriting training-graph plumbing

## What This Proves

- validation no longer uses the training graph by default
- the resident validation runner keeps the exact resident-weight and
  mutable-token-buffer posture that `#481` introduced
- the validation receipt can cite one explicit eval graph surface

## What Remains

This does not close the remaining wallclock gap by itself.

The next speed-critical items are still:

- fresh H100 receipts for the new forward runtime
- the remaining backward replay work on the train path
- later distributed validation sharding and aggregation on `8xH100`
