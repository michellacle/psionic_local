# Tassadar Phase 14 Promotion Green Audit

Date: March 16, 2026

## Result

The 4x4 learned lane is now promotable.

The canonical green bundle is:

- `fixtures/tassadar/runs/sudoku_v0_promotion_v3`

The canonical replay command is:

- `cargo run -p psionic-research --example tassadar_executor_attention_promotion_run`

The canonical gate check is:

- `scripts/check-tassadar-4x4-promotion-gate.sh fixtures/tassadar/runs/sudoku_v0_promotion_v3`

## Decisive Artifacts

- `fixtures/tassadar/runs/sudoku_v0_promotion_v3/best_checkpoint_manifest.json`
- `fixtures/tassadar/runs/sudoku_v0_promotion_v3/exactness_curve.json`
- `fixtures/tassadar/runs/sudoku_v0_promotion_v3/failure_samples.json`
- `fixtures/tassadar/runs/sudoku_v0_promotion_v3/exact_trace_samples.json`
- `fixtures/tassadar/runs/sudoku_v0_promotion_v3/promotion_gate_report.json`

## Selected Checkpoint

- checkpoint id:
  - `tassadar-executor-attention-sudoku-v0-promotion-v3.checkpoint.epoch_0015`
- stage:
  - `prompt_to_first_32_tokens`
- gate metrics:
  - `first_target_exactness_bps = 10000`
  - `first_32_token_exactness_bps = 10000`
  - `exact_trace_case_count = 2`

## What Changed

The green bundle is not the old lookup family with more schedule churn.

It is a learned attention-family continuation that:

- bootstraps from the preserved red attention boundary checkpoint at
  `fixtures/tassadar/runs/sudoku_v0_attention_boundary_v9`
- persists that bootstrap stage under
  `fixtures/tassadar/runs/sudoku_v0_promotion_v3/bootstrap_pc_boundary`
- finishes the early learned trace with a bounded
  `relative_target_output_bias + trace_schema_output_bias` continuation over
  the first `32` target tokens

## Validation Evidence

- `family_report.json` records both validation cases as exact:
  - `sudoku_v0_validation_a`
  - `sudoku_v0_validation_b`
- `exact_trace_samples.json` captures both exact validation traces
- `failure_samples.json` is empty because there are no remaining validation
  misses inside the bounded promotion window
- the standalone gate checker revalidates the stored report digest and pass
  fields as internally consistent

## Conclusion

Phase 14 is green.

Phase 16 is no longer blocked on the learned 4x4 promotion gate.
