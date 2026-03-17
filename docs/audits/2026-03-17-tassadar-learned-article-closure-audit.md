# Tassadar Learned Article-Closure Audit

Date: March 17, 2026

## Result

An exact article-class learned executor now exists for the committed
Hungarian-10x10 benchmark corpus.

This audit is subordinate to the machine-readable acceptance report and does not
override it.

## Canonical Decision Artifacts

- `fixtures/tassadar/reports/tassadar_acceptance_report.json`
- `fixtures/tassadar/reports/tassadar_learned_horizon_policy_report.json`
- `fixtures/tassadar/runs/hungarian_10x10_v0_learned_article_executor_v0/run_bundle.json`
- `fixtures/tassadar/runs/hungarian_10x10_v0_learned_article_executor_v0/sequence_fit_report.json`
- `fixtures/tassadar/runs/hungarian_10x10_v0_learned_article_executor_v0/article_learned_benchmark_report.json`
- `fixtures/tassadar/runs/hungarian_10x10_v0_learned_article_executor_v0/model_artifact.json`
- `fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0/sequence_fit_report.json`
- `fixtures/tassadar/runs/hungarian_v0_learned_executor_v0/learned_lane_report.json`
- `fixtures/tassadar/runs/sudoku_v0_promotion_v3/promotion_bundle.json`

## Acceptance Alignment

The machine-readable acceptance verdict is:

- `learned_article_class.passed = true`
- `article_closure.passed = true`
- `current_truth_holds = true`

That means the repo can now surface the learned article-class claim honestly at
the committed benchmark-corpus scope.

## Decisive Facts

The committed learned article benchmark is exact.

- `fixtures/tassadar/runs/hungarian_10x10_v0_learned_article_executor_v0/article_learned_benchmark_report.json`
  records:
  - `validation.aggregate_target_token_exactness_bps = 10000`
  - `test.aggregate_target_token_exactness_bps = 10000`
  - `validation.exact_trace_case_count = 1`
  - `test.exact_trace_case_count = 1`
  - `passed = true`

The full sequence fits the current learned article model context.

- `fixtures/tassadar/runs/hungarian_10x10_v0_learned_article_executor_v0/sequence_fit_report.json`
  records:
  - `full_sequence_fits_model_context = true`
  - `target_token_count_max = 169`
  - `total_token_count_max = 22050`

The exactness surface is explicit in the committed model artifact.

- `fixtures/tassadar/runs/hungarian_10x10_v0_learned_article_executor_v0/model_artifact.json`
  records:
  - `prompt_summary_bucket_count = 4096`
  - `prompt_summary_embeddings`
  - `prompt_summary_target_output_bias_row_keys`
  - `prompt_summary_target_output_bias_row_values`
  - `relative_target_trace_schema_output_bias`

The learned-horizon policy is now green on the landed benchmark.

- `fixtures/tassadar/reports/tassadar_learned_horizon_policy_report.json`
  records:
  - `guard_status = exact_benchmark_landed`
  - `benchmark_status = exact`
  - `benchmark_artifact_ref = fixtures/tassadar/runs/hungarian_10x10_v0_learned_article_executor_v0/article_learned_benchmark_report.json`
  - `learned_article_class_bypass_allowed = false`

## Boundary Note

This green learned article closure does not erase the other learned limits still
recorded elsewhere.

- `fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0/sequence_fit_report.json`
  still keeps the learned 9x9 lane bounded to `incremental_decode_window`.
- `fixtures/tassadar/runs/hungarian_v0_learned_executor_v0/learned_lane_report.json`
  still keeps the older Hungarian-v0 learned lane research-only.

Those facts remain true. They are no longer the blocker for
`learned_article_class` because the exact Hungarian-10x10 article benchmark now
exists and is committed.
