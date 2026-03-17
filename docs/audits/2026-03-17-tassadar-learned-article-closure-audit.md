# Tassadar Learned Article-Closure Audit

Date: March 17, 2026

## Result

No exact article-class learned executor exists today.

The learned lane remains bounded.

This audit is subordinate to the machine-readable acceptance report and does not
override it.

## Canonical Decision Artifacts

- `fixtures/tassadar/reports/tassadar_acceptance_report.json`
- `fixtures/tassadar/reports/tassadar_learned_horizon_policy_report.json`
- `fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0/sequence_fit_report.json`
- `fixtures/tassadar/runs/hungarian_v0_learned_executor_v0/learned_lane_report.json`
- `fixtures/tassadar/runs/hungarian_v0_learned_executor_v0/sequence_fit_report.json`
- `fixtures/tassadar/runs/sudoku_v0_promotion_v3/promotion_bundle.json`
- `fixtures/tassadar/runs/sudoku_v0_promotion_v3/promotion_gate_report.json`

## Acceptance Alignment

The machine-readable acceptance verdict is:

- `learned_article_class.passed = false`
- `article_closure.passed = false`
- `current_truth_holds = true`

That means the repo is behaving honestly by keeping the learned article-class
claim red.

## Decisive Facts

The green learned artifact that exists today is only the bounded 4x4 promotion
lane:

- `fixtures/tassadar/runs/sudoku_v0_promotion_v3/promotion_gate_report.json`
  records:
  - `first_target_exactness_bps = 10000`
  - `first_32_token_exactness_bps = 10000`
  - `exact_trace_case_count = 2`

That is enough for `learned_bounded`. It is not enough for
`learned_article_class`.

The current learned 9x9 lane remains bounded:

- `fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0/sequence_fit_report.json`
  records:
  - `full_sequence_fits_model_context = false`
  - `fit_disposition = bounded_scope_replacement`
  - full-sequence overflow up to `4811021` tokens

The current learned Hungarian lane remains research-only:

- `fixtures/tassadar/runs/hungarian_v0_learned_executor_v0/learned_lane_report.json`
  records:
  - `aggregate_target_token_exactness_bps = 6839`
  - `first_target_exactness_bps = 0`
  - `first_32_token_exactness_bps = 6875`
  - `exact_trace_case_count = 0`
  - `final_output_exact_case_count = 0`

The long-horizon learned bar is now explicitly refused rather than implied:

- `fixtures/tassadar/reports/tassadar_learned_horizon_policy_report.json`
  records:
  - `guard_status = explicit_refusal_policy`
  - `benchmark_status = not_landed`
  - `refusal_kind = unsupported_horizon`
  - `learned_article_class_bypass_allowed = false`
  - `article_class_trace_step_floor = 1048575`

## What Would Turn This Green

This audit only turns green if the acceptance artifacts turn green first.

In practice that requires:

- one exact learned long-horizon benchmark bundle that replaces the current
  refusal policy
- one learned article-class workload that is exact rather than bounded-window
  or research-only
- one acceptance report with `learned_article_class.passed = true`

Until those artifacts exist, any prose stronger than "the learned lane remains
bounded" is wrong.

## Conclusion

The repo has real learned progress:

- bounded 4x4 learned promotion is green
- learned Hungarian-v0 now exists with explicit state supervision
- the learned horizon limit is now explicit and machine-readable

But the article-class learned claim is still red.

The correct current statement is:

- the learned lane remains bounded

Not:

- exact article-class learned executor exists
