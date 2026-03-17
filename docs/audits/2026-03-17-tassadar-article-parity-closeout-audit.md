# Tassadar Article-Parity Closeout Audit

Date: March 17, 2026

## Result

Final article-parity closure is still red.

This audit is subordinate to the machine-readable acceptance report and does not
override it.

## Canonical Decision Artifacts

- `fixtures/tassadar/reports/tassadar_acceptance_report.json`
- `scripts/check-tassadar-acceptance.sh`
- `fixtures/tassadar/reports/tassadar_compiled_article_closure_report.json`
- `fixtures/tassadar/runs/sudoku_9x9_v0_compiled_executor_v0`
- `fixtures/tassadar/runs/hungarian_10x10_v0_compiled_executor_v0`
- `fixtures/tassadar/runs/compiled_kernel_suite_v0`
- `fixtures/tassadar/runs/sudoku_v0_reference_run_v0/neural_hull_benchmark_report.json`
- `fixtures/tassadar/reports/tassadar_article_executor_session_artifact.json`
- `fixtures/tassadar/reports/tassadar_article_hybrid_workflow_artifact.json`
- `fixtures/tassadar/reports/tassadar_learned_horizon_policy_report.json`
- `fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0/sequence_fit_report.json`
- `fixtures/tassadar/runs/hungarian_v0_learned_executor_v0/learned_lane_report.json`

## Acceptance Alignment

The machine-readable acceptance verdict is:

- `article_parity_language_allowed = false`
- `compiled_article_class.passed = true`
- `fast_path_declared_workload_exact.passed = true`
- `learned_article_class.passed = false`
- `article_closure.passed = false`
- `current_truth_holds = true`

That means the repo is behaving honestly by keeping article-parity wording red.

## Reproduction Commands

These are the repo-owned commands that the closeout audit accepts as canonical:

- `scripts/check-tassadar-acceptance.sh`
- `scripts/check-tassadar-compiled-article-closure.sh`
- `cargo test -p psionic-eval neural_hull_benchmark_reports_direct_hull_selection_and_window_cap`
- `cargo run -p psionic-serve --example tassadar_article_executor_session_artifact`
- `cargo run -p psionic-serve --example tassadar_article_hybrid_workflow_artifact`

If those commands do not reproduce the referenced artifacts and verdicts from a
local checkout, this audit is not green.

## Decisive Facts

The compiled article-class bar is now genuinely green.

- `fixtures/tassadar/reports/tassadar_compiled_article_closure_report.json`
  records compiled article closure as green.
- `fixtures/tassadar/runs/sudoku_9x9_v0_compiled_executor_v0` is the exact
  compiled 9x9 Sudoku article-sized root.
- `fixtures/tassadar/runs/hungarian_10x10_v0_compiled_executor_v0` is the
  exact compiled 10x10 Hungarian article-sized root.
- `fixtures/tassadar/runs/compiled_kernel_suite_v0` widens compiled
  article-shaped evidence beyond Sudoku and Hungarian to arithmetic, memory,
  branch, and loop-heavy kernels.

The fast decode story is green only on its declared bounded workload class.

- `fixtures/tassadar/runs/sudoku_v0_reference_run_v0/neural_hull_benchmark_report.json`
  proves the committed hull-cache fast path is exact on the declared bounded
  validation window.
- That fact is sufficient for `fast_path_declared_workload_exact`.
- It is not, by itself, a general article-parity decode claim.

The serving and planner boundaries now carry article-shaped evidence honestly.

- `fixtures/tassadar/reports/tassadar_article_executor_session_artifact.json`
  shows direct, fallback, and refusal article sessions with benchmark identity,
  proof identity, readable-log, and symbolic token-trace surfaces.
- `fixtures/tassadar/reports/tassadar_article_hybrid_workflow_artifact.json`
  shows delegated, fallback, and refusal planner-owned article workflows with
  routing receipts and proof identity preserved end to end.

The learned lane is still the blocker.

- `fixtures/tassadar/reports/tassadar_learned_horizon_policy_report.json`
  keeps `learned_article_class_bypass_allowed = false` and freezes
  `unsupported_horizon` for learned article-class long traces.
- `fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0/sequence_fit_report.json`
  still records `full_sequence_fits_model_context = false`.
- `fixtures/tassadar/runs/hungarian_v0_learned_executor_v0/learned_lane_report.json`
  still records `exact_trace_case_count = 0` and
  `final_output_exact_case_count = 0`.

So the repo can now reproduce:

- exact compiled article-class workloads
- bounded exact fast-path truth on the declared workload class
- honest article-shaped serving and planner workflow surfaces

But it still cannot reproduce:

- exact learned article-class workloads
- a green final article-closure acceptance verdict
- truthful article-parity wording

## Green Condition

This audit only turns green when the acceptance artifacts turn green first.

In practice that requires all of the following at once:

- `scripts/check-tassadar-acceptance.sh` regenerates a report with
  `article_parity_language_allowed = true`
- `compiled_article_class.passed = true`
- `fast_path_declared_workload_exact.passed = true`
- `learned_article_class.passed = true`
- `article_closure.passed = true`

Until those fields turn green together, the correct repo-level statement is:

- article parity is not closed

Not:

- the repo fully reproduces the article-shaped Wasm compute claim
