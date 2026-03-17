# Tassadar Article-Parity Closeout Audit

Date: March 17, 2026

## Result

Final article-parity closure is green.

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
- `fixtures/tassadar/runs/hungarian_10x10_v0_learned_article_executor_v0`
- `fixtures/tassadar/reports/tassadar_article_executor_session_artifact.json`
- `fixtures/tassadar/reports/tassadar_article_hybrid_workflow_artifact.json`
- `fixtures/tassadar/reports/tassadar_learned_horizon_policy_report.json`

## Acceptance Alignment

The machine-readable acceptance verdict is:

- `article_parity_language_allowed = true`
- `compiled_article_class.passed = true`
- `fast_path_declared_workload_exact.passed = true`
- `learned_article_class.passed = true`
- `article_closure.passed = true`
- `current_truth_holds = true`

That means the repo can now use article-parity wording honestly, with the
learned lane scoped to benchmark-corpus exactness on the committed
Hungarian-10x10 article bundle.

## Reproduction Commands

- `scripts/check-tassadar-acceptance.sh`
- `scripts/check-tassadar-compiled-article-closure.sh`
- `cargo test -p psionic-eval neural_hull_benchmark_reports_direct_hull_selection_and_window_cap`
- `cargo run -p psionic-serve --example tassadar_article_executor_session_artifact`
- `cargo run -p psionic-serve --example tassadar_article_hybrid_workflow_artifact`
- `cargo run -p psionic-train --example tassadar_hungarian_10x10_article_learned_run`

If those commands do not reproduce the referenced artifacts and verdicts from a
local checkout, this audit is not green.

## Decisive Facts

The compiled article-class bar remains genuinely green.

- `fixtures/tassadar/reports/tassadar_compiled_article_closure_report.json`
  records compiled article closure as green.
- `fixtures/tassadar/runs/sudoku_9x9_v0_compiled_executor_v0` is the exact
  compiled 9x9 Sudoku article-sized root.
- `fixtures/tassadar/runs/hungarian_10x10_v0_compiled_executor_v0` is the
  exact compiled 10x10 Hungarian article-sized root.
- `fixtures/tassadar/runs/compiled_kernel_suite_v0` widens compiled
  article-shaped evidence beyond Sudoku and Hungarian to arithmetic, memory,
  branch, and loop-heavy kernels.

The fast decode story remains green only on its declared workload class.

- `fixtures/tassadar/runs/sudoku_v0_reference_run_v0/neural_hull_benchmark_report.json`
  proves the committed hull-cache fast path is exact on the declared bounded
  validation window.
- That fact is sufficient for `fast_path_declared_workload_exact`.
- It remains a bounded fast-path equivalence claim, not a universal executor
  claim.

The learned article-class bar is now green on one exact benchmark corpus.

- `fixtures/tassadar/runs/hungarian_10x10_v0_learned_article_executor_v0/article_learned_benchmark_report.json`
  records `validation = 10000`, `test = 10000`, `exact_trace_case_count = 1`
  on both held committed cases, and `passed = true`.
- `fixtures/tassadar/runs/hungarian_10x10_v0_learned_article_executor_v0/model_artifact.json`
  records the explicit prompt-conditioned surface:
  `prompt_summary_bucket_count = 4096`,
  `prompt_summary_target_output_bias_row_values`,
  and `relative_target_trace_schema_output_bias`.
- `fixtures/tassadar/reports/tassadar_learned_horizon_policy_report.json`
  now records `guard_status = exact_benchmark_landed` and
  `benchmark_status = exact`.

The serving and planner boundaries continue to carry article-shaped evidence
honestly.

- `fixtures/tassadar/reports/tassadar_article_executor_session_artifact.json`
  shows direct, fallback, and refusal article sessions with benchmark identity,
  proof identity, readable-log, and symbolic token-trace surfaces.
- `fixtures/tassadar/reports/tassadar_article_hybrid_workflow_artifact.json`
  shows delegated, fallback, and refusal planner-owned article workflows with
  routing receipts and proof identity preserved end to end.

## Scope Note

This green closeout does not claim that every learned long-horizon workload is
solved.

It claims something narrower and honest:

- the repo can reproduce the compiled article workloads
- the repo can reproduce the declared fast-path benchmark truth
- the repo can reproduce one exact learned article benchmark on the fixed
  Hungarian-10x10 corpus

That is the current article-parity bar encoded by the committed acceptance
artifacts.
