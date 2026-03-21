#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${REGENERATE:-0}" == "1" ]]; then
  cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
  cargo run -p psionic-research --example tassadar_article_equivalence_blocker_matrix_summary
  cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
  cargo run -p psionic-eval --example tassadar_article_fixture_transformer_parity_report
  cargo run -p psionic-research --example tassadar_article_fixture_transformer_parity_summary
  cargo run -p psionic-serve --example tassadar_article_transformer_replacement_publication
  cargo run -p psionic-eval --example tassadar_article_class_benchmark_report
  cargo run -p psionic-eval --example tassadar_hull_cache_closure_report
  cargo run -p psionic-serve --example tassadar_article_fast_route_exactness_session_artifact
  cargo run -p psionic-serve --example tassadar_article_fast_route_exactness_hybrid_workflow_artifact
  cargo run -p psionic-eval --example tassadar_article_fast_route_architecture_selection_report
  cargo run -p psionic-research --example tassadar_article_fast_route_architecture_selection_summary
  cargo run -p psionic-eval --example tassadar_article_fast_route_implementation_report
  cargo run -p psionic-research --example tassadar_article_fast_route_implementation_summary
  cargo run -p psionic-eval --example tassadar_article_fast_route_exactness_report
  cargo run -p psionic-research --example tassadar_article_fast_route_exactness_summary
fi

jq -e '
  .exactness_green == true
  and .article_equivalence_green == true
  and ((.acceptance_gate_tie.tied_requirement_id) == "TAS-174")
  and ((.acceptance_gate_tie.tied_requirement_satisfied) == true)
  and ((.implementation_prerequisite.selected_candidate_kind) == "hull_cache_runtime")
  and ((.implementation_prerequisite.fast_route_implementation_green) == true)
  and ((.hull_cache_closure_review.all_article_workloads_exact) == true)
  and ((.hull_cache_closure_review.long_loop_fallback_case_count) == 0)
  and ((.hull_cache_closure_review.sudoku_fallback_case_count) == 0)
  and ((.hull_cache_closure_review.hungarian_fallback_case_count) == 0)
  and ((.article_session_reviews | length) == 3)
  and ((.article_session_reviews | map(select(.exact_direct_hull_cache == true)) | length) == 3)
  and ((.hybrid_route_reviews | length) == 3)
  and ((.hybrid_route_reviews | map(select(.exact_direct_hull_cache == true)) | length) == 3)
  and ((.article_session_reviews[] | select(.case_name == "direct_long_loop_hull") | .selection_state) == "direct")
  and ((.article_session_reviews[] | select(.case_name == "direct_sudoku_v0_hull") | .selection_state) == "direct")
  and ((.article_session_reviews[] | select(.case_name == "direct_hungarian_hull") | .selection_state) == "direct")
  and ((.hybrid_route_reviews[] | select(.case_name == "delegated_long_loop_hull") | .selection_state) == "direct")
  and ((.hybrid_route_reviews[] | select(.case_name == "delegated_sudoku_v0_hull") | .selection_state) == "direct")
  and ((.hybrid_route_reviews[] | select(.case_name == "delegated_hungarian_hull") | .selection_state) == "direct")
' fixtures/tassadar/reports/tassadar_article_fast_route_exactness_report.json >/dev/null

jq -e '
  .tied_requirement_id == "TAS-174"
  and .tied_requirement_satisfied == true
  and .implementation_prerequisite_green == true
  and .all_article_workloads_exact == true
  and .article_session_direct_case_count == 3
  and .hybrid_direct_case_count == 3
  and .exactness_green == true
  and .article_equivalence_green == true
' fixtures/tassadar/reports/tassadar_article_fast_route_exactness_summary.json >/dev/null
