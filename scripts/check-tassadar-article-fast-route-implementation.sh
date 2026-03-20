#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-research --example tassadar_article_equivalence_blocker_matrix_summary
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-eval --example tassadar_article_fixture_transformer_parity_report
cargo run -p psionic-research --example tassadar_article_fixture_transformer_parity_summary
cargo run -p psionic-serve --example tassadar_article_transformer_replacement_publication
cargo run -p psionic-serve --example tassadar_article_executor_session_artifact
cargo run -p psionic-serve --example tassadar_article_hybrid_workflow_artifact
cargo run -p psionic-serve --example tassadar_direct_model_weight_execution_proof_report
cargo run -p psionic-eval --example tassadar_article_fast_route_architecture_selection_report
cargo run -p psionic-research --example tassadar_article_fast_route_architecture_selection_summary
cargo run -p psionic-eval --example tassadar_article_fast_route_implementation_report
cargo run -p psionic-research --example tassadar_article_fast_route_implementation_summary

jq -e '
  .fast_route_implementation_green == true
  and .article_equivalence_green == false
  and ((.acceptance_gate_tie.tied_requirement_id) == "TAS-173")
  and ((.acceptance_gate_tie.tied_requirement_satisfied) == true)
  and ((.selected_candidate_kind) == "hull_cache_runtime")
  and ((.descriptor_review.model_id) == "tassadar-article-transformer-trace-bound-trained-v0")
  and (.descriptor_review.hull_cache_supported == true)
  and ((.article_session_review.fast_path_integrated) == true)
  and ((.hybrid_route_review.fast_path_integrated) == true)
  and ((.direct_proof_review.descriptor_binding_green) == true)
' fixtures/tassadar/reports/tassadar_article_fast_route_implementation_report.json >/dev/null
