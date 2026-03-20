#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-research --example tassadar_article_equivalence_blocker_matrix_summary
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-eval --example tassadar_article_fixture_transformer_parity_report
cargo run -p psionic-research --example tassadar_article_fixture_transformer_parity_summary
cargo run -p psionic-serve --example tassadar_direct_model_weight_execution_proof_report
cargo run -p psionic-eval --example tassadar_article_transformer_reference_linear_exactness_gate_report
cargo run -p psionic-research --example tassadar_article_transformer_reference_linear_exactness_summary
cargo run -p psionic-eval --example tassadar_article_transformer_generalization_gate_report
cargo run -p psionic-research --example tassadar_article_transformer_generalization_summary

jq -e '
  .generalization_green == true
  and .article_equivalence_green == false
  and .case_count == 6
  and .exact_case_count == 6
  and .mismatch_case_count == 0
  and .refused_case_count == 0
  and .out_of_distribution_case_count == 6
  and .randomized_program_review.case_count == 2
  and .adversarial_variant_review.case_count == 2
  and .curriculum_order_review.run_count == 2
  and .exactness_prerequisite_review.passed == true
  and ((.acceptance_gate_tie.tied_requirement_id) == "TAS-171B")
  and ((.acceptance_gate_tie.tied_requirement_satisfied) == true)
  and ((.mismatch_case_ids | length) == 0)
  and ((.refused_case_ids | length) == 0)
' fixtures/tassadar/reports/tassadar_article_transformer_generalization_gate_report.json >/dev/null
