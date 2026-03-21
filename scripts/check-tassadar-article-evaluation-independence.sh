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
cargo run -p psionic-eval --example tassadar_article_evaluation_independence_audit_report
cargo run -p psionic-research --example tassadar_article_evaluation_independence_summary

jq -e '
  .evaluation_independence_green == true
  and .article_equivalence_green == true
  and ((.training_case_rows | length) == 4)
  and ((.evaluation_case_rows | length) == 6)
  and .generalization_prerequisite_review.passed == true
  and .training_lineage_review.passed == true
  and .exclusion_manifest.passed == true
  and .near_duplicate_review.passed == true
  and .generator_overlap_audit.passed == true
  and .feature_distribution_review.passed == true
  and ((.acceptance_gate_tie.tied_requirement_id) == "TAS-171C")
  and ((.acceptance_gate_tie.tied_requirement_satisfied) == true)
  and ((.exclusion_manifest.exact_case_id_overlap_ids | length) == 0)
  and ((.exclusion_manifest.exact_source_token_overlap_case_ids | length) == 0)
  and ((.exclusion_manifest.exact_target_token_overlap_case_ids | length) == 0)
  and ((.exclusion_manifest.exact_sequence_overlap_case_ids | length) == 0)
  and ((.near_duplicate_review.near_duplicate_pair_count) == 0)
  and ((.generator_overlap_audit.shared_generator_ids | length) == 0)
  and ((.generator_overlap_audit.shared_generator_rule_digests | length) == 0)
  and ((.feature_distribution_review.shared_profile_ids | length) == 0)
' fixtures/tassadar/reports/tassadar_article_evaluation_independence_audit_report.json >/dev/null
