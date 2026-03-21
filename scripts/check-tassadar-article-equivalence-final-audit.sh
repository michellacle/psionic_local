#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-eval --example tassadar_article_equivalence_claim_checker_report
cargo run -p psionic-eval --example tassadar_article_equivalence_final_audit_report
cargo run -p psionic-research --example tassadar_article_equivalence_final_audit_summary

jq -e '
  .blocker_matrix_report.article_equivalence_green == true
  and .acceptance_gate_report.acceptance_status == "green"
  and .acceptance_gate_report.public_claim_allowed == true
  and ((.green_prerequisite_ids | length) == 15)
  and ((.failed_prerequisite_ids | length) == 0)
  and .mechanistic_verdict_green == true
  and .behavioral_verdict_green == true
  and .operational_verdict_green == true
  and .canonical_identity_review.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .canonical_identity_review.canonical_decode_mode == "hull_cache"
  and ((.canonical_identity_review.supported_machine_class_ids | length) == 2)
  and .public_article_equivalence_claim_allowed == true
  and .article_equivalence_green == true
  and (.exclusion_review.optional_open_issue_ids == ["TAS-R1"])
' fixtures/tassadar/reports/tassadar_article_equivalence_claim_checker_report.json >/dev/null

jq -e '
  .all_article_lines_matched == true
  and (.matched_article_line_count > 0)
  and .verdict_review.mechanistic_verdict_green == true
  and .verdict_review.behavioral_verdict_green == true
  and .verdict_review.operational_verdict_green == true
  and .canonical_closure_review.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .canonical_closure_review.canonical_decode_mode == "hull_cache"
  and ((.machine_matrix_review.supported_machine_class_ids | length) == 2)
  and .public_article_equivalence_claim_allowed == true
  and .article_equivalence_green == true
  and (.exclusion_review.optional_open_issue_ids == ["TAS-R1"])
' fixtures/tassadar/reports/tassadar_article_equivalence_final_audit_report.json >/dev/null

jq -e '
  .all_article_lines_matched == true
  and (.matched_article_line_count > 0)
  and .mechanistic_verdict_green == true
  and .behavioral_verdict_green == true
  and .operational_verdict_green == true
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .canonical_decode_mode == "hull_cache"
  and ((.supported_machine_class_ids | length) == 2)
  and .public_article_equivalence_claim_allowed == true
  and .article_equivalence_green == true
  and (.optional_open_issue_ids == ["TAS-R1"])
' fixtures/tassadar/reports/tassadar_article_equivalence_final_audit_summary.json >/dev/null
