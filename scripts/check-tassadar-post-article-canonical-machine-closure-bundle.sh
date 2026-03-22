#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_post_article_canonical_machine_closure_bundle_report
cargo run -p psionic-research --example tassadar_post_article_canonical_machine_closure_bundle_summary

cargo test -p psionic-transformer closure_bundle_contract_keeps_machine_subject_and_inheritance_requirements_explicit -- --nocapture
cargo test -p psionic-eval canonical_machine_closure_bundle_ -- --nocapture
cargo test -p psionic-research closure_bundle_summary_ -- --nocapture
cargo test -p psionic-provider post_article_canonical_machine_closure_bundle_receipt_projects_summary -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_canonical_machine_closure_bundle.report.v1"
  and .closure_subject.closure_bundle_id == "tassadar.post_article.canonical_machine.closure_bundle.v1"
  and .bundle_status == "green"
  and .bundle_green == true
  and .closure_subject.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .closure_subject.canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .closure_subject.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and ((.closure_bundle_contract.artifact_classification_rows | length) == 15)
  and ((.closure_bundle_contract.required_subject_field_rows | length) == 13)
  and ((.closure_bundle_contract.invalidation_laws | length) == 9)
  and ((.closure_bundle_contract.claim_inheritance_rows | length) == 5)
  and ((.supporting_material_rows | length) == 19)
  and ((.dependency_rows | length) == 8)
  and ((.invalidation_rows | length) == 9)
  and ((.validation_rows | length) == 9)
  and .proof_and_audit_classification_complete == true
  and .machine_subject_complete == true
  and .control_execution_and_continuation_bound == true
  and .hidden_state_and_observer_model_bound == true
  and .portability_and_minimality_bound == true
  and .anti_drift_closeout_inherited == true
  and .terminal_claims_must_reference_bundle_digest == true
  and .plugin_claims_must_reference_bundle_digest == true
  and .platform_claims_must_reference_bundle_digest == true
  and .closure_bundle_issue_id == "TAS-215"
  and .next_issue_id == "TAS-217"
  and (.closure_bundle_digest | length) > 0
' fixtures/tassadar/reports/tassadar_post_article_canonical_machine_closure_bundle_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_canonical_machine_closure_bundle.report.v1"
  and .closure_bundle_id == "tassadar.post_article.canonical_machine.closure_bundle.v1"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .bundle_status == "green"
  and .supporting_material_row_count == 19
  and .dependency_row_count == 8
  and .invalidation_row_count == 9
  and .validation_row_count == 9
  and .proof_and_audit_classification_complete == true
  and .machine_subject_complete == true
  and .control_execution_and_continuation_bound == true
  and .hidden_state_and_observer_model_bound == true
  and .portability_and_minimality_bound == true
  and .anti_drift_closeout_inherited == true
  and .terminal_claims_must_reference_bundle_digest == true
  and .plugin_claims_must_reference_bundle_digest == true
  and .platform_claims_must_reference_bundle_digest == true
  and .closure_bundle_issue_id == "TAS-215"
  and .next_issue_id == "TAS-217"
  and (.closure_bundle_digest | length) > 0
' fixtures/tassadar/reports/tassadar_post_article_canonical_machine_closure_bundle_summary.json >/dev/null
