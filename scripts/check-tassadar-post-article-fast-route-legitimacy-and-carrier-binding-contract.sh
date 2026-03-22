#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_fast_route_architecture_selection_report
cargo run -p psionic-research --example tassadar_article_fast_route_architecture_selection_summary
cargo run -p psionic-eval --example tassadar_article_fast_route_implementation_report
cargo run -p psionic-research --example tassadar_article_fast_route_implementation_summary
cargo run -p psionic-runtime --example tassadar_post_article_canonical_computational_model_statement_report
cargo run -p psionic-research --example tassadar_post_article_canonical_computational_model_statement_summary
cargo run -p psionic-eval --example tassadar_post_article_execution_semantics_proof_transport_audit_report
cargo run -p psionic-research --example tassadar_post_article_execution_semantics_proof_transport_audit_summary
cargo run -p psionic-eval --example tassadar_post_article_universality_bridge_contract_report
cargo run -p psionic-research --example tassadar_post_article_universality_bridge_contract_summary
cargo run -p psionic-eval --example tassadar_post_article_canonical_machine_identity_lock_report
cargo run -p psionic-research --example tassadar_post_article_canonical_machine_identity_lock_summary
cargo run -p psionic-eval --example tassadar_post_article_continuation_non_computationality_contract_report
cargo run -p psionic-research --example tassadar_post_article_continuation_non_computationality_contract_summary
cargo run -p psionic-eval --example tassadar_post_article_fast_route_legitimacy_and_carrier_binding_contract_report
cargo run -p psionic-research --example tassadar_post_article_fast_route_legitimacy_and_carrier_binding_contract_summary

cargo test -p psionic-transformer fast_route_legitimacy_and_carrier_binding_contract -- --nocapture
cargo test -p psionic-eval fast_route_legitimacy_and_carrier_binding_contract_report_ -- --nocapture
cargo test -p psionic-runtime canonical_computational_model_statement_report_ -- --nocapture
cargo test -p psionic-eval execution_semantics_proof_transport_audit_ -- --nocapture
cargo test -p psionic-eval continuation_non_computationality_contract_report_ -- --nocapture
cargo test -p psionic-research fast_route_legitimacy_and_carrier_binding_contract_summary_ -- --nocapture
cargo test -p psionic-provider post_article_fast_route_legitimacy_receipt_projects_summary -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_fast_route_legitimacy_and_carrier_binding_contract.report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .carrier_binding_complete == true
  and .unproven_fast_routes_quarantined == true
  and .resumable_family_not_presented_as_direct_machine == true
  and .served_or_plugin_machine_overclaim_refused == true
  and .fast_route_legitimacy_complete == true
  and .machine_identity_binding.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .machine_identity_binding.canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .machine_identity_binding.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .machine_identity_binding.canonical_fast_decode_mode == "hull_cache"
  and .machine_identity_binding.proof_transport_boundary_id == "tassadar.post_article_universal_machine_proof.proof_transport_boundary.v1"
  and ((.supporting_material_rows | length) == 9)
  and ((.dependency_rows | length) == 6)
  and ((.route_family_rows | length) == 6)
  and ((.invalidation_rows | length) == 6)
  and ((.validation_rows | length) == 8)
  and (([.invalidation_rows[] | select(.present == true)] | length) == 0)
  and (([.validation_rows[] | select(.green == false)] | length) == 0)
  and (.route_family_rows | all(.route_family_green == true))
  and ([.route_family_rows[] | select(.route_family_id == "hull_cache")][0].carrier_relation == "canonical_direct_carrier_bound")
  and ([.route_family_rows[] | select(.route_family_id == "reference_linear")][0].carrier_relation == "historical_proof_baseline")
  and ([.route_family_rows[] | select(.route_family_id == "resumable_continuation_family")][0].carrier_relation == "continuation_carrier_only")
  and .next_stability_issue_id == "TAS-212"
  and .closure_bundle_issue_id == "TAS-215"
' fixtures/tassadar/reports/tassadar_post_article_fast_route_legitimacy_and_carrier_binding_contract_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_fast_route_legitimacy_and_carrier_binding_contract.report.v1"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .contract_status == "green"
  and .supporting_material_row_count == 9
  and .dependency_row_count == 6
  and .route_family_row_count == 6
  and .invalidation_row_count == 6
  and .validation_row_count == 8
  and .carrier_binding_complete == true
  and .unproven_fast_routes_quarantined == true
  and .resumable_family_not_presented_as_direct_machine == true
  and .served_or_plugin_machine_overclaim_refused == true
  and .fast_route_legitimacy_complete == true
  and .next_stability_issue_id == "TAS-212"
  and .closure_bundle_issue_id == "TAS-215"
' fixtures/tassadar/reports/tassadar_post_article_fast_route_legitimacy_and_carrier_binding_contract_summary.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_canonical_computational_model_statement.report.v1"
  and .next_stability_issue_id == "TAS-212"
  and .statement_status == "green"
  and .statement_green == true
' fixtures/tassadar/reports/tassadar_post_article_canonical_computational_model_statement_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_execution_semantics_proof_transport_audit.report.v1"
  and .next_stability_issue_id == "TAS-212"
  and .audit_status == "green"
  and .audit_green == true
' fixtures/tassadar/reports/tassadar_post_article_execution_semantics_proof_transport_audit_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_continuation_non_computationality_contract.report.v1"
  and .next_stability_issue_id == "TAS-212"
  and .contract_status == "green"
  and .contract_green == true
' fixtures/tassadar/reports/tassadar_post_article_continuation_non_computationality_contract_report.json >/dev/null

jq -e '
  .bridge_machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and (.reserved_capability_issue_ids | index("TAS-212")) != null
  and (.reserved_capability_issue_ids | index("TAS-211")) == null
' fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_summary.json >/dev/null

jq -e '
  .report_id == "tassadar.article_fast_route_architecture_selection.v1"
  and .selected_candidate_kind == "hull_cache_runtime"
  and .fast_route_selection_green == true
' fixtures/tassadar/reports/tassadar_article_fast_route_architecture_selection_report.json >/dev/null

jq -e '
  .report_id == "tassadar.article_fast_route_implementation.v1"
  and .selected_candidate_kind == "hull_cache_runtime"
  and .fast_route_implementation_green == true
' fixtures/tassadar/reports/tassadar_article_fast_route_implementation_report.json >/dev/null
