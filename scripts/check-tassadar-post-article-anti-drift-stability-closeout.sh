#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_canonical_computational_model_statement_report
cargo run -p psionic-research --example tassadar_post_article_canonical_computational_model_statement_summary
cargo run -p psionic-eval --example tassadar_post_article_universality_bridge_contract_report
cargo run -p psionic-research --example tassadar_post_article_universality_bridge_contract_summary
cargo run -p psionic-eval --example tassadar_post_article_control_plane_decision_provenance_proof_report
cargo run -p psionic-research --example tassadar_post_article_control_plane_decision_provenance_proof_summary
cargo run -p psionic-eval --example tassadar_post_article_execution_semantics_proof_transport_audit_report
cargo run -p psionic-research --example tassadar_post_article_execution_semantics_proof_transport_audit_summary
cargo run -p psionic-eval --example tassadar_post_article_rebased_universality_verdict_split_report
cargo run -p psionic-research --example tassadar_post_article_rebased_universality_verdict_split_summary
cargo run -p psionic-eval --example tassadar_post_article_universality_portability_minimality_matrix_report
cargo run -p psionic-research --example tassadar_post_article_universality_portability_minimality_matrix_summary
cargo run -p psionic-eval --example tassadar_post_article_canonical_machine_identity_lock_report
cargo run -p psionic-research --example tassadar_post_article_canonical_machine_identity_lock_summary
cargo run -p psionic-eval --example tassadar_post_article_continuation_non_computationality_contract_report
cargo run -p psionic-research --example tassadar_post_article_continuation_non_computationality_contract_summary
cargo run -p psionic-eval --example tassadar_post_article_fast_route_legitimacy_and_carrier_binding_contract_report
cargo run -p psionic-research --example tassadar_post_article_fast_route_legitimacy_and_carrier_binding_contract_summary
cargo run -p psionic-eval --example tassadar_post_article_equivalent_choice_neutrality_and_admissibility_contract_report
cargo run -p psionic-research --example tassadar_post_article_equivalent_choice_neutrality_and_admissibility_contract_summary
cargo run -p psionic-eval --example tassadar_post_article_downward_non_influence_and_served_conformance_report
cargo run -p psionic-research --example tassadar_post_article_downward_non_influence_and_served_conformance_summary
cargo run -p psionic-sandbox --example tassadar_post_article_plugin_charter_authority_boundary_report
cargo run -p psionic-research --example tassadar_post_article_plugin_charter_authority_boundary_summary
cargo run -p psionic-eval --example tassadar_post_article_bounded_weighted_plugin_platform_closeout_audit_report
cargo run -p psionic-research --example tassadar_post_article_bounded_weighted_plugin_platform_closeout_summary
cargo run -p psionic-eval --example tassadar_post_article_anti_drift_stability_closeout_audit_report
cargo run -p psionic-research --example tassadar_post_article_anti_drift_stability_closeout_summary

cargo test -p psionic-transformer anti_drift_contract_keeps_required_surface_and_invalidation_sets_explicit -- --nocapture
cargo test -p psionic-eval anti_drift_closeout_ -- --nocapture
cargo test -p psionic-research anti_drift_summary_ -- --nocapture
cargo test -p psionic-provider post_article_anti_drift_closeout_receipt_projects_summary -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_anti_drift_stability_closeout_audit.report.v1"
  and .closeout_status == "green"
  and .closeout_green == true
  and .all_required_surface_locks_green == true
  and .machine_identity_lock_complete == true
  and .control_and_replay_posture_locked == true
  and .semantics_and_continuation_locked == true
  and .equivalent_choice_and_served_boundary_locked == true
  and .portability_and_minimality_locked == true
  and .plugin_capability_boundary_locked == true
  and .served_and_public_overclaim_suppressed == true
  and .stronger_terminal_claims_require_closure_bundle == true
  and .stronger_plugin_platform_claims_require_closure_bundle == true
  and .machine_identity_binding.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .machine_identity_binding.canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .machine_identity_binding.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and ((.supporting_material_rows | length) == 16)
  and ((.dependency_rows | length) == 9)
  and ((.lock_rows | length) == 12)
  and ((.invalidation_rows | length) == 9)
  and ((.validation_rows | length) == 9)
  and (.lock_rows | all(.green == true))
  and (.dependency_rows | all(.satisfied == true))
  and (.invalidation_rows | all(.present == true))
  and (.validation_rows | all(.green == true))
  and .closure_bundle_issue_id == "TAS-215"
' fixtures/tassadar/reports/tassadar_post_article_anti_drift_stability_closeout_audit_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_anti_drift_stability_closeout_audit.report.v1"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .closeout_status == "green"
  and .supporting_material_row_count == 16
  and .dependency_row_count == 9
  and .lock_row_count == 12
  and .invalidation_row_count == 9
  and .validation_row_count == 9
  and .all_required_surface_locks_green == true
  and .machine_identity_lock_complete == true
  and .control_and_replay_posture_locked == true
  and .semantics_and_continuation_locked == true
  and .equivalent_choice_and_served_boundary_locked == true
  and .portability_and_minimality_locked == true
  and .plugin_capability_boundary_locked == true
  and .stronger_terminal_claims_require_closure_bundle == true
  and .stronger_plugin_platform_claims_require_closure_bundle == true
  and .closure_bundle_issue_id == "TAS-215"
' fixtures/tassadar/reports/tassadar_post_article_anti_drift_stability_closeout_summary.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_canonical_computational_model_statement.report.v1"
  and .next_stability_issue_id == "TAS-215"
  and .statement_status == "green"
  and .statement_green == true
' fixtures/tassadar/reports/tassadar_post_article_canonical_computational_model_statement_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_execution_semantics_proof_transport_audit.report.v1"
  and .next_stability_issue_id == "TAS-215"
  and .audit_status == "green"
  and .audit_green == true
' fixtures/tassadar/reports/tassadar_post_article_execution_semantics_proof_transport_audit_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_continuation_non_computationality_contract.report.v1"
  and .next_stability_issue_id == "TAS-215"
  and .contract_status == "green"
  and .contract_green == true
' fixtures/tassadar/reports/tassadar_post_article_continuation_non_computationality_contract_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_fast_route_legitimacy_and_carrier_binding_contract.report.v1"
  and .next_stability_issue_id == "TAS-215"
  and .contract_status == "green"
  and .contract_green == true
' fixtures/tassadar/reports/tassadar_post_article_fast_route_legitimacy_and_carrier_binding_contract_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_equivalent_choice_neutrality_and_admissibility.report.v1"
  and .next_stability_issue_id == "TAS-215"
  and .contract_status == "green"
  and .contract_green == true
' fixtures/tassadar/reports/tassadar_post_article_equivalent_choice_neutrality_and_admissibility_contract_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_downward_non_influence_and_served_conformance.report.v1"
  and .next_stability_issue_id == "TAS-215"
  and .contract_status == "green"
  and .contract_green == true
' fixtures/tassadar/reports/tassadar_post_article_downward_non_influence_and_served_conformance_report.json >/dev/null

jq -e '
  .bridge_machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and (.reserved_capability_issue_ids | index("TAS-215")) != null
  and (.reserved_capability_issue_ids | index("TAS-214")) == null
' fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_summary.json >/dev/null
