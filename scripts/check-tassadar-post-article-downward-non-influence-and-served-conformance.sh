#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_canonical_computational_model_statement_report
cargo run -p psionic-research --example tassadar_post_article_canonical_computational_model_statement_summary
cargo run -p psionic-eval --example tassadar_post_article_execution_semantics_proof_transport_audit_report
cargo run -p psionic-research --example tassadar_post_article_execution_semantics_proof_transport_audit_summary
cargo run -p psionic-eval --example tassadar_post_article_universality_bridge_contract_report
cargo run -p psionic-research --example tassadar_post_article_universality_bridge_contract_summary
cargo run -p psionic-serve --example tassadar_post_article_universality_served_conformance_envelope
cargo run -p psionic-eval --example tassadar_post_article_rebased_universality_verdict_split_report
cargo run -p psionic-research --example tassadar_post_article_rebased_universality_verdict_split_summary
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

cargo test -p psionic-transformer downward_non_influence_contract_covers_lower_plane_truth_and_served_envelope -- --nocapture
cargo test -p psionic-eval downward_non_influence_and_served_conformance_report_ -- --nocapture
cargo test -p psionic-runtime canonical_computational_model_statement_report_ -- --nocapture
cargo test -p psionic-eval execution_semantics_proof_transport_audit_ -- --nocapture
cargo test -p psionic-eval continuation_non_computationality_contract_report_ -- --nocapture
cargo test -p psionic-eval fast_route_legitimacy_and_carrier_binding_contract_report_ -- --nocapture
cargo test -p psionic-eval equivalent_choice_neutrality_report_ -- --nocapture
cargo test -p psionic-research downward_non_influence_summary_ -- --nocapture
cargo test -p psionic-provider post_article_downward_non_influence_receipt_projects_summary -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_downward_non_influence_and_served_conformance.report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .downward_non_influence_complete == true
  and .served_conformance_envelope_complete == true
  and .lower_plane_truth_rewrite_refused == true
  and .served_posture_narrower_than_operator_truth == true
  and .served_posture_fail_closed == true
  and .plugin_or_served_overclaim_refused == true
  and .machine_identity_binding.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .machine_identity_binding.canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .machine_identity_binding.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .machine_identity_binding.current_served_internal_compute_profile_id == "tassadar.internal_compute.article_closeout.v1"
  and .machine_identity_binding.served_conformance_envelope_id == "tassadar.post_article_universality_served_conformance_envelope.v1"
  and ((.lower_plane_truth_rows | length) == 6)
  and ((.served_deviation_rows | length) == 3)
  and (.lower_plane_truth_rows | all(.green == true))
  and (.served_deviation_rows | all(.green == true))
  and ([.validation_rows[] | select(.validation_id == "anti_drift_frontier_moves_to_tas_214")][0].green == true)
  and .next_stability_issue_id == "TAS-215"
  and .closure_bundle_issue_id == "TAS-215"
' fixtures/tassadar/reports/tassadar_post_article_downward_non_influence_and_served_conformance_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_downward_non_influence_and_served_conformance.report.v1"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .current_served_internal_compute_profile_id == "tassadar.internal_compute.article_closeout.v1"
  and .contract_status == "green"
  and .lower_plane_truth_row_count == 6
  and .served_deviation_row_count == 3
  and .downward_non_influence_complete == true
  and .served_conformance_envelope_complete == true
  and .lower_plane_truth_rewrite_refused == true
  and .served_posture_narrower_than_operator_truth == true
  and .served_posture_fail_closed == true
  and .plugin_or_served_overclaim_refused == true
  and .next_stability_issue_id == "TAS-215"
  and .closure_bundle_issue_id == "TAS-215"
' fixtures/tassadar/reports/tassadar_post_article_downward_non_influence_and_served_conformance_summary.json >/dev/null

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
  .bridge_machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and (.reserved_capability_issue_ids | index("TAS-216")) != null
  and (.reserved_capability_issue_ids | index("TAS-213")) == null
' fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_summary.json >/dev/null
