#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-sandbox --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_report
cargo run -p psionic-runtime --example tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_bundle
cargo run -p psionic-sandbox --example tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_report
cargo run -p psionic-research --example tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_summary
cargo run -p psionic-eval --example tassadar_post_article_control_plane_decision_provenance_proof_report
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
cargo run -p psionic-eval --example tassadar_post_article_equivalent_choice_neutrality_and_admissibility_contract_report
cargo run -p psionic-research --example tassadar_post_article_equivalent_choice_neutrality_and_admissibility_contract_summary

cargo test -p psionic-transformer equivalent_choice_neutrality_contract_covers_declared_choice_classes -- --nocapture
cargo test -p psionic-eval equivalent_choice_neutrality_report_ -- --nocapture
cargo test -p psionic-runtime canonical_computational_model_statement_report_ -- --nocapture
cargo test -p psionic-eval execution_semantics_proof_transport_audit_ -- --nocapture
cargo test -p psionic-eval continuation_non_computationality_contract_report_ -- --nocapture
cargo test -p psionic-research equivalent_choice_neutrality_summary_ -- --nocapture
cargo test -p psionic-provider post_article_equivalent_choice_neutrality_receipt_projects_summary -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_equivalent_choice_neutrality_and_admissibility.report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .equivalent_choice_neutrality_complete == true
  and .admissibility_narrowing_receipt_visible == true
  and .hidden_ordering_or_ranking_quarantined == true
  and .latency_cost_and_soft_failure_channels_blocked == true
  and .served_or_plugin_equivalence_overclaim_refused == true
  and .machine_identity_binding.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .machine_identity_binding.canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .machine_identity_binding.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .machine_identity_binding.control_plane_equivalent_choice_relation_id == "singleton_exact_control_trace.v1"
  and .machine_identity_binding.runtime_api_report_id == "tassadar.post_article_plugin_runtime_api_and_engine_abstraction.report.v1"
  and .machine_identity_binding.admissibility_report_id == "tassadar.post_article_plugin_world_mount_envelope_compiler_and_admissibility.report.v1"
  and ((.equivalent_choice_class_rows | length) == 5)
  and ((.case_binding_rows | length) == 7)
  and (.equivalent_choice_class_rows | all(.green == true))
  and (.case_binding_rows | all(.green == true))
  and ([.equivalent_choice_class_rows[] | select(.equivalent_choice_class_id == "choice.search_core_pair.closed_world_neutral.v1")][0].bounded_candidate_count == 2)
  and ([.equivalent_choice_class_rows[] | select(.equivalent_choice_class_id == "choice.search_core_pair.closed_world_neutral.v1")][0].neutral_choice_auditable == true)
  and (([.case_binding_rows[] | select(.receipt_visible_justification_present == true)] | length) >= 1)
  and .next_stability_issue_id == "TAS-215"
  and .closure_bundle_issue_id == "TAS-215"
' fixtures/tassadar/reports/tassadar_post_article_equivalent_choice_neutrality_and_admissibility_contract_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_equivalent_choice_neutrality_and_admissibility.report.v1"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .control_plane_equivalent_choice_relation_id == "singleton_exact_control_trace.v1"
  and .admissibility_contract_id == "tassadar.plugin_runtime.admissibility.v1"
  and .contract_status == "green"
  and .equivalent_choice_class_row_count == 5
  and .case_binding_row_count == 7
  and .equivalent_choice_neutrality_complete == true
  and .admissibility_narrowing_receipt_visible == true
  and .hidden_ordering_or_ranking_quarantined == true
  and .latency_cost_and_soft_failure_channels_blocked == true
  and .served_or_plugin_equivalence_overclaim_refused == true
  and .next_stability_issue_id == "TAS-215"
  and .closure_bundle_issue_id == "TAS-215"
' fixtures/tassadar/reports/tassadar_post_article_equivalent_choice_neutrality_and_admissibility_contract_summary.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_fast_route_legitimacy_and_carrier_binding_contract.report.v1"
  and .next_stability_issue_id == "TAS-215"
  and .contract_status == "green"
  and .contract_green == true
' fixtures/tassadar/reports/tassadar_post_article_fast_route_legitimacy_and_carrier_binding_contract_report.json >/dev/null

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
  .report_id == "tassadar.post_article_canonical_computational_model_statement.report.v1"
  and .next_stability_issue_id == "TAS-215"
  and .statement_status == "green"
  and .statement_green == true
' fixtures/tassadar/reports/tassadar_post_article_canonical_computational_model_statement_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_canonical_machine_identity_lock.report.v1"
  and .lock_status == "green"
  and .lock_green == true
' fixtures/tassadar/reports/tassadar_post_article_canonical_machine_identity_lock_report.json >/dev/null

jq -e '
  .bridge_machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and (.reserved_capability_issue_ids | index("TAS-215")) != null
  and (.reserved_capability_issue_ids | index("TAS-212")) == null
' fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_summary.json >/dev/null
