#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_bundle
cargo run -p psionic-sandbox --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_report
cargo run -p psionic-research --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_summary
cargo run -p psionic-serve --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_publication
cargo run -p psionic-runtime --example tassadar_post_article_plugin_invocation_receipts_and_replay_classes_bundle
cargo run -p psionic-eval --example tassadar_post_article_plugin_invocation_receipts_and_replay_classes_report
cargo run -p psionic-research --example tassadar_post_article_plugin_invocation_receipts_and_replay_classes_summary
cargo run -p psionic-runtime --example tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_bundle
cargo run -p psionic-sandbox --example tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_report
cargo run -p psionic-research --example tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_summary
cargo test -p psionic-provider post_article_plugin_invocation_receipts_receipt_projects_summary -- --nocapture
cargo test -p psionic-provider post_article_plugin_world_mount_admissibility_receipt_projects_summary -- --nocapture
cargo test -p psionic-serve post_article_plugin_runtime_api_publication_keeps_served_surface_blocked -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article_plugin_world_mount_envelope_compiler_and_admissibility.runtime_bundle.v1"
  and .world_mount_envelope_compiler_id == "tassadar.plugin_runtime.world_mount_envelope_compiler.v1"
  and .admissibility_contract_id == "tassadar.plugin_runtime.admissibility.v1"
  and .host_owned_runtime_api_id == "tassadar.plugin_runtime.host_owned_api.v1"
  and .invocation_receipt_profile_id == "tassadar.plugin_runtime.invocation_receipts.v1"
  and ((.admissibility_rule_rows | length) == 9)
  and ((.candidate_set_rows | length) == 5)
  and ((.equivalent_choice_rows | length) == 5)
  and ((.envelope_rows | length) == 2)
  and ((.case_receipts | length) == 7)
  and .exact_admitted_case_count == 2
  and .exact_denied_case_count == 2
  and .exact_suppressed_case_count == 2
  and .exact_quarantined_case_count == 1
' fixtures/tassadar/runs/tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_v1/tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_bundle.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_world_mount_envelope_compiler_and_admissibility.report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .machine_identity_binding.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .machine_identity_binding.world_mount_envelope_compiler_id == "tassadar.plugin_runtime.world_mount_envelope_compiler.v1"
  and .machine_identity_binding.admissibility_contract_id == "tassadar.plugin_runtime.admissibility.v1"
  and ((.dependency_rows | length) == 5)
  and ((.admissibility_rule_rows | length) == 9)
  and ((.candidate_set_rows | length) == 5)
  and ((.equivalent_choice_rows | length) == 5)
  and ((.envelope_rows | length) == 2)
  and ((.case_rows | length) == 7)
  and ((.validation_rows | length) == 9)
  and .operator_internal_only_posture == true
  and .admissibility_frozen == true
  and .candidate_set_enumeration_frozen == true
  and .equivalent_choice_model_frozen == true
  and .world_mount_envelope_compiler_frozen == true
  and .receipt_visible_filtering_required == true
  and .version_constraint_binding_required == true
  and .trust_posture_binding_required == true
  and .publication_posture_binding_required == true
  and .denial_behavior_frozen == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (.deferred_issue_ids == [])
' fixtures/tassadar/reports/tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_world_mount_envelope_compiler_and_admissibility.report.v1"
  and .packet_abi_version == "packet.v1"
  and .host_owned_runtime_api_id == "tassadar.plugin_runtime.host_owned_api.v1"
  and .engine_abstraction_id == "tassadar.plugin_runtime.engine_abstraction.v1"
  and .invocation_receipt_profile_id == "tassadar.plugin_runtime.invocation_receipts.v1"
  and .world_mount_envelope_compiler_id == "tassadar.plugin_runtime.world_mount_envelope_compiler.v1"
  and .admissibility_contract_id == "tassadar.plugin_runtime.admissibility.v1"
  and .contract_status == "green"
  and .dependency_row_count == 5
  and .admissibility_rule_row_count == 9
  and .candidate_set_row_count == 5
  and .equivalent_choice_row_count == 5
  and .envelope_row_count == 2
  and .case_row_count == 7
  and .validation_row_count == 9
  and (.deferred_issue_ids == [])
  and .operator_internal_only_posture == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_summary.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_invocation_receipts_and_replay_classes.report.v1"
  and (.deferred_issue_ids == [])
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_invocation_receipts_and_replay_classes_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_invocation_receipts_and_replay_classes.report.v1"
  and (.deferred_issue_ids == [])
  and .rebase_claim_allowed == true
' fixtures/tassadar/reports/tassadar_post_article_plugin_invocation_receipts_and_replay_classes_summary.json >/dev/null

jq -e '
  .publication_id == "tassadar.post_article_plugin_runtime_api_and_engine_abstraction.publication.v1"
  and (.served_plugin_surface_ids == [])
  and (.blocked_by == [])
  and .plugin_publication_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_publication.json >/dev/null
