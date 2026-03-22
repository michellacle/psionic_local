#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_bundle
cargo run -p psionic-sandbox --example tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_report
cargo run -p psionic-eval --example tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_eval_report
cargo run -p psionic-research --example tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_summary
cargo run -p psionic-runtime --example tassadar_post_article_plugin_result_binding_schema_stability_and_composition_bundle
cargo run -p psionic-eval --example tassadar_post_article_plugin_result_binding_schema_stability_and_composition_report
cargo run -p psionic-research --example tassadar_post_article_plugin_result_binding_schema_stability_and_composition_summary
cargo test -p psionic-provider post_article_plugin_conformance_harness_receipt_projects_summary -- --nocapture
cargo test -p psionic-provider post_article_plugin_result_binding_receipt_projects_summary -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article_plugin_conformance_sandbox_and_benchmark_harness.runtime_bundle.v1"
  and ((.conformance_rows | length) == 9)
  and ((.workflow_rows | length) == 5)
  and ((.isolation_negative_rows | length) == 8)
  and ((.benchmark_rows | length) == 7)
  and ((.trace_receipts | length) == 9)
' fixtures/tassadar/runs/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_v1/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_bundle.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_conformance_sandbox_and_benchmark_harness.report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and (.deferred_issue_ids == [])
  and .rebase_claim_allowed == true
  and .weighted_plugin_control_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_conformance_sandbox_and_benchmark_harness.eval_report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and (.deferred_issue_ids == [])
  and .rebase_claim_allowed == true
  and .weighted_plugin_control_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_eval_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_conformance_sandbox_and_benchmark_harness.eval_report.v1"
  and (.deferred_issue_ids == [])
  and .rebase_claim_allowed == true
  and .weighted_plugin_control_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_summary.json >/dev/null

jq -e '
  .bundle_id == "tassadar.post_article_plugin_result_binding_schema_stability_and_composition.runtime_bundle.v1"
  and .result_binding_contract_id == "tassadar.weighted_plugin.result_binding_contract.v1"
  and .model_loop_return_profile_id == "tassadar.weighted_plugin.model_loop_return_profile.v1"
  and ((.binding_rows | length) == 5)
  and ((.evidence_boundary_rows | length) == 3)
  and ((.composition_rows | length) == 4)
  and ((.negative_rows | length) == 4)
' fixtures/tassadar/runs/tassadar_post_article_plugin_result_binding_schema_stability_and_composition_v1/tassadar_post_article_plugin_result_binding_schema_stability_and_composition_bundle.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_result_binding_schema_stability_and_composition.report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .machine_identity_binding.result_binding_contract_id == "tassadar.weighted_plugin.result_binding_contract.v1"
  and .machine_identity_binding.model_loop_return_profile_id == "tassadar.weighted_plugin.model_loop_return_profile.v1"
  and ((.dependency_rows | length) == 7)
  and ((.binding_rows | length) == 5)
  and ((.evidence_boundary_rows | length) == 3)
  and ((.composition_rows | length) == 4)
  and ((.negative_rows | length) == 4)
  and ((.validation_rows | length) == 12)
  and .result_binding_contract_green == true
  and .explicit_output_to_state_digest_binding == true
  and .schema_evolution_fail_closed == true
  and .typed_refusal_normalization_preserved == true
  and .version_skew_fail_closed == true
  and .proof_vs_observational_boundary_explicit == true
  and .semantic_composition_closure_green == true
  and .non_lossy_schema_transition_required == true
  and .ambiguous_composition_blocked == true
  and .adapter_defined_return_path_blocked == true
  and .operator_internal_only_posture == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (.deferred_issue_ids == [])
' fixtures/tassadar/reports/tassadar_post_article_plugin_result_binding_schema_stability_and_composition_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_result_binding_schema_stability_and_composition.report.v1"
  and .result_binding_contract_id == "tassadar.weighted_plugin.result_binding_contract.v1"
  and .model_loop_return_profile_id == "tassadar.weighted_plugin.model_loop_return_profile.v1"
  and .contract_status == "green"
  and .dependency_row_count == 7
  and .binding_row_count == 5
  and .evidence_boundary_row_count == 3
  and .composition_row_count == 4
  and .negative_row_count == 4
  and .validation_row_count == 12
  and .result_binding_contract_green == true
  and .semantic_composition_closure_green == true
  and .operator_internal_only_posture == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (.deferred_issue_ids == [])
' fixtures/tassadar/reports/tassadar_post_article_plugin_result_binding_schema_stability_and_composition_summary.json >/dev/null
