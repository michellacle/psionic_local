#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_bundle
cargo run -p psionic-sandbox --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_report
cargo run -p psionic-research --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_summary
cargo run -p psionic-runtime --example tassadar_post_article_plugin_invocation_receipts_and_replay_classes_bundle
cargo run -p psionic-eval --example tassadar_post_article_plugin_invocation_receipts_and_replay_classes_report
cargo run -p psionic-research --example tassadar_post_article_plugin_invocation_receipts_and_replay_classes_summary
cargo run -p psionic-runtime --example tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_bundle
cargo run -p psionic-sandbox --example tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_report
cargo run -p psionic-research --example tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_summary
cargo run -p psionic-runtime --example tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_bundle
cargo run -p psionic-sandbox --example tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_report
cargo run -p psionic-eval --example tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_eval_report
cargo run -p psionic-research --example tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_summary
cargo test -p psionic-provider post_article_plugin_world_mount_admissibility_receipt_projects_summary -- --nocapture
cargo test -p psionic-provider post_article_plugin_conformance_harness_receipt_projects_summary -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article_plugin_conformance_sandbox_and_benchmark_harness.runtime_bundle.v1"
  and .conformance_harness_id == "tassadar.plugin_runtime.conformance_harness.v1"
  and .benchmark_harness_id == "tassadar.plugin_runtime.benchmark_harness.v1"
  and .host_owned_runtime_api_id == "tassadar.plugin_runtime.host_owned_api.v1"
  and .invocation_receipt_profile_id == "tassadar.plugin_runtime.invocation_receipts.v1"
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
  and .machine_identity_binding.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .machine_identity_binding.conformance_harness_id == "tassadar.plugin_runtime.conformance_harness.v1"
  and .machine_identity_binding.benchmark_harness_id == "tassadar.plugin_runtime.benchmark_harness.v1"
  and ((.dependency_rows | length) == 6)
  and ((.conformance_rows | length) == 9)
  and ((.workflow_rows | length) == 5)
  and ((.isolation_negative_rows | length) == 8)
  and ((.benchmark_rows | length) == 7)
  and ((.validation_rows | length) == 12)
  and .operator_internal_only_posture == true
  and .static_harness_only == true
  and .host_scripted_trace_only == true
  and .receipt_integrity_frozen == true
  and .envelope_compatibility_explicit == true
  and .workflow_integrity_frozen == true
  and .failure_domain_isolation_frozen == true
  and .side_channel_negatives_green == true
  and .covert_channel_negatives_green == true
  and .hot_swap_compatibility_frozen == true
  and .replay_under_partial_cancellation_frozen == true
  and .benchmark_paths_measured == true
  and .evidence_overhead_explicit == true
  and .timeout_enforcement_measured == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (.deferred_issue_ids == [])
' fixtures/tassadar/reports/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_conformance_sandbox_and_benchmark_harness.eval_report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .machine_identity_binding.conformance_harness_id == "tassadar.plugin_runtime.conformance_harness.v1"
  and .machine_identity_binding.benchmark_harness_id == "tassadar.plugin_runtime.benchmark_harness.v1"
  and ((.dependency_rows | length) == 7)
  and ((.conformance_rows | length) == 9)
  and ((.workflow_rows | length) == 5)
  and ((.isolation_negative_rows | length) == 8)
  and ((.benchmark_rows | length) == 7)
  and ((.validation_rows | length) == 11)
  and .conformance_sandbox_green == true
  and .cold_path_benchmarked == true
  and .warm_path_benchmarked == true
  and .pooled_path_benchmarked == true
  and .queued_path_benchmarked == true
  and .cancelled_path_benchmarked == true
  and .queue_saturation_explicit == true
  and .cancellation_latency_bounded == true
  and .evidence_overhead_explicit == true
  and .timeout_enforcement_measured == true
  and .receipt_integrity_and_envelope_compatibility_explicit == true
  and .operator_internal_only_posture == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (.deferred_issue_ids == [])
' fixtures/tassadar/reports/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_eval_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_conformance_sandbox_and_benchmark_harness.eval_report.v1"
  and .conformance_harness_id == "tassadar.plugin_runtime.conformance_harness.v1"
  and .benchmark_harness_id == "tassadar.plugin_runtime.benchmark_harness.v1"
  and .contract_status == "green"
  and .dependency_row_count == 7
  and .conformance_row_count == 9
  and .workflow_row_count == 5
  and .isolation_negative_row_count == 8
  and .benchmark_row_count == 7
  and .validation_row_count == 11
  and (.deferred_issue_ids == [])
  and .conformance_sandbox_green == true
  and .operator_internal_only_posture == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_summary.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_world_mount_envelope_compiler_and_admissibility.report.v1"
  and (.deferred_issue_ids == [])
  and .rebase_claim_allowed == true
' fixtures/tassadar/reports/tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_world_mount_envelope_compiler_and_admissibility.report.v1"
  and (.deferred_issue_ids == [])
  and .rebase_claim_allowed == true
' fixtures/tassadar/reports/tassadar_post_article_plugin_world_mount_envelope_compiler_and_admissibility_summary.json >/dev/null
