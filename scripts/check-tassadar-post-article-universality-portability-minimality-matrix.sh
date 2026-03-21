#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_post_article_universality_portability_minimality_matrix_report
cargo run -p psionic-research --example tassadar_post_article_universality_portability_minimality_matrix_summary
cargo run -p psionic-serve --example tassadar_post_article_universality_served_conformance_envelope
cargo test -p psionic-provider universality_portability_minimality_matrix_receipt_projects_summary -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_universality_portability_minimality_matrix.report.v1"
  and .matrix_status == "green"
  and .matrix_green == true
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and ((.supporting_material_rows | length) == 8)
  and ((.supporting_material_rows | map(select(.material_class == "proof_carrying" and .satisfied == true)) | length) == 7)
  and ((.supporting_material_rows | map(select(.material_class == "observational_context")) | length) == 1)
  and ((.machine_matrix_rows | length) == 3)
  and ((.route_classification_rows | length) == 4)
  and ((.route_classification_rows | map(select(.route_carrier_status == "inside_universality_carrier")) | length) == 1)
  and ((.route_classification_rows | map(select(.route_carrier_status == "outside_universality_carrier_acceleration_only")) | length) == 1)
  and ((.route_classification_rows | map(select(.route_carrier_status == "outside_universality_carrier_research_only")) | length) == 2)
  and ((.minimality_rows | length) == 3)
  and ((.validation_rows | length) == 8)
  and .bounded_universality_story_carried == true
  and .universal_substrate_gate_allowed == true
  and .served_suppression_boundary_preserved == true
  and .served_conformance_envelope_defined == true
  and (.deferred_issue_ids == ["TAS-194"])
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_universality_portability_minimality_matrix_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_universality_portability_minimality_matrix.report.v1"
  and .matrix_status == "green"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .machine_row_count == 3
  and .route_row_count == 4
  and .minimality_row_count == 3
  and .validation_row_count == 8
  and .served_suppression_boundary_preserved == true
  and .served_conformance_envelope_defined == true
  and (.deferred_issue_ids == ["TAS-194"])
  and .universal_substrate_gate_allowed == true
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_universality_portability_minimality_matrix_summary.json >/dev/null

jq -e '
  .publication_id == "tassadar.post_article_universality_served_conformance_envelope.v1"
  and .current_served_internal_compute_profile_id == "tassadar.internal_compute.article_closeout.v1"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .selected_decode_mode == "tassadar.decode.hull_cache.v1"
  and (.supported_machine_class_ids == ["host_cpu_aarch64", "host_cpu_x86_64"])
  and ((.allowed_narrower_deviation_ids | length) == 3)
  and ((.required_identical_property_ids | length) == 5)
  and ((.fail_closed_condition_ids | length) == 5)
  and .matrix_green == true
  and .route_minimality_publication_green == true
  and .cross_machine_reproducibility_green == true
  and .served_suppression_boundary_preserved == true
  and .served_public_universality_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_universality_served_conformance_envelope.json >/dev/null
