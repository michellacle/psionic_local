#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_post_article_canonical_route_universal_substrate_gate_report
cargo run -p psionic-research --example tassadar_post_article_canonical_route_universal_substrate_gate_summary
cargo test -p psionic-provider canonical_route_universal_substrate_gate_receipt_projects_summary -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_canonical_route_universal_substrate_gate.report.v1"
  and .gate_status == "green"
  and .gate_green == true
  and .bounded_universality_story_carried == true
  and .proof_rebinding_complete == true
  and .witness_suite_reissued == true
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and ((.supporting_material_rows | length) == 9)
  and ((.supporting_material_rows | map(select(.material_class == "proof_carrying" and .satisfied == true)) | length) == 8)
  and ((.supporting_material_rows | map(select(.material_class == "observational_context")) | length) == 1)
  and ((.portability_rows | length) == 3)
  and ((.refusal_boundary_rows | length) == 2)
  and ((.validation_rows | length) == 7)
  and .universal_substrate_gate_allowed == true
  and (.deferred_issue_ids == ["TAS-193"])
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_canonical_route_universal_substrate_gate_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_canonical_route_universal_substrate_gate.report.v1"
  and .gate_status == "green"
  and .bounded_universality_story_carried == true
  and .proof_rebinding_complete == true
  and .witness_suite_reissued == true
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .portability_row_count == 3
  and .refusal_boundary_row_count == 2
  and .universal_substrate_gate_allowed == true
  and (.deferred_issue_ids == ["TAS-193"])
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_canonical_route_universal_substrate_gate_summary.json >/dev/null
