#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_post_article_universality_witness_suite_reissue_report
cargo run -p psionic-research --example tassadar_post_article_universality_witness_suite_reissue_summary
cargo test -p psionic-provider witness_suite_reissue_receipt_projects_summary -- --nocapture

jq -e '
  .witness_suite_status == "green"
  and .proof_rebinding_complete == true
  and .witness_suite_reissued == true
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .exact_family_count == 5
  and .refusal_boundary_count == 2
  and ((.supporting_material_rows | length) == 9)
  and ((.reissued_family_rows | length) == 7)
  and ((.validation_rows | length) == 8)
  and .universal_substrate_gate_allowed == false
  and (.deferred_issue_ids == ["TAS-192"])
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_universality_witness_suite_reissue_report.json >/dev/null

jq -e '
  .witness_suite_status == "green"
  and .proof_rebinding_complete == true
  and .witness_suite_reissued == true
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .exact_family_count == 5
  and .refusal_boundary_count == 2
  and (.deferred_issue_ids == ["TAS-192"])
  and .universal_substrate_gate_allowed == false
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_universality_witness_suite_reissue_summary.json >/dev/null
