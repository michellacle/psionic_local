#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_post_article_canonical_route_semantic_preservation_audit_report
cargo run -p psionic-research --example tassadar_post_article_canonical_route_semantic_preservation_summary
cargo test -p psionic-provider semantic_preservation_receipt_projects_summary -- --nocapture

jq -e '
  .semantic_preservation_status == "green"
  and .semantic_preservation_audit_green == true
  and .state_ownership_green == true
  and .semantic_preservation_green == true
  and .canonical_identity_review.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_identity_review.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .control_ownership_boundary_review.host_executes_declared_mechanics == true
  and .control_ownership_boundary_review.host_decides_workflow == false
  and .decision_provenance_proof_complete == false
  and .carrier_split_publication_complete == false
  and (.deferred_issue_ids == ["TAS-188A", "TAS-189"])
  and ((.supporting_material_rows | length) == 7)
  and ((.state_class_rows | length) == 5)
  and ((.continuation_mechanism_rows | length) == 3)
  and ((.validation_rows | length) == 6)
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_canonical_route_semantic_preservation_audit_report.json >/dev/null

jq -e '
  .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .semantic_preservation_audit_green == true
  and .state_ownership_green == true
  and .control_ownership_rule_green == true
  and .semantic_preservation_green == true
  and .decision_provenance_proof_complete == false
  and .carrier_split_publication_complete == false
  and (.deferred_issue_ids == ["TAS-188A", "TAS-189"])
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_canonical_route_semantic_preservation_summary.json >/dev/null
