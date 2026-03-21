#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_post_article_carrier_split_contract_report
cargo run -p psionic-research --example tassadar_post_article_carrier_split_contract_summary
cargo test -p psionic-provider carrier_split_receipt_projects_summary -- --nocapture

jq -e '
  .carrier_split_status == "green"
  and .carrier_split_publication_complete == true
  and .carrier_collapse_refused == true
  and .reserved_capability_plane_explicit == true
  and .decision_provenance_proof_complete == true
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .carrier_topology == "explicit_split_across_direct_and_resumable_lanes"
  and .direct_carrier_id == "tassadar.post_article_universality_bridge.direct_article_equivalent_carrier.v1"
  and .resumable_carrier_id == "tassadar.post_article_universality_bridge.resumable_universality_carrier.v1"
  and .reserved_capability_plane_id == "tassadar.post_article_universality_bridge.reserved_capability_plane.v1"
  and ((.supporting_material_rows | length) == 8)
  and ((.primary_carrier_rows | length) == 2)
  and ((.claim_class_binding_rows | length) == 7)
  and ((.validation_rows | length) == 7)
  and (.deferred_issue_ids == ["TAS-190"])
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_carrier_split_contract_report.json >/dev/null

jq -e '
  .carrier_split_status == "green"
  and .carrier_split_publication_complete == true
  and .carrier_collapse_refused == true
  and .reserved_capability_plane_explicit == true
  and .decision_provenance_proof_complete == true
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .direct_carrier_id == "tassadar.post_article_universality_bridge.direct_article_equivalent_carrier.v1"
  and .resumable_carrier_id == "tassadar.post_article_universality_bridge.resumable_universality_carrier.v1"
  and .reserved_capability_plane_id == "tassadar.post_article_universality_bridge.reserved_capability_plane.v1"
  and (.deferred_issue_ids == ["TAS-190"])
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_carrier_split_contract_summary.json >/dev/null
