#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report

jq -e '
  .acceptance_status == "green"
  and .article_equivalence_green == true
  and .blocker_matrix_contract_green == true
  and .prerequisite_transformer_boundary_green == true
  and .blocker_matrix_article_equivalence_green == true
  and .public_claim_allowed == true
  and .closed_required_issue_count == 37
  and .passed_required_requirement_count == 40
  and .open_blocker_count == 0
  and ((.green_requirement_ids | index("TAS-158")) != null)
  and ((.green_requirement_ids | index("TAS-159")) != null)
  and ((.green_requirement_ids | index("TAS-160")) != null)
  and ((.green_requirement_ids | index("TAS-161")) != null)
  and ((.green_requirement_ids | index("TAS-162")) != null)
  and ((.green_requirement_ids | index("TAS-163")) != null)
  and ((.green_requirement_ids | index("TAS-176")) != null)
  and ((.green_requirement_ids | index("TAS-177")) != null)
  and ((.green_requirement_ids | index("TAS-178")) != null)
  and ((.green_requirement_ids | index("TAS-179")) != null)
  and ((.green_requirement_ids | index("TAS-179A")) != null)
  and ((.green_requirement_ids | index("TAS-180")) != null)
  and ((.green_requirement_ids | index("TAS-181")) != null)
  and ((.green_requirement_ids | index("TAS-182")) != null)
  and ((.green_requirement_ids | index("TAS-183")) != null)
  and ((.green_requirement_ids | index("TAS-184")) != null)
  and ((.green_requirement_ids | index("TAS-184A")) != null)
  and ((.green_requirement_ids | index("TAS-185")) != null)
  and ((.green_requirement_ids | index("TAS-185A")) != null)
  and ((.green_requirement_ids | index("TAS-186")) != null)
  and ((.green_requirement_ids | index("article_equivalence_blockers_closed")) != null)
  and ((.failed_requirement_ids | length) == 0)
  and ((.optional_open_issue_ids | index("TAS-R1")) != null)
  and ((.blocked_issue_ids | length) == 0)
  and ((.blocked_blocker_ids | length) == 0)
' fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json >/dev/null
