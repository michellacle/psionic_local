#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_minimal_universal_substrate_runtime_report
cargo run -p psionic-eval --example tassadar_minimal_universal_substrate_acceptance_gate_report

jq -e '
  .overall_green == true
  and .acceptance_status == "green"
  and (.green_requirement_ids | length) == 8
  and (.failed_requirement_ids | length) == 0
' fixtures/tassadar/reports/tassadar_minimal_universal_substrate_acceptance_gate_report.json >/dev/null
