#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_rust_only_article_closeout_audit_report

jq -e '
  .green == true
  and .harness_green == true
  and .acceptance_gate_green == true
  and .direct_model_weight_proof_green == true
  and .cpu_reproducibility_green == true
' fixtures/tassadar/reports/tassadar_rust_only_article_closeout_audit_report.json >/dev/null
