#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-serve --example tassadar_rust_only_article_acceptance_gate_v2

jq -e '
  .green == true
  and .prerequisite_count == 8
  and .passed_prerequisite_count == 8
  and (.failed_prerequisite_ids | length) == 0
' fixtures/tassadar/reports/tassadar_rust_only_article_acceptance_gate_v2.json >/dev/null
