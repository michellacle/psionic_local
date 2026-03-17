#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

cd "${repo_root}"

input_path="${1:-fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0}"
if [[ -d "${input_path}" ]]; then
  report_path="${input_path%/}/promotion_gate_report.json"
else
  report_path="${input_path}"
fi

rustflags="${RUSTFLAGS:-}"
if [[ -n "${rustflags}" ]]; then
  rustflags="${rustflags} -Awarnings"
else
  rustflags="-Awarnings"
fi

RUSTFLAGS="${rustflags}" cargo run --quiet -p psionic-train --example tassadar_sudoku_9x9_promotion_gate_check -- "${report_path}"
