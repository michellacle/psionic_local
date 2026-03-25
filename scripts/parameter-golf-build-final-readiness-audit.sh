#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

if ! command -v cargo >/dev/null 2>&1; then
  if [[ -d "${HOME}/.cargo/bin" ]]; then
    export PATH="${HOME}/.cargo/bin:${PATH}"
  fi
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo is required but was not found in PATH or \${HOME}/.cargo/bin" >&2
  exit 1
fi

campaign_report_path=""
output_path=""
final_pr_bundle_path="${repo_root}/fixtures/parameter_golf/reports/parameter_golf_final_pr_bundle.json"
local_clone_dry_run_path="${repo_root}/fixtures/parameter_golf/reports/parameter_golf_local_clone_dry_run.json"

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-build-final-readiness-audit.sh \
  --campaign-report <path> \
  --output <path> \
  [--final-pr-bundle <path>] \
  [--local-clone-dry-run <path>]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-report)
      campaign_report_path="$2"
      shift 2
      ;;
    --output)
      output_path="$2"
      shift 2
      ;;
    --final-pr-bundle)
      final_pr_bundle_path="$2"
      shift 2
      ;;
    --local-clone-dry-run)
      local_clone_dry_run_path="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${campaign_report_path}" || -z "${output_path}" ]]; then
  echo "error: --campaign-report and --output are required" >&2
  usage
  exit 1
fi

cargo run -q -p psionic-train \
  --manifest-path "${repo_root}/crates/psionic-train/Cargo.toml" \
  --example parameter_golf_final_readiness_audit -- \
  "${campaign_report_path}" \
  "${final_pr_bundle_path}" \
  "${local_clone_dry_run_path}" \
  "${output_path}"
