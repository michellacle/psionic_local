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

campaign_id=""
frozen_config_path=""
output_path=""
run_evidence_paths=()
promotion_receipt_paths=()

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-build-record-candidate-campaign.sh \
  --campaign-id <id> \
  --frozen-config <path> \
  --output <path> \
  --run-evidence <path> --promotion-receipt <path> \
  [--run-evidence <path> --promotion-receipt <path> ...]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-id)
      campaign_id="$2"
      shift 2
      ;;
    --frozen-config)
      frozen_config_path="$2"
      shift 2
      ;;
    --output)
      output_path="$2"
      shift 2
      ;;
    --run-evidence)
      run_evidence_paths+=("$2")
      shift 2
      ;;
    --promotion-receipt)
      promotion_receipt_paths+=("$2")
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

if [[ -z "${campaign_id}" || -z "${frozen_config_path}" || -z "${output_path}" ]]; then
  echo "error: --campaign-id, --frozen-config, and --output are required" >&2
  usage
  exit 1
fi

if [[ ${#run_evidence_paths[@]} -eq 0 || ${#run_evidence_paths[@]} -ne ${#promotion_receipt_paths[@]} ]]; then
  echo "error: supply one or more matched --run-evidence / --promotion-receipt pairs" >&2
  usage
  exit 1
fi

args=(
  "${campaign_id}"
  "${frozen_config_path}"
  "${output_path}"
)

for idx in "${!run_evidence_paths[@]}"; do
  args+=("${run_evidence_paths[$idx]}" "${promotion_receipt_paths[$idx]}")
done

cargo run -q -p psionic-train \
  --manifest-path "${repo_root}/crates/psionic-train/Cargo.toml" \
  --example parameter_golf_record_candidate_campaign_report -- \
  "${args[@]}"
