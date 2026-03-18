#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

cd "${repo_root}"

rustflags="${RUSTFLAGS:-}"
if [[ -n "${rustflags}" ]]; then
  rustflags="${rustflags} -Awarnings"
else
  rustflags="-Awarnings"
fi

RUSTFLAGS="${rustflags}" cargo test --quiet -p psionic-research promotion_policy_report_matches_committed_truth -- --nocapture
