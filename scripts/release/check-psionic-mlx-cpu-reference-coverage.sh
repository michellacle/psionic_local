#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

cargo run --quiet -p psionic-array --example mlx_cpu_reference_coverage_report -- "$@"
