#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-research --example tassadar_learned_horizon_policy_report -- "$@"
cargo run -p psionic-research --example tassadar_acceptance_report -- "$@"
