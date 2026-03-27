#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture="$repo_root/fixtures/parameter_golf/reports/parameter_golf_homegolf_mixed_hardware_manifest.json"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
generated="$tmpdir/parameter_golf_homegolf_mixed_hardware_manifest.json"

cargo run -q -p psionic-train --bin parameter_golf_homegolf_mixed_hardware_manifest -- "$generated"
cmp -s "$fixture" "$generated"
echo "HOMEGOLF mixed hardware manifest matches fixture"
