#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/operator_bootstrap_package_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/operator_bootstrap_package_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/operator_bootstrap_package_contract_v1.json"
cargo run -q -p psionic-train --bin operator_bootstrap_package_contract -- "${generated_path}" >/dev/null

python3 - "${contract_path}" "${generated_path}" <<'PY'
import json
import sys
from pathlib import Path

committed = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
generated = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

if committed != generated:
    fail("operator bootstrap package contract check: committed fixture drifted from generator output")
if len(committed["packages"]) != 2:
    fail("operator bootstrap package contract check: expected two packages")
if len(committed["preflight_checks"]) != 4:
    fail("operator bootstrap package contract check: expected four preflight checks")
if len(committed["bootstrap_kits"]) != 2:
    fail("operator bootstrap package contract check: expected two bootstrap kits")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
}
print(json.dumps(summary, indent=2))
PY
