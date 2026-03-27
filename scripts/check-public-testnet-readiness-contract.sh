#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/public_testnet_readiness_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/public_testnet_readiness_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/public_testnet_readiness_contract_v1.json"
cargo run -q -p psionic-train --bin public_testnet_readiness_contract -- "${generated_path}" >/dev/null

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
    fail("public testnet readiness contract check: committed fixture drifted from generator output")
if len(committed["candidates"]) != 5:
    fail("public testnet readiness contract check: expected five candidates")
if len(committed["compliance_receipts"]) != 8:
    fail("public testnet readiness contract check: expected eight compliance receipts")
if len(committed["graduation_decisions"]) != 5:
    fail("public testnet readiness contract check: expected five graduation decisions")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
}
print(json.dumps(summary, indent=2))
PY
