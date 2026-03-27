#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/fraud_quarantine_slashing_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/fraud_quarantine_slashing_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/fraud_quarantine_slashing_contract_v1.json"
cargo run -q -p psionic-train --bin fraud_quarantine_slashing_contract -- "${generated_path}" >/dev/null

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
    fail("fraud quarantine slashing contract check: committed fixture drifted from generator output")
if len(committed["fraud_signals"]) != 4:
    fail("fraud quarantine slashing contract check: expected four fraud signals")
if len(committed["quarantine_decisions"]) != 2:
    fail("fraud quarantine slashing contract check: expected two quarantine decisions")
if len(committed["slashing_decisions"]) != 1:
    fail("fraud quarantine slashing contract check: expected one slashing decision")
if len(committed["appeal_windows"]) != 1:
    fail("fraud quarantine slashing contract check: expected one appeal window")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "slashing_decision_id": committed["slashing_decisions"][0]["decision_id"],
}
print(json.dumps(summary, indent=2))
PY
