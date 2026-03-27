#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/settlement_publication_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/settlement_publication_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/settlement_publication_contract_v1.json"
cargo run -q -p psionic-train --bin settlement_publication_contract -- "${generated_path}" >/dev/null

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
    fail("settlement publication contract check: committed fixture drifted from generator output")
if len(committed["validator_weight_publications"]) != 2:
    fail("settlement publication contract check: expected two validator weight publications")
if len(committed["settlement_records"]) != 1:
    fail("settlement publication contract check: expected one settlement record")
if len(committed["payout_exports"]) != 3:
    fail("settlement publication contract check: expected three payout exports")
if len(committed["refusals"]) != 1:
    fail("settlement publication contract check: expected one refusal")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "settlement_record_id": committed["settlement_records"][0]["record_id"],
}
print(json.dumps(summary, indent=2))
PY
