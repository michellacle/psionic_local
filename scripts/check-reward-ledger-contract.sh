#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/reward_ledger_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/reward_ledger_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/reward_ledger_contract_v1.json"
cargo run -q -p psionic-train --bin reward_ledger_contract -- "${generated_path}" >/dev/null

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
    fail("reward ledger contract check: committed fixture drifted from generator output")
if len(committed["contribution_entries"]) != 5:
    fail("reward ledger contract check: expected five contribution entries")
if len(committed["penalty_entries"]) != 1:
    fail("reward ledger contract check: expected one penalty entry")
if len(committed["final_allocations"]) != 4:
    fail("reward ledger contract check: expected four final allocations")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "accounting_period_id": committed["accounting_period"]["accounting_period_id"],
}
print(json.dumps(summary, indent=2))
PY
