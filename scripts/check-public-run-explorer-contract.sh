#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/public_run_explorer_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/public_run_explorer_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/public_run_explorer_contract_v1.json"
cargo run -q -p psionic-train --bin public_run_explorer_contract -- "${generated_path}" >/dev/null

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
    fail("public run explorer contract check: committed fixture drifted from generator output")
if len(committed["panes"]) != 6:
    fail("public run explorer contract check: expected six panes")
if len(committed["score_rows"]) != 4:
    fail("public run explorer contract check: expected four score rows")
if len(committed["stale_data_policies"]) != 6:
    fail("public run explorer contract check: expected six stale-data policies")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
}
print(json.dumps(summary, indent=2))
PY
