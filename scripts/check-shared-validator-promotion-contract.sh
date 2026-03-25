#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/shared_validator_promotion_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/shared_validator_promotion_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/shared_validator_promotion_contract_v1.json"
cargo run -q -p psionic-train --bin shared_validator_promotion_contract -- "${generated_path}" >/dev/null

python3 - "${fixture_path}" "${generated_path}" <<'PY'
import json
import sys
from pathlib import Path

fixture_path = Path(sys.argv[1])
generated_path = Path(sys.argv[2])

fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

if fixture != generated:
    fail("shared validator/promotion contract check: committed fixture drifted from generator output")

if fixture["contract_id"] != "psionic.shared_validator_promotion_contract.v1":
    fail("shared validator/promotion contract check: contract id drifted")

dispositions = set(fixture["admitted_validator_dispositions"])
if dispositions != {"accepted", "quarantined", "rejected", "replay_required"}:
    fail(f"shared validator/promotion contract check: disposition set drifted: {sorted(dispositions)}")

promotion_outcomes = set(fixture["admitted_promotion_outcomes"])
if promotion_outcomes != {"promoted_revision", "held_no_promotion", "refused_promotion"}:
    fail(f"shared validator/promotion contract check: promotion outcome set drifted: {sorted(promotion_outcomes)}")

summary = {
    "verdict": "verified",
    "execution_class_policy_count": len(fixture["execution_class_policies"]),
    "validator_disposition_count": len(dispositions),
    "promotion_outcome_count": len(promotion_outcomes),
}
print(json.dumps(summary, indent=2))
PY
