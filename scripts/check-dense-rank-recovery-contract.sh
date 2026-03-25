#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/dense_rank_recovery_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/dense_rank_recovery_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/dense_rank_recovery_contract_v1.json"
cargo run -q -p psionic-train --bin dense_rank_recovery_contract -- "${generated_path}" >/dev/null

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
    fail("dense-rank recovery contract check: committed fixture drifted from generator output")

if fixture["schema_version"] != "psionic.dense_rank_recovery_contract.v1":
    fail("dense-rank recovery contract check: schema version drifted")

if len(fixture["scenarios"]) != 4:
    fail("dense-rank recovery contract check: scenario count drifted")

recovered = [scenario for scenario in fixture["scenarios"] if scenario["disposition"] == "recovered"]
refused = [scenario for scenario in fixture["scenarios"] if scenario["disposition"] == "refused"]
if len(recovered) != 3 or len(refused) != 1:
    fail("dense-rank recovery contract check: recovered/refused split drifted")

kinds = {scenario["failure_kind"] for scenario in fixture["scenarios"]}
if kinds != {"preemption", "node_loss", "provider_loss"}:
    fail("dense-rank recovery contract check: failure-kind coverage drifted")

provider_loss = next(
    scenario for scenario in fixture["scenarios"]
    if scenario["scenario_id"] == "dense_rank.provider_loss.rank2.cross_provider_replace"
)
if provider_loss["replacement_provider_id"] == provider_loss["departing_provider_id"]:
    fail("dense-rank recovery contract check: provider-loss replacement lost cross-provider swap")

summary = {
    "verdict": "verified",
    "scenario_count": len(fixture["scenarios"]),
    "recovered_count": len(recovered),
    "refused_count": len(refused),
    "checkpoint_digest": fixture["distributed_checkpoint_contract_digest"],
}
print(json.dumps(summary, indent=2))
PY
