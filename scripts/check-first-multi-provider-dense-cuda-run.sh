#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/first_multi_provider_dense_cuda_run_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/first_multi_provider_dense_cuda_run.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/first_multi_provider_dense_cuda_run_v1.json"
cargo run -q -p psionic-train --bin first_multi_provider_dense_cuda_run -- "${generated_path}" >/dev/null

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
    fail("first multi-provider dense CUDA run check: committed fixture drifted from generator output")

if fixture["schema_version"] != "psionic.first_multi_provider_dense_cuda_run.v1":
    fail("first multi-provider dense CUDA run check: schema version drifted")

if len(fixture["phases"]) != 4:
    fail("first multi-provider dense CUDA run check: phase count drifted")
if len(fixture["recovery_events"]) != 1:
    fail("first multi-provider dense CUDA run check: recovery-event count drifted")

if not any(phase["active_provider_ids"] == ["google", "runpod"] for phase in fixture["phases"]):
    fail("first multi-provider dense CUDA run check: mixed-provider dense phase disappeared")

summary = {
    "verdict": "verified",
    "phase_count": len(fixture["phases"]),
    "recovery_event_count": len(fixture["recovery_events"]),
    "final_disposition": fixture["final_disposition"],
    "run_id": fixture["run_id"],
}
print(json.dumps(summary, indent=2))
PY
