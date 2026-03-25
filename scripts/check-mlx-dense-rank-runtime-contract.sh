#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/mlx_dense_rank_runtime_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/mlx_dense_rank_runtime.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/mlx_dense_rank_runtime_contract_v1.json"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-}" cargo run -q -p psionic-train --bin mlx_dense_rank_runtime_contract -- "${generated_path}" >/dev/null

python3 - "${fixture_path}" "${generated_path}" <<'PY'
import json
import sys
from pathlib import Path

fixture_path = Path(sys.argv[1])
generated_path = Path(sys.argv[2])

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

committed = json.loads(fixture_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))
if committed != generated:
    fail("MLX dense-rank runtime check: committed fixture drifted from generator output")

runtime = committed["runtime"]
if runtime["requested_backend"] != "mlx_metal" or runtime["world_size"] != 1:
    fail("MLX dense-rank runtime check: runtime identity drifted from the bounded MLX dense path")

if committed["source_id"] != "local_mlx_mac_workstation":
    fail("MLX dense-rank runtime check: source_id drifted")

if committed["execution_receipt"]["train_step"]["gradient_sync_ms"] != 0:
    fail("MLX dense-rank runtime check: single-rank MLX path should not claim gradient sync time")

unsupported = {surface["surface_kind"] for surface in committed["unsupported_surfaces"]}
expected = {
    "bf16_mixed_precision",
    "cross_host_collectives",
    "sharded_optimizer_state_exchange",
    "mixed_backend_dense_mesh",
}
if unsupported != expected:
    fail("MLX dense-rank runtime check: unsupported surface set drifted")

required_roles = set(committed["evidence_hook"]["required_artifact_roles"])
if required_roles != {
    "launch_contract",
    "dense_runtime_receipt",
    "checkpoint_manifest",
    "metric_log",
    "acceptance_audit",
}:
    fail("MLX dense-rank runtime check: evidence hook roles drifted")

summary = {
    "verdict": "verified",
    "runtime_family_id": runtime["runtime_family_id"],
    "source_id": committed["source_id"],
    "metric_event_count": len(committed["metric_events"]),
    "unsupported_surface_count": len(committed["unsupported_surfaces"]),
}
print(json.dumps(summary, indent=2))
PY
