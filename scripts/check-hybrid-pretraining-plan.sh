#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/hybrid_pretraining_plan_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/hybrid_pretraining_plan.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/hybrid_pretraining_plan_v1.json"
cargo run -q -p psionic-train --bin hybrid_pretraining_plan -- "${generated_path}" >/dev/null

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
    fail("hybrid pretraining plan check: committed fixture drifted from generator output")

if fixture["schema_version"] != "psionic.hybrid_pretraining_plan.v1":
    fail("hybrid pretraining plan check: schema version drifted")
if fixture["dataset_family_id"] != "psion.curated_pretrain.dataset_family.v1":
    fail("hybrid pretraining plan check: dataset family drifted")
if fixture["checkpoint_family"] != "psion.cross_provider.pretrain.v1":
    fail("hybrid pretraining plan check: checkpoint family drifted")
if len(fixture["dense_rank_assignments"]) != 8:
    fail("hybrid pretraining plan check: dense rank count drifted")
if len(fixture["contributor_window_assignments"]) < 2:
    fail("hybrid pretraining plan check: contributor window count drifted")

lineage_classes = {binding["execution_class"] for binding in fixture["lineage_bindings"]}
required_classes = {
    "dense_full_model_rank",
    "validated_contributor_window",
    "validator",
    "eval_worker",
    "checkpoint_writer",
}
if not required_classes.issubset(lineage_classes):
    fail("hybrid pretraining plan check: lineage bindings lost one or more work classes")

summary = {
    "verdict": "verified",
    "dense_rank_count": len(fixture["dense_rank_assignments"]),
    "contributor_window_count": len(fixture["contributor_window_assignments"]),
    "lineage_class_count": len(lineage_classes),
}
print(json.dumps(summary, indent=2))
PY
