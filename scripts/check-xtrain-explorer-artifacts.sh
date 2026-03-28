#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
snapshot_path="${repo_root}/fixtures/training/xtrain_explorer_snapshot_v1.json"
index_path="${repo_root}/fixtures/training/xtrain_explorer_index_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/xtrain_explorer_artifacts.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_snapshot="${tmpdir}/xtrain_explorer_snapshot_v1.json"
generated_index="${tmpdir}/xtrain_explorer_index_v1.json"
cargo run -q -p psionic-train --bin xtrain_explorer_artifacts -- \
  --snapshot-output "${generated_snapshot}" \
  --index-output "${generated_index}" >/dev/null

python3 - "${snapshot_path}" "${generated_snapshot}" "${index_path}" "${generated_index}" <<'PY'
import json
import sys
from pathlib import Path

committed_snapshot = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
generated_snapshot = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
committed_index = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
generated_index = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

if committed_snapshot != generated_snapshot:
    fail("xtrain explorer artifact check: committed snapshot fixture drifted from generator output")
if committed_index != generated_index:
    fail("xtrain explorer artifact check: committed index fixture drifted from generator output")
if len(committed_snapshot["participants"]) != 4:
    fail("xtrain explorer artifact check: expected four participants")
if len(committed_snapshot["participant_edges"]) != 4:
    fail("xtrain explorer artifact check: expected four participant edges")
if len(committed_snapshot["events"]) != 4:
    fail("xtrain explorer artifact check: expected four explorer events")
if len(committed_snapshot["run_surface_links"]) != 1:
    fail("xtrain explorer artifact check: expected one sibling run-surface link")
if len(committed_index["entries"]) != 1:
    fail("xtrain explorer artifact check: expected one explorer index entry")

summary = {
    "verdict": "verified",
    "snapshot_id": committed_snapshot["snapshot_id"],
    "snapshot_digest": committed_snapshot["snapshot_digest"],
    "index_id": committed_index["index_id"],
    "index_digest": committed_index["index_digest"],
}
print(json.dumps(summary, indent=2))
PY
