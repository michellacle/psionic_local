#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_path="${repo_root}/fixtures/parameter_golf/reports/parameter_golf_homegolf_dense_baseline_surface.json"
tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/parameter_golf_homegolf_dense_baseline_surface.XXXXXX")"
generated_path="${tmpdir}/parameter_golf_homegolf_dense_baseline_surface.json"

cargo run -q -p psionic-train --bin parameter_golf_homegolf_dense_baseline_surface -- "${generated_path}" >/dev/null

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
    raise SystemExit(1)

if fixture != generated:
    fail("parameter golf HOMEGOLF dense baseline check: committed fixture drifted from generator output")

if fixture["track_id"] != "parameter_golf.home_cluster_compatible_10min.v1":
    fail("parameter golf HOMEGOLF dense baseline check: track_id drifted")
if fixture["baseline_model_id"] != "parameter-golf-sp1024-9x512":
    fail("parameter golf HOMEGOLF dense baseline check: baseline_model_id drifted")
if fixture["baseline_revision"] != "public-2026-03-18":
    fail("parameter golf HOMEGOLF dense baseline check: baseline_revision drifted")
if fixture["baseline_config"]["vocab_size"] != 1024 or fixture["baseline_config"]["num_layers"] != 9:
    fail("parameter golf HOMEGOLF dense baseline check: exact baseline config drifted")
if fixture["baseline_config"]["model_dim"] != 512 or fixture["baseline_config"]["num_heads"] != 8:
    fail("parameter golf HOMEGOLF dense baseline check: exact baseline width/head config drifted")
if fixture["baseline_config"]["num_kv_heads"] != 4 or fixture["baseline_config"]["mlp_mult"] != 2:
    fail("parameter golf HOMEGOLF dense baseline check: exact baseline kv/mlp config drifted")
if fixture["baseline_config"]["tie_embeddings"] is not True:
    fail("parameter golf HOMEGOLF dense baseline check: tied-embedding contract drifted")
if fixture["final_metric"]["val_bpb"] <= 0 or fixture["compressed_model_bytes"] <= 0:
    fail("parameter golf HOMEGOLF dense baseline check: retained metric/artifact fields drifted")
PY
