#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_path="${repo_root}/fixtures/parameter_golf/reports/parameter_golf_homegolf_track_contract.json"
tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/parameter_golf_homegolf_track_contract.XXXXXX")"
generated_path="${tmpdir}/parameter_golf_homegolf_track_contract.json"

cargo run -q -p psionic-train --bin parameter_golf_homegolf_track_contract -- "${generated_path}" >/dev/null

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
    fail("parameter golf HOMEGOLF track contract check: committed fixture drifted from generator output")

if fixture["track_id"] != "parameter_golf.home_cluster_compatible_10min.v1":
    fail("parameter golf HOMEGOLF track contract check: track_id drifted")
if fixture["strict_profile_id"] != "parameter_golf_challenge_sp1024_v0":
    fail("parameter golf HOMEGOLF track contract check: strict profile id drifted")
if fixture["artifact_cap_bytes"] != 16000000 or fixture["wallclock_cap_seconds"] != 600:
    fail("parameter golf HOMEGOLF track contract check: contest cap values drifted")
if not fixture["exact_fineweb_sp1024_identity_required"] or not fixture["exact_contest_bpb_accounting_required"]:
    fail("parameter golf HOMEGOLF track contract check: strict scorepath flags drifted")
if "dense_trainer_entrypoint" not in {surface["surface_id"] for surface in fixture["required_surfaces"]}:
    fail("parameter golf HOMEGOLF track contract check: dense trainer surface disappeared")
PY
