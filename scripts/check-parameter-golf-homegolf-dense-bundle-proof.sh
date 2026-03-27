#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_path="${repo_root}/fixtures/parameter_golf/reports/parameter_golf_homegolf_dense_bundle_proof.json"
tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/parameter_golf_homegolf_dense_bundle_proof.XXXXXX")"

cargo run -q -p psionic-serve --example parameter_golf_homegolf_dense_bundle_proof -- "${tmpdir}" >/dev/null

python3 - "${fixture_path}" "${tmpdir}/parameter_golf_homegolf_dense_bundle_proof.json" <<'PY'
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
    fail("parameter golf HOMEGOLF dense bundle proof check: committed fixture drifted from generator output")

if fixture["track_id"] != "parameter_golf.home_cluster_compatible_10min.v1":
    fail("parameter golf HOMEGOLF dense bundle proof check: track_id drifted")
if fixture["baseline_model_id"] != "parameter-golf-sp1024-9x512":
    fail("parameter golf HOMEGOLF dense bundle proof check: baseline model drifted")
if fixture["profile_id"] != "psion_small_decoder_pgolf_core_v0":
    fail("parameter golf HOMEGOLF dense bundle proof check: promoted profile drifted")
if not fixture["descriptor_digest"]:
    fail("parameter golf HOMEGOLF dense bundle proof check: descriptor digest disappeared")
if not fixture["tokenizer_digest"]:
    fail("parameter golf HOMEGOLF dense bundle proof check: tokenizer digest disappeared")
if not fixture["direct_and_served_match"]:
    fail("parameter golf HOMEGOLF dense bundle proof check: direct/served parity drifted")
if fixture["final_validation_bits_per_byte"] <= 0:
    fail("parameter golf HOMEGOLF dense bundle proof check: validation metric drifted")
PY
