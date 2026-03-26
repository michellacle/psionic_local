#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/signed_node_identity_contract_set_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/signed_node_identity_contract_set.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/signed_node_identity_contract_set_v1.json"
cargo run -q -p psionic-train --bin signed_node_identity_contract_set -- "${generated_path}" >/dev/null

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
    fail("signed node identity contract set check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.signed_node_identity_contract_set.v1":
    fail("signed node identity contract set check: schema_version drifted")
if committed["contract_id"] != "psionic.signed_node_identity_contract_set.v1":
    fail("signed node identity contract set check: contract_id drifted")
if committed["network_id"] != "psionic.decentralized_training.testnet.v1":
    fail("signed node identity contract set check: network_id drifted")
if committed["governance_revision_id"] != "psionic.decentralized_training_governance.v1":
    fail("signed node identity contract set check: governance_revision_id drifted")
if committed["revocation_authority"]["policy_id"] != "psionic.decentralized_training_identity_revocation.v1":
    fail("signed node identity contract set check: revocation policy drifted")
if len(committed["identities"]) != 4:
    fail("signed node identity contract set check: expected exactly four identities")

source_ids = {identity["source_id"] for identity in committed["identities"]}
expected_source_ids = {
    "google_l4_validator_node",
    "runpod_8xh100_dense_node",
    "local_rtx4080_workstation",
    "local_mlx_mac_workstation",
}
if source_ids != expected_source_ids:
    fail("signed node identity contract set check: source set drifted")

google = next(identity for identity in committed["identities"] if identity["source_id"] == "google_l4_validator_node")
runpod = next(identity for identity in committed["identities"] if identity["source_id"] == "runpod_8xh100_dense_node")
if "relay" not in google["admitted_role_classes"]:
    fail("signed node identity contract set check: google relay admission drifted")
if "public_miner" in runpod["admitted_role_classes"]:
    fail("signed node identity contract set check: runpod unexpectedly admits public_miner")
if google["capability_signature"]["signature_scheme"] != "ed25519_detached":
    fail("signed node identity contract set check: signature scheme drifted")
if committed["authority_paths"]["check_script_path"] != "scripts/check-signed-node-identity-contract-set.sh":
    fail("signed node identity contract set check: check_script_path drifted")
if committed["authority_paths"]["reference_doc_path"] != "docs/SIGNED_NODE_IDENTITY_CONTRACT_REFERENCE.md":
    fail("signed node identity contract set check: reference_doc_path drifted")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "identity_ids": [identity["node_identity_id"] for identity in committed["identities"]],
    "relay_nodes": [
        identity["source_id"]
        for identity in committed["identities"]
        if "relay" in identity["admitted_role_classes"]
    ],
}
print(json.dumps(summary, indent=2))
PY
