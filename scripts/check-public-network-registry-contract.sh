#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/public_network_registry_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/public_network_registry_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/public_network_registry_contract_v1.json"
cargo run -q -p psionic-train --bin public_network_registry_contract -- "${generated_path}" >/dev/null

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
    fail("public network registry contract check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.public_network_registry_contract.v1":
    fail("public network registry contract check: schema_version drifted")
if committed["contract_id"] != "psionic.public_network_registry_contract.v1":
    fail("public network registry contract check: contract_id drifted")
if committed["network_id"] != "psionic.decentralized_training.testnet.v1":
    fail("public network registry contract check: network_id drifted")
if committed["current_epoch_id"] != "network_epoch_00000123":
    fail("public network registry contract check: current_epoch_id drifted")
if len(committed["registry_records"]) != 4:
    fail("public network registry contract check: expected exactly four registry records")
if len(committed["discovery_examples"]) != 4:
    fail("public network registry contract check: expected exactly four discovery examples")
if len(committed["matchmaking_offers"]) != 3:
    fail("public network registry contract check: expected exactly three matchmaking offers")

query_ids = {query["query_id"] for query in committed["discovery_examples"]}
expected_query_ids = {
    "discover_public_miner_nodes",
    "discover_public_validator_nodes",
    "discover_checkpoint_authority_nodes",
    "discover_relay_nodes",
}
if query_ids != expected_query_ids:
    fail("public network registry contract check: discovery query set drifted")

relay_query = next(query for query in committed["discovery_examples"] if query["query_id"] == "discover_relay_nodes")
if relay_query["matched_registry_record_ids"] != ["google_l4_validator_node.registry"]:
    fail("public network registry contract check: relay discovery result drifted")

validator_offer = next(offer for offer in committed["matchmaking_offers"] if offer["offer_id"] == "validator_quorum_offer_v1")
if validator_offer["selected_registry_record_ids"] != [
    "google_l4_validator_node.registry",
    "local_mlx_mac_workstation.registry",
]:
    fail("public network registry contract check: validator quorum offer drifted")

if committed["authority_paths"]["check_script_path"] != "scripts/check-public-network-registry-contract.sh":
    fail("public network registry contract check: check_script_path drifted")
if committed["authority_paths"]["reference_doc_path"] != "docs/PUBLIC_NETWORK_REGISTRY_REFERENCE.md":
    fail("public network registry contract check: reference_doc_path drifted")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "registry_record_ids": [record["registry_record_id"] for record in committed["registry_records"]],
    "query_ids": [query["query_id"] for query in committed["discovery_examples"]],
}
print(json.dumps(summary, indent=2))
PY
