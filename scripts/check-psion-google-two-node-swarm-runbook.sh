#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
runbook_path="${repo_root}/docs/PSION_GOOGLE_TWO_NODE_SWARM_RUNBOOK.md"

python3 - "${repo_root}" "${runbook_path}" <<'PY'
import re
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
runbook_path = Path(sys.argv[2])
text = runbook_path.read_text(encoding="utf-8")


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


if "# Psion Google Two-Node Swarm Runbook" not in text:
    fail("google two-node swarm runbook check: title drifted")

required_literals = [
    "fixtures/psion/google/psion_google_two_node_swarm_contract_v1.json",
    "fixtures/psion/google/psion_google_two_node_swarm_launch_profiles_v1.json",
    "fixtures/psion/google/psion_google_two_node_swarm_network_posture_v1.json",
    "fixtures/psion/google/psion_google_two_node_swarm_identity_profile_v1.json",
    "fixtures/psion/google/psion_google_two_node_swarm_operator_preflight_policy_v1.json",
    "fixtures/psion/google/psion_google_two_node_swarm_impairment_policy_v1.json",
    "scripts/check-psion-google-two-node-swarm-contract.sh",
    "scripts/psion-google-operator-preflight-two-node-swarm.sh",
    "scripts/psion-google-launch-two-node-swarm.sh",
    "scripts/psion-google-two-node-swarm-startup.sh",
    "scripts/psion-google-two-node-swarm-impair.sh",
    "scripts/psion-google-finalize-two-node-swarm-run.sh",
    "scripts/check-psion-google-two-node-swarm-evidence-bundle.sh",
    "scripts/psion-google-delete-two-node-swarm.sh",
    "configured_peer_launch_failure",
    "cluster_membership_failure",
    "network_impairment_gate_failure",
    "contributor_execution_failure",
    "validator_refusal",
    "aggregation_failure",
    "bounded_success",
    "trusted-cluster full-model Google training claim",
    "no public or wider-network discovery claim",
]
for literal in required_literals:
    if literal not in text:
        fail(f"google two-node swarm runbook check: missing required literal `{literal}`")

referenced_paths = set(
    re.findall(r"(?:fixtures|scripts|docs)/[A-Za-z0-9_./-]+", text)
)
missing_paths = sorted(
    str(path) for path in referenced_paths if not (repo_root / path).exists()
)
if missing_paths:
    fail(
        "google two-node swarm runbook check: referenced repo paths missing: "
        + ", ".join(missing_paths)
    )

required_sections = [
    "## Operator Preconditions",
    "## Clean Baseline Launch",
    "## Impaired Rerun",
    "## Finalization",
    "## Result Classifications",
    "## Teardown",
]
for heading in required_sections:
    if heading not in text:
        fail(f"google two-node swarm runbook check: missing section `{heading}`")

summary = {
    "verdict": "verified",
    "runbook_path": str(runbook_path.relative_to(repo_root)),
    "referenced_repo_path_count": len(referenced_paths),
}
print(summary)
PY
