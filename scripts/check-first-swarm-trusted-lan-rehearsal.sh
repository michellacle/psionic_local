#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
report_path=""

usage() {
  cat <<'EOF' >&2
Usage: check-first-swarm-trusted-lan-rehearsal.sh [--report <path>]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report)
      report_path="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cleanup_report=0
if [[ -z "${report_path}" ]]; then
  report_path="$(mktemp "${TMPDIR:-/tmp}/first_swarm_trusted_lan_rehearsal.XXXXXX.json")"
  cleanup_report=1
fi

cleanup() {
  if [[ "${cleanup_report}" -eq 1 ]]; then
    rm -f -- "${report_path}"
  fi
}
trap cleanup EXIT

cargo run -q -p psionic-train --bin first_swarm_trusted_lan_rehearsal_report -- "${report_path}"

python3 - "${report_path}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


if report["schema_version"] != "swarm.first_trusted_lan_rehearsal_report.v1":
    fail("first swarm rehearsal check: schema version drifted")
if report["recommendation"] != "no_go":
    fail("first swarm rehearsal check: recommendation must stay no_go until a live two-node path exists")
if "not a live two-node swarm evidence bundle" not in report["claim_boundary"]:
    fail("first swarm rehearsal check: claim boundary drifted")
phases = report["phases"]
if len(phases) < 6:
    fail("first swarm rehearsal check: expected at least six rehearsal phases")
phase_ids = {phase["phase_id"] for phase in phases}
required_phase_ids = {
    "operator_bundle_materialization",
    "mac_bringup_validation",
    "linux_bringup_validation",
    "contributor_execution_projection",
    "artifact_upload_projection",
    "validator_aggregation_projection",
}
if phase_ids != required_phase_ids:
    fail("first swarm rehearsal check: phase set drifted")
if not any(phase["evidence_posture"] == "measured" for phase in phases):
    fail("first swarm rehearsal check: report lost measured phases")
if not any(phase["evidence_posture"] == "simulated" for phase in phases):
    fail("first swarm rehearsal check: report lost simulated phases")
if report["topology_used"]["cluster_namespace"] != "cluster.swarm.local.trusted_lan":
    fail("first swarm rehearsal check: topology namespace drifted")
if report["topology_used"]["coordinator_node_id"] != "swarm-mac-a":
    fail("first swarm rehearsal check: coordinator node id drifted")
if sorted(report["topology_used"]["contributor_node_ids"]) != ["swarm-linux-4080-a", "swarm-mac-a"]:
    fail("first swarm rehearsal check: contributor node set drifted")
operator_phase = next(phase for phase in phases if phase["phase_id"] == "operator_bundle_materialization")
if operator_phase["parallelization_posture"] != "serial":
    fail("first swarm rehearsal check: operator bundle materialization must stay serial")
if operator_phase["wallclock_ms"] <= 0:
    fail("first swarm rehearsal check: operator bundle materialization must have positive wallclock")
execution_phase = next(phase for phase in phases if phase["phase_id"] == "contributor_execution_projection")
if execution_phase["parallelization_posture"] != "parallelizable":
    fail("first swarm rehearsal check: contributor execution must stay parallelizable in the report")
if len(execution_phase["worker_timings"]) != 2:
    fail("first swarm rehearsal check: contributor execution phase must keep both backend-specific worker rows")
validator_phase = next(phase for phase in phases if phase["phase_id"] == "validator_aggregation_projection")
if validator_phase["parallelization_posture"] != "serial":
    fail("first swarm rehearsal check: validator/aggregation phase must stay serial")
if "validator_aggregation_projection" not in report["remaining_serial_phase_ids"]:
    fail("first swarm rehearsal check: validator/aggregation must remain in the serial phase list")
if len(report["remaining_blockers"]) < 3:
    fail("first swarm rehearsal check: report lost concrete remaining blockers")
bottlenecks = report["bottlenecks"]
if len(bottlenecks) < 3:
    fail("first swarm rehearsal check: expected at least three bottlenecks")
if not any(b["phase_id"] == "validator_aggregation_projection" and b["severity"] == "high" for b in bottlenecks):
    fail("first swarm rehearsal check: validator/aggregation bottleneck drifted")

summary = {
    "verdict": "verified",
    "report_digest": report["report_digest"],
    "recommendation": report["recommendation"],
    "topology_contract_digest": report["topology_used"]["topology_contract_digest"],
    "phase_count": len(phases),
    "serial_phases": report["remaining_serial_phase_ids"],
}
print(json.dumps(summary, indent=2))
PY
