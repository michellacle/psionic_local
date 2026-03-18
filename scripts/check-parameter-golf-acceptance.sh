#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
input_path="${repo_root}/fixtures/parameter_golf/reports/parameter_golf_acceptance_report.json"
report_path=""
selected_categories=()

usage() {
    cat <<'EOF' >&2
Usage: scripts/check-parameter-golf-acceptance.sh [--input <path>] [--report <path>] [--only <category>]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            [[ $# -ge 2 ]] || {
                echo "missing path after --input" >&2
                usage
                exit 1
            }
            input_path="$2"
            shift 2
            ;;
        --report)
            [[ $# -ge 2 ]] || {
                echo "missing path after --report" >&2
                usage
                exit 1
            }
            report_path="$2"
            shift 2
            ;;
        --only)
            [[ $# -ge 2 ]] || {
                echo "missing category after --only" >&2
                usage
                exit 1
            }
            selected_categories+=("$2")
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

python_args=("$input_path" "$report_path")
if [[ ${#selected_categories[@]} -gt 0 ]]; then
    python_args+=("${selected_categories[@]}")
fi

python3 - "${python_args[@]}" <<'PY'
import json
import hashlib
import sys
from pathlib import Path

INPUT_PATH = Path(sys.argv[1])
REPORT_PATH = sys.argv[2]
SELECTED = sys.argv[3:]

ALLOWED_CATEGORIES = [
    "challenge-oracle-parity",
    "single-device-trainer-parity",
    "distributed-throughput-closure",
    "packaging-readiness",
    "record-track-readiness",
]
ALLOWED_STATUSES = {
    "implemented",
    "implemented_early",
    "partial",
    "partial_outside_psionic",
    "planned",
}
ALLOWED_CLAIMS = {
    "research",
    "non_record_submission",
    "record_candidate_blocked_on_accounting",
    "record_ready",
}


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


def stable_digest(payload: dict) -> str:
    material = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def validate_report(data: dict) -> None:
    required_top = [
        "schema_version",
        "schema_path",
        "runner",
        "report_posture",
        "claim_vocabulary_version",
        "current_claim_posture",
        "accounting_posture",
        "selected_categories",
        "categories",
        "report_digest",
    ]
    for key in required_top:
        if key not in data:
            fail(f"parameter golf acceptance error: missing top-level key `{key}`")

    if data["schema_version"] != 1:
        fail("parameter golf acceptance error: `schema_version` must be 1")
    if data["schema_path"] != "docs/parameter_golf_acceptance_report.schema.json":
        fail("parameter golf acceptance error: unexpected `schema_path`")
    if data["runner"] != "scripts/check-parameter-golf-acceptance.sh":
        fail("parameter golf acceptance error: unexpected `runner`")
    if data["report_posture"] != "tracking_only":
        fail("parameter golf acceptance error: `report_posture` must be `tracking_only`")
    if data["current_claim_posture"] not in ALLOWED_CLAIMS:
        fail("parameter golf acceptance error: unknown `current_claim_posture`")

    accounting = data["accounting_posture"]
    required_accounting = [
        "artifact_cap_bytes",
        "artifact_cap_unit",
        "counted_components",
        "rust_runtime_posture",
        "wrapper_posture",
        "build_dependency_posture",
        "offline_evaluation_required",
    ]
    for key in required_accounting:
        if key not in accounting:
            fail(f"parameter golf acceptance error: missing accounting key `{key}`")
    if accounting["artifact_cap_bytes"] != 16000000:
        fail("parameter golf acceptance error: artifact cap must be 16000000 bytes")
    if accounting["artifact_cap_unit"] != "decimal_bytes":
        fail("parameter golf acceptance error: artifact cap unit must be `decimal_bytes`")
    if accounting["rust_runtime_posture"] != "must_count_when_shipped":
        fail("parameter golf acceptance error: unexpected `rust_runtime_posture`")
    if accounting["wrapper_posture"] != "allowed_only_when_counted_and_challenge_compatible":
        fail("parameter golf acceptance error: unexpected `wrapper_posture`")
    if accounting["build_dependency_posture"] != "not_exempt_by_default":
        fail("parameter golf acceptance error: unexpected `build_dependency_posture`")
    if accounting["offline_evaluation_required"] is not True:
        fail("parameter golf acceptance error: `offline_evaluation_required` must be true")

    selected_categories = data["selected_categories"]
    if not isinstance(selected_categories, list) or not selected_categories:
        fail("parameter golf acceptance error: `selected_categories` must be a non-empty array")
    if any(category not in ALLOWED_CATEGORIES for category in selected_categories):
        fail("parameter golf acceptance error: unknown category in `selected_categories`")
    if len(set(selected_categories)) != len(selected_categories):
        fail("parameter golf acceptance error: `selected_categories` must be unique")

    categories = data["categories"]
    if not isinstance(categories, list) or not categories:
        fail("parameter golf acceptance error: `categories` must be a non-empty array")
    seen = []
    for category in categories:
        for key in [
            "category_id",
            "matrix_status",
            "issue_refs",
            "green_definition",
            "current_repo_truth",
            "boundary_note",
        ]:
            if key not in category:
                fail(
                    f"parameter golf acceptance error: category `{category.get('category_id', '<unknown>')}` is missing `{key}`"
                )
        category_id = category["category_id"]
        if category_id not in ALLOWED_CATEGORIES:
            fail(f"parameter golf acceptance error: unknown category id `{category_id}`")
        if category["matrix_status"] not in ALLOWED_STATUSES:
            fail(f"parameter golf acceptance error: unknown matrix status for `{category_id}`")
        if not isinstance(category["issue_refs"], list) or not category["issue_refs"]:
            fail(f"parameter golf acceptance error: `{category_id}` needs at least one issue ref")
        seen.append(category_id)

    if seen != selected_categories:
        fail("parameter golf acceptance error: category ordering must match `selected_categories`")

    expected_digest = stable_digest({k: v for k, v in data.items() if k != "report_digest"})
    if data["report_digest"] != expected_digest:
        fail(
            "parameter golf acceptance error: `report_digest` does not match the canonical payload"
        )

    claim = data["current_claim_posture"]
    statuses = {category["category_id"]: category["matrix_status"] for category in categories}
    if claim == "research":
        pass
    elif claim == "non_record_submission":
        needed = ["challenge-oracle-parity", "packaging-readiness"]
        if any(statuses.get(category) != "implemented" for category in needed):
            fail("parameter golf acceptance error: `non_record_submission` requires oracle and packaging closure")
    elif claim == "record_candidate_blocked_on_accounting":
        needed = [
            "challenge-oracle-parity",
            "single-device-trainer-parity",
            "distributed-throughput-closure",
        ]
        if any(statuses.get(category) != "implemented" for category in needed):
            fail("parameter golf acceptance error: `record_candidate_blocked_on_accounting` requires oracle, trainer, and throughput closure")
        if statuses.get("record-track-readiness") == "implemented":
            fail("parameter golf acceptance error: blocked-on-accounting posture cannot coexist with green record-track readiness")
    elif claim == "record_ready":
        if any(status != "implemented" for status in statuses.values()):
            fail("parameter golf acceptance error: `record_ready` requires every category to be implemented")


def filtered_report(data: dict, selected: list[str]) -> dict:
    if not selected:
        payload = dict(data)
    else:
        unknown = [category for category in selected if category not in ALLOWED_CATEGORIES]
        if unknown:
            fail(
                "parameter golf acceptance error: unknown category filter(s): "
                + ", ".join(unknown)
            )
        deduped = []
        for category in selected:
            if category not in deduped:
                deduped.append(category)
        category_map = {category["category_id"]: category for category in data["categories"]}
        payload = dict(data)
        payload["selected_categories"] = deduped
        payload["categories"] = [category_map[category] for category in deduped]
    payload["report_digest"] = stable_digest({k: v for k, v in payload.items() if k != "report_digest"})
    return payload


with INPUT_PATH.open("r", encoding="utf-8") as handle:
    report = json.load(handle)

validate_report(report)
output = filtered_report(report, SELECTED)
validate_report(output)

encoded = json.dumps(output, indent=2) + "\n"
if REPORT_PATH:
    Path(REPORT_PATH).write_text(encoded, encoding="utf-8")
else:
    sys.stdout.write(encoded)
PY
