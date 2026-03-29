#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate retained Hermes strict-parallel row reports into one attribution matrix."
    )
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--row-path", action="append", required=True)
    return parser.parse_args()


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def extract_parallel_case(row: dict) -> dict | None:
    for case in row.get("cases", []):
        if case.get("case_id") == "parallel_tool_turn":
            return case
    return None


def main() -> int:
    args = parse_args()
    report_path = Path(args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.repo_root).resolve()
    row_paths = [Path(path).resolve() for path in args.row_path]
    rows = [json.loads(path.read_text()) for path in row_paths]

    matrix_rows = []
    for row, path in zip(rows, row_paths):
        case = extract_parallel_case(row)
        assistant_tool_names = []
        request_summaries = []
        if case is not None:
            assistant_tool_names = case.get("details", {}).get("assistant_tool_names", [])
            request_summaries = case.get("request_summaries", [])
        matrix_rows.append(
            {
                "row_path": display_path(path, repo_root),
                "backend_label": row.get("matrix_backend_label", row.get("backend_label")),
                "model_label": row.get("matrix_model_label"),
                "model_ref": row.get("matrix_model_ref", row.get("model_path", row.get("model"))),
                "probe_exit_status": row.get("probe_exit_status"),
                "overall_pass": row.get("overall_pass"),
                "case_present": case is not None,
                "parallel_tool_turn_pass": (case or {}).get("pass"),
                "assistant_tool_names": assistant_tool_names,
                "request_summaries": request_summaries,
                "failure_detail": row.get("failure_detail"),
                "row_status": row.get("row_status"),
                "case_summary": (case or {}).get("summary"),
            }
        )

    by_model: dict[str, dict[str, dict]] = {}
    for item in matrix_rows:
        by_model.setdefault(item["model_label"], {})[item["backend_label"]] = item

    per_model_verdicts = []
    for model_label in sorted(by_model):
        pair = by_model[model_label]
        psionic = pair.get("psionic")
        ollama = pair.get("ollama")
        if psionic and ollama:
            if psionic["parallel_tool_turn_pass"] and ollama["parallel_tool_turn_pass"]:
                verdict = "passes_on_both_backends"
            elif (psionic["parallel_tool_turn_pass"] is False) and (ollama["parallel_tool_turn_pass"] is False):
                if psionic["assistant_tool_names"] == ollama["assistant_tool_names"]:
                    verdict = "failure_reproduced_with_same_tool_sequence"
                else:
                    verdict = "failure_reproduced_with_different_tool_sequence"
            elif psionic["parallel_tool_turn_pass"] and not ollama["parallel_tool_turn_pass"]:
                verdict = "psionic_only_pass"
            elif ollama["parallel_tool_turn_pass"] and not psionic["parallel_tool_turn_pass"]:
                verdict = "ollama_only_pass"
            else:
                verdict = "mixed_or_incomplete"
        else:
            verdict = "single_backend_only"
        per_model_verdicts.append(
            {
                "model_label": model_label,
                "verdict": verdict,
                "psionic": psionic,
                "ollama": ollama,
            }
        )

    dual_backend_rows = [item for item in per_model_verdicts if item["verdict"] != "single_backend_only"]
    if dual_backend_rows and all(item["verdict"] == "failure_reproduced_with_same_tool_sequence" for item in dual_backend_rows):
        overall_attribution = "shared_model_or_artifact_behavior_reproduced_across_backends"
    elif dual_backend_rows and all(
        item["verdict"] in {"failure_reproduced_with_same_tool_sequence", "failure_reproduced_with_different_tool_sequence"}
        for item in dual_backend_rows
    ):
        overall_attribution = "shared_failure_reproduced_across_backends_with_backend_specific_symptoms"
    elif any(item["verdict"] == "psionic_only_pass" for item in dual_backend_rows):
        overall_attribution = "ollama_lane_specific_failure_present"
    elif any(item["verdict"] == "ollama_only_pass" for item in dual_backend_rows):
        overall_attribution = "native_psionic_lane_specific_failure_present"
    elif dual_backend_rows and all(item["verdict"] == "passes_on_both_backends" for item in dual_backend_rows):
        overall_attribution = "strict_parallel_tool_turn_solved_on_all_reachable_rows"
    else:
        overall_attribution = "mixed_parallel_tool_behavior"

    report = {
        "report_kind": "psionic_hermes_qwen35_parallel_tool_attribution_matrix",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "psionic_revision": rows[0].get("psionic_revision") if rows else None,
        "hermes_revision": rows[0].get("hermes_revision") if rows else None,
        "fixed_contract": {
            "case_id": "parallel_tool_turn",
            "task": "emit exactly one assistant tool-call turn containing get_paris_weather then get_tokyo_weather in one same-turn tool_calls array",
            "temperature": 0,
            "seed": 0,
            "tool_policy": "required_then_auto",
            "benchmark_boundary": "Hermes fixed; backend and artifact/model swap only",
        },
        "rows": matrix_rows,
        "per_model_verdicts": per_model_verdicts,
        "overall_attribution": overall_attribution,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
