#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

report_path=""
poll_seconds="30"
timeout_seconds="0"
output_path=""
summary_path=""
binary_path=""
tmpdir=""
cargo_target_dir=""

usage() {
  cat <<'EOF' >&2
Usage: scripts/wait-parameter-golf-homegolf-artifact-score.sh --report <path> [options]

Options:
  --report <path>                      Training report to wait for.
  --poll-seconds <n>                   Poll interval while waiting. Default: 30
  --timeout-seconds <n>                Optional timeout. Default: 0 (no timeout)
  --output <path>                      Detached score report output path.
  --summary <path>                     Detached score summary output path.
  --binary-path <path>                 Optional scorer binary path.
  --tmpdir <path>                      Optional TMPDIR for score closeout.
  --cargo-target-dir <path>            Optional cargo target dir for score closeout.
  --help|-h                            Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report)
      report_path="$2"
      shift 2
      ;;
    --poll-seconds)
      poll_seconds="$2"
      shift 2
      ;;
    --timeout-seconds)
      timeout_seconds="$2"
      shift 2
      ;;
    --output)
      output_path="$2"
      shift 2
      ;;
    --summary)
      summary_path="$2"
      shift 2
      ;;
    --binary-path)
      binary_path="$2"
      shift 2
      ;;
    --tmpdir)
      tmpdir="$2"
      shift 2
      ;;
    --cargo-target-dir)
      cargo_target_dir="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${report_path}" ]]; then
  echo "error: --report is required" >&2
  usage
  exit 1
fi

report_path="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${report_path}")"
report_dir="$(dirname "${report_path}")"
report_stem="$(basename "${report_path}" .json)"

if [[ -z "${output_path}" ]]; then
  output_path="${report_dir}/${report_stem}_artifact_score.json"
fi
if [[ -z "${summary_path}" ]]; then
  summary_path="${report_dir}/${report_stem}_artifact_score_summary.json"
fi

mkdir -p "$(dirname "${output_path}")" "$(dirname "${summary_path}")"

start_epoch="$(date +%s)"
while true; do
  if [[ -f "${report_path}" ]]; then
    if python3 - <<'PY' "${report_path}" >/dev/null 2>&1
import json, sys
json.load(open(sys.argv[1], "r", encoding="utf-8"))
PY
    then
      break
    fi
  fi
  if [[ "${timeout_seconds}" != "0" ]]; then
    now_epoch="$(date +%s)"
    elapsed="$((now_epoch - start_epoch))"
    if (( elapsed >= timeout_seconds )); then
      echo "error: timed out waiting for ${report_path}" >&2
      exit 1
    fi
  fi
  sleep "${poll_seconds}"
done

score_command=()
if [[ -n "${binary_path}" ]]; then
  if [[ ! -x "${binary_path}" ]]; then
    echo "error: binary path ${binary_path} is not executable" >&2
    exit 1
  fi
  score_command=("${binary_path}")
else
  score_command=(
    cargo
    run
    -q
    -p
    psionic-train
    --bin
    parameter_golf_homegolf_artifact_score
    --
  )
fi

cd "${repo_root}"
score_env=()
if [[ -n "${tmpdir}" ]]; then
  score_env+=("TMPDIR=${tmpdir}")
fi
if [[ -n "${cargo_target_dir}" ]]; then
  score_env+=("CARGO_TARGET_DIR=${cargo_target_dir}")
fi
env "${score_env[@]}" "${score_command[@]}" \
  --report \
  "${report_path}" \
  --output \
  "${output_path}" >/dev/null

python3 - <<'PY' "${report_path}" "${output_path}" "${summary_path}"
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
score_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])

report = json.loads(report_path.read_text(encoding="utf-8"))
score = json.loads(score_path.read_text(encoding="utf-8"))
validation = score.get("validation") or {}
summary = {
    "schema_version": "psionic.homegolf_artifact_score_summary.v1",
    "report_path": str(report_path),
    "score_report_path": str(score_path),
    "run_id": report.get("run_id"),
    "machine_profile": report.get("machine_profile"),
    "model_variant": report.get("model_variant"),
    "disposition": report.get("disposition"),
    "executed_steps": report.get("executed_steps"),
    "observed_training_time_ms": report.get("observed_training_time_ms"),
    "compressed_model_artifact_path": report.get("compressed_model_artifact_path"),
    "final_validation_mode": report.get("final_validation_mode"),
    "validation_eval_mode": score.get("validation_eval_mode"),
    "batch_token_budget": score.get("batch_token_budget"),
    "detached_validation_bits_per_byte": validation.get("bits_per_byte"),
    "detached_validation_mean_loss": validation.get("mean_loss"),
    "detached_eval_ms": score.get("observed_eval_ms"),
    "claim_boundary": score.get("claim_boundary"),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
