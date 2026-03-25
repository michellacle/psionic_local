#!/usr/bin/env bash

set -euo pipefail

report_path=""

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-read-input-materialization-env.sh --report <path>
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
      echo "error: unknown argument $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${report_path}" ]]; then
  usage
  exit 1
fi

python3 - "${report_path}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
with report_path.open("r", encoding="utf-8") as handle:
    report = json.load(handle)

dataset_root = report.get("dataset_root")
tokenizer_path = report.get("tokenizer_path")
if not dataset_root or not tokenizer_path:
    raise SystemExit(
        f"materialization report {report_path} is missing dataset_root or tokenizer_path"
    )

print(
    "export PSIONIC_PARAMETER_GOLF_DATASET_ROOT="
    + shlex.quote(str(dataset_root))
)
print(
    "export PSIONIC_PARAMETER_GOLF_TOKENIZER_PATH="
    + shlex.quote(str(tokenizer_path))
)
PY
