#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

hermes_root="${PSIONIC_HERMES_ROOT:-}"
if [[ -z "$hermes_root" ]]; then
  echo "set PSIONIC_HERMES_ROOT to a local Hermes checkout" >&2
  exit 1
fi
if [[ ! -d "$hermes_root" ]]; then
  echo "missing Hermes checkout: $hermes_root" >&2
  exit 1
fi

python_bin="${PSIONIC_HERMES_PYTHON:-$hermes_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "missing Hermes Python interpreter: $python_bin" >&2
  echo "set PSIONIC_HERMES_PYTHON to override the default path" >&2
  exit 1
fi

server_bin="${PSIONIC_HERMES_SERVER_BIN:-$repo_root/target/debug/psionic-openai-server}"
if [[ ! -x "$server_bin" ]]; then
  cargo build -q -p psionic-serve --bin psionic-openai-server
fi

psionic_model_path="${PSIONIC_HERMES_PSIONIC_MODEL_PATH:-/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf}"
if [[ ! -f "$psionic_model_path" ]]; then
  echo "missing Psionic qwen35 model artifact: $psionic_model_path" >&2
  echo "set PSIONIC_HERMES_PSIONIC_MODEL_PATH to override the default path" >&2
  exit 1
fi

ollama_root="${PSIONIC_HERMES_OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
ollama_root="${ollama_root%/}"
ollama_model="${PSIONIC_HERMES_OLLAMA_MODEL:-qwen3.5:2b}"

psionic_host="${PSIONIC_HERMES_HOST:-127.0.0.1}"
psionic_port="${PSIONIC_HERMES_QWEN35_PORT:-8097}"
psionic_base_url="http://${psionic_host}:${psionic_port}/v1"
ollama_base_url="${ollama_root}/v1"

report_path="${PSIONIC_HERMES_BENCHMARK_REPORT_PATH:-$repo_root/fixtures/qwen35/hermes/hermes_psionic_vs_ollama_benchmark_report.json}"
raw_dir="${PSIONIC_HERMES_BENCHMARK_RAW_DIR:-$repo_root/fixtures/qwen35/hermes/backend_rows}"
psionic_row_path="${raw_dir}/psionic_row.json"
ollama_row_path="${raw_dir}/ollama_row.json"
psionic_log_path="${PSIONIC_HERMES_SERVER_LOG_PATH:-$repo_root/target/hermes/hermes_qwen35_psionic_benchmark_server.log}"
tmpdir="${TMPDIR:-$repo_root/target/hermes/tmp}"

mkdir -p "$(dirname "$report_path")" "$raw_dir" "$(dirname "$psionic_log_path")" "$tmpdir"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

TMPDIR="$tmpdir" "$server_bin" \
  --backend cuda \
  --host "$psionic_host" \
  --port "$psionic_port" \
  -m "$psionic_model_path" \
  >"$psionic_log_path" 2>&1 &
server_pid="$!"

psionic_ready_start="$(python3 - <<'PY'
import time
print(f"{time.monotonic():.9f}")
PY
)"
for _ in $(seq 1 60); do
  if curl -fsS "http://${psionic_host}:${psionic_port}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl -fsS "http://${psionic_host}:${psionic_port}/health" >/dev/null 2>&1; then
  echo "psionic-openai-server failed health check on ${psionic_host}:${psionic_port}" >&2
  exit 1
fi
psionic_ready_end="$(python3 - <<'PY'
import time
print(f"{time.monotonic():.9f}")
PY
)"

ollama_ready_start="$(python3 - <<'PY'
import time
print(f"{time.monotonic():.9f}")
PY
)"
if ! curl -fsS "${ollama_root}/api/tags" >/dev/null 2>&1; then
  echo "ollama did not answer on ${ollama_root}" >&2
  exit 1
fi
ollama_ready_end="$(python3 - <<'PY'
import time
print(f"{time.monotonic():.9f}")
PY
)"

psionic_model_name="$(basename "$psionic_model_path")"

OPENAI_API_KEY=dummy \
OPENAI_BASE_URL="$psionic_base_url" \
"$python_bin" "$repo_root/scripts/release/hermes_backend_benchmark_probe.py" \
  --hermes-root "$hermes_root" \
  --base-url "$psionic_base_url" \
  --model "$psionic_model_name" \
  --model-path "$psionic_model_path" \
  --psionic-root "$repo_root" \
  --report-path "$psionic_row_path" \
  --backend-label psionic || true

OPENAI_API_KEY=dummy \
OPENAI_BASE_URL="$ollama_base_url" \
"$python_bin" "$repo_root/scripts/release/hermes_backend_benchmark_probe.py" \
  --hermes-root "$hermes_root" \
  --base-url "$ollama_base_url" \
  --model "$ollama_model" \
  --model-path "$ollama_model" \
  --psionic-root "$repo_root" \
  --report-path "$ollama_row_path" \
  --backend-label ollama || true

python3 - "$report_path" "$psionic_row_path" "$ollama_row_path" "$psionic_ready_start" "$psionic_ready_end" "$ollama_ready_start" "$ollama_ready_end" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

report_path = Path(sys.argv[1])
psionic_row_path = Path(sys.argv[2])
ollama_row_path = Path(sys.argv[3])
psionic_ready_start = float(sys.argv[4])
psionic_ready_end = float(sys.argv[5])
ollama_ready_start = float(sys.argv[6])
ollama_ready_end = float(sys.argv[7])

psionic_row = json.loads(psionic_row_path.read_text())
ollama_row = json.loads(ollama_row_path.read_text())

rows = [psionic_row, ollama_row]
by_label = {row["backend_label"]: row for row in rows}
case_ids = [case["case_id"] for case in psionic_row["cases"]]
comparison = []
for case_id in case_ids:
    psionic_case = next(case for case in psionic_row["cases"] if case["case_id"] == case_id)
    ollama_case = next(case for case in ollama_row["cases"] if case["case_id"] == case_id)
    faster_backend = None
    if psionic_case["wallclock_s"] < ollama_case["wallclock_s"]:
        faster_backend = "psionic"
    elif ollama_case["wallclock_s"] < psionic_case["wallclock_s"]:
        faster_backend = "ollama"
    comparison.append(
        {
            "case_id": case_id,
            "psionic_pass": psionic_case["pass"],
            "ollama_pass": ollama_case["pass"],
            "psionic_wallclock_s": psionic_case["wallclock_s"],
            "ollama_wallclock_s": ollama_case["wallclock_s"],
            "psionic_completion_tok_s": psionic_case["completion_tok_s"],
            "ollama_completion_tok_s": ollama_case["completion_tok_s"],
            "faster_backend": faster_backend,
        }
    )

output = {
    "report_kind": "psionic_hermes_backend_benchmark",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "host": psionic_row["host"],
    "psionic_revision": psionic_row["psionic_revision"],
    "hermes_revision": psionic_row["hermes_revision"],
    "fixed_contract": {
        "api_mode": "chat_completions",
        "same_host": True,
        "same_case_ids": case_ids,
        "temperature": 0,
        "seed": 0,
        "tool_policy": "required_then_auto for tool cases; auto for no-tool case",
        "benchmark_boundary": "Hermes fixed; backend swaps only base URL plus backend-specific model identifier",
    },
    "startup_probe": {
        "psionic_ready_s": psionic_ready_end - psionic_ready_start,
        "ollama_ready_s": ollama_ready_end - ollama_ready_start,
    },
    "backends": rows,
    "comparison": comparison,
}
report_path.write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
PY
