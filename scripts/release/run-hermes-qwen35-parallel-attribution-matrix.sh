#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "missing required command: $name" >&2
    exit 1
  fi
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing ${label}: $path" >&2
    exit 1
  fi
}

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
  exit 1
fi

server_bin="${PSIONIC_HERMES_SERVER_BIN:-$repo_root/target/debug/psionic-openai-server}"
if [[ "${PSIONIC_HERMES_SKIP_BUILD:-0}" != "1" ]]; then
  cargo build -q -p psionic-serve --bin psionic-openai-server
fi
if [[ ! -x "$server_bin" ]]; then
  echo "missing psionic-openai-server binary after build: $server_bin" >&2
  exit 1
fi

require_command curl
require_command python3

ollama_root="${PSIONIC_HERMES_OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
ollama_root="${ollama_root%/}"
if ! curl -fsS "${ollama_root}/api/tags" >/dev/null 2>&1; then
  echo "ollama did not answer on ${ollama_root}" >&2
  exit 1
fi

host_label="${PSIONIC_HERMES_HOST_LABEL:-$(hostname -s | tr '[:upper:]' '[:lower:]')}"
models_csv="${PSIONIC_HERMES_PARALLEL_MATRIX_MODELS:-2b,4b,9b}"
backend="${PSIONIC_HERMES_BACKEND:-cuda}"
host="${PSIONIC_HERMES_HOST:-127.0.0.1}"
base_port="${PSIONIC_HERMES_PARALLEL_BASE_PORT:-8105}"
timestamp="$(date -u +%Y%m%d)"
report_path="${PSIONIC_HERMES_PARALLEL_MATRIX_REPORT_PATH:-$repo_root/fixtures/qwen35/hermes/hermes_qwen35_parallel_tool_attribution_matrix_${timestamp}_${host_label}.json}"
rows_dir="${PSIONIC_HERMES_PARALLEL_MATRIX_ROWS_DIR:-$repo_root/fixtures/qwen35/hermes/parallel_matrix_rows}"
log_root="${PSIONIC_HERMES_PARALLEL_MATRIX_LOG_ROOT:-$repo_root/target/hermes/parallel_matrix}"
tmp_root="${TMPDIR:-$repo_root/target/hermes/tmp/parallel_matrix}"

mkdir -p "$(dirname "$report_path")" "$rows_dir" "$log_root" "$tmp_root"

psionic_model_path_for_label() {
  case "$1" in
    2b) printf '%s' "${PSIONIC_HERMES_QWEN35_2B_MODEL_PATH:-/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf}" ;;
    4b) printf '%s' "${PSIONIC_HERMES_QWEN35_4B_MODEL_PATH:-/home/christopherdavid/models/qwen3.5/qwen3.5-4b-q8_0-registry.gguf}" ;;
    9b) printf '%s' "${PSIONIC_HERMES_QWEN35_9B_MODEL_PATH:-/home/christopherdavid/models/qwen3.5/qwen3.5-9b-q4_k_m-registry.gguf}" ;;
    *) echo "unsupported psionic model label: $1" >&2; exit 1 ;;
  esac
}

ollama_model_for_label() {
  case "$1" in
    2b) printf '%s' "${PSIONIC_HERMES_OLLAMA_2B_MODEL:-qwen3.5:2b}" ;;
    4b) printf '%s' "${PSIONIC_HERMES_OLLAMA_4B_MODEL:-qwen3.5:4b}" ;;
    9b) printf '%s' "${PSIONIC_HERMES_OLLAMA_9B_MODEL:-qwen3.5:9b}" ;;
    *) echo "unsupported ollama model label: $1" >&2; exit 1 ;;
  esac
}

annotate_row() {
  local row_path="$1"
  local backend_label="$2"
  local model_label="$3"
  local model_ref="$4"
  local probe_exit_status="$5"
  python3 - "$row_path" "$backend_label" "$model_label" "$model_ref" "$probe_exit_status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

row_path = Path(sys.argv[1])
backend_label = sys.argv[2]
model_label = sys.argv[3]
model_ref = sys.argv[4]
probe_exit_status = int(sys.argv[5])

if row_path.exists():
    data = json.loads(row_path.read_text())
else:
    data = {
        "report_kind": "psionic_hermes_qwen35_parallel_tool_row",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend_label": backend_label,
        "cases": [],
        "overall_pass": False,
        "passing_case_count": 0,
        "total_case_count": 0,
        "row_status": "missing_report",
        "failure_detail": f"probe exited {probe_exit_status} without writing a report",
    }

data["matrix_backend_label"] = backend_label
data["matrix_model_label"] = model_label
data["matrix_model_ref"] = model_ref
data["probe_exit_status"] = probe_exit_status
row_path.write_text(json.dumps(data, indent=2) + "\n")
PY
}

write_synthetic_row() {
  local row_path="$1"
  local backend_label="$2"
  local model_label="$3"
  local model_ref="$4"
  local row_status="$5"
  local failure_detail="$6"
  local probe_exit_status="$7"
  python3 - "$row_path" "$backend_label" "$model_label" "$model_ref" "$row_status" "$failure_detail" "$probe_exit_status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

row_path = Path(sys.argv[1])
backend_label = sys.argv[2]
model_label = sys.argv[3]
model_ref = sys.argv[4]
row_status = sys.argv[5]
failure_detail = sys.argv[6]
probe_exit_status = int(sys.argv[7])

data = {
    "report_kind": "psionic_hermes_qwen35_parallel_tool_row",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "backend_label": backend_label,
    "matrix_backend_label": backend_label,
    "matrix_model_label": model_label,
    "matrix_model_ref": model_ref,
    "probe_exit_status": probe_exit_status,
    "overall_pass": False,
    "passing_case_count": 0,
    "total_case_count": 0,
    "row_status": row_status,
    "failure_detail": failure_detail,
    "cases": [],
}
row_path.write_text(json.dumps(data, indent=2) + "\n")
PY
}

run_psionic_row() {
  local model_label="$1"
  local port="$2"
  local model_path="$3"
  local row_path="$4"
  local stdout_path="$5"
  local server_log_path="$6"
  local tmpdir="$7"
  local server_pid=""

  TMPDIR="$tmpdir" "$server_bin" \
    --backend "$backend" \
    --host "$host" \
    --port "$port" \
    -m "$model_path" \
    >"$server_log_path" 2>&1 &
  server_pid="$!"

  for _ in $(seq 1 60); do
    if curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if ! curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1; then
    if [[ -n "$server_pid" ]]; then
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
    write_synthetic_row \
      "$row_path" \
      psionic \
      "$model_label" \
      "$model_path" \
      startup_failure \
      "psionic-openai-server failed health check on ${host}:${port}" \
      124
    echo "psionic ${model_label} startup_failure row=${row_path}"
    return 0
  fi

  set +e
  OPENAI_API_KEY=dummy \
  OPENAI_BASE_URL="http://${host}:${port}/v1" \
  "$python_bin" "$repo_root/scripts/release/hermes_qwen35_compatibility_probe.py" \
    --hermes-root "$hermes_root" \
    --base-url "http://${host}:${port}/v1" \
    --model "$(basename "$model_path")" \
    --model-path "$model_path" \
    --psionic-root "$repo_root" \
    --report-path "$row_path" \
    --backend-label psionic \
    --only-case parallel_tool_turn \
    >"$stdout_path" 2>&1
  local probe_exit_status=$?
  set -e
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  annotate_row "$row_path" psionic "$model_label" "$model_path" "$probe_exit_status"
  echo "psionic ${model_label} exit=${probe_exit_status} row=${row_path}"
}

run_ollama_row() {
  local model_label="$1"
  local model_ref="$2"
  local row_path="$3"
  local stdout_path="$4"
  set +e
  OPENAI_API_KEY=dummy \
  OPENAI_BASE_URL="${ollama_root}/v1" \
  "$python_bin" "$repo_root/scripts/release/hermes_qwen35_compatibility_probe.py" \
    --hermes-root "$hermes_root" \
    --base-url "${ollama_root}/v1" \
    --model "$model_ref" \
    --model-path "$model_ref" \
    --psionic-root "$repo_root" \
    --report-path "$row_path" \
    --backend-label ollama \
    --only-case parallel_tool_turn \
    >"$stdout_path" 2>&1
  local probe_exit_status=$?
  set -e
  annotate_row "$row_path" ollama "$model_label" "$model_ref" "$probe_exit_status"
  echo "ollama ${model_label} exit=${probe_exit_status} row=${row_path}"
}

row_paths=()
IFS=',' read -r -a models <<< "$models_csv"
port="$base_port"
for model_label in "${models[@]}"; do
  model_label="${model_label// /}"
  [[ -n "$model_label" ]] || continue

  psionic_model_path="$(psionic_model_path_for_label "$model_label")"
  ollama_model_ref="$(ollama_model_for_label "$model_label")"
  require_file "psionic ${model_label} artifact" "$psionic_model_path"
  if ! curl -fsS "${ollama_root}/api/show" -d "{\"name\":\"${ollama_model_ref}\"}" >/dev/null 2>&1; then
    echo "missing Ollama model: ${ollama_model_ref}" >&2
    exit 1
  fi

  psionic_row_path="${rows_dir}/hermes_psionic_parallel_tool_${model_label}_${host_label}.json"
  psionic_stdout_path="${log_root}/hermes_psionic_parallel_tool_${model_label}_${host_label}.stdout.log"
  psionic_server_log_path="${log_root}/hermes_psionic_parallel_tool_${model_label}_${host_label}.server.log"
  psionic_tmpdir="${tmp_root}/psionic_${model_label}"
  mkdir -p "$psionic_tmpdir"
  run_psionic_row "$model_label" "$port" "$psionic_model_path" "$psionic_row_path" "$psionic_stdout_path" "$psionic_server_log_path" "$psionic_tmpdir"
  row_paths+=("$psionic_row_path")
  port=$((port + 1))

  ollama_row_path="${rows_dir}/hermes_ollama_parallel_tool_${model_label}_${host_label}.json"
  ollama_stdout_path="${log_root}/hermes_ollama_parallel_tool_${model_label}_${host_label}.stdout.log"
  run_ollama_row "$model_label" "$ollama_model_ref" "$ollama_row_path" "$ollama_stdout_path"
  row_paths+=("$ollama_row_path")
done

aggregate_args=(
  --report-path "$report_path"
  --repo-root "$repo_root"
  --host "$host_label"
)
for row_path in "${row_paths[@]}"; do
  aggregate_args+=(--row-path "$row_path")
done

"$python_bin" "$repo_root/scripts/release/hermes_qwen35_parallel_matrix_aggregate.py" "${aggregate_args[@]}"
