#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/benchmark-qwen35-vs-ollama-matrix.sh [--output-dir DIR] [--repeats N] [--ollama-base-url URL] [--models CSV]

Defaults:
  output-dir:       /tmp/qwen35_ollama_matrix_<timestamp>
  repeats:          5
  ollama-base-url:  http://127.0.0.1:11434
  models:           qwen3.5:0.8b,qwen3.5:2b,qwen3.5:4b,qwen3.5:9b

Notes:
  - Runs a serialized matrix: 3 contracts x N models x 2 backends.
  - Captures full per-run JSONL evidence from qwen35_cuda_bench.
  - Captures GPU telemetry with nvidia-smi while the matrix runs.
  - Generates an enriched artifact + one-page report with comparable/non-comparable row split.
EOF
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/tmp/qwen35_ollama_matrix_${TIMESTAMP}"
REPEATS=5
OLLAMA_BASE_URL="http://127.0.0.1:11434"
MODELS_CSV="qwen3.5:0.8b,qwen3.5:2b,qwen3.5:4b,qwen3.5:9b"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR=${2:?missing value for --output-dir}
      shift 2
      ;;
    --repeats)
      REPEATS=${2:?missing value for --repeats}
      shift 2
      ;;
    --ollama-base-url)
      OLLAMA_BASE_URL=${2:?missing value for --ollama-base-url}
      shift 2
      ;;
    --models)
      MODELS_CSV=${2:?missing value for --models}
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unrecognized argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [[ "$REPEATS" -lt 1 ]]; then
  echo "--repeats must be a positive integer" >&2
  exit 1
fi

for tool in cargo jq ollama nvidia-smi python3 git awk sed date; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required tool: $tool" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
RAW_LOG="$OUTPUT_DIR/raw.log"
JSONL="$OUTPUT_DIR/runs.jsonl"
TELEMETRY="$OUTPUT_DIR/telemetry.csv"
RUN_ID="qwen35_ollama_matrix_${TIMESTAMP}"

touch "$RAW_LOG" "$JSONL" "$TELEMETRY"

GIT_COMMIT=$(git rev-parse HEAD)
OLLAMA_VERSION=$(ollama --version | awk '{print $NF}')
KERNEL=$(uname -srmo)

GPU_META=$(nvidia-smi --query-gpu=name,memory.total,power.limit,power.max_limit,driver_version --format=csv,noheader,nounits | head -n 1)
GPU_NAME=$(echo "$GPU_META" | awk -F',' '{gsub(/^ +| +$/,"",$1); print $1}')
GPU_VRAM_MIB=$(echo "$GPU_META" | awk -F',' '{gsub(/^ +| +$/,"",$2); print $2}')
GPU_POWER_LIMIT_W=$(echo "$GPU_META" | awk -F',' '{gsub(/^ +| +$/,"",$3); print $3}')
GPU_POWER_MAX_LIMIT_W=$(echo "$GPU_META" | awk -F',' '{gsub(/^ +| +$/,"",$4); print $4}')
GPU_DRIVER_VERSION=$(echo "$GPU_META" | awk -F',' '{gsub(/^ +| +$/,"",$5); print $5}')

CUDA_RUNTIME_VERSION=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n 1)
NVCC_VERSION=$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p' | head -n 1)

{
  echo "run_id=$RUN_ID"
  echo "timestamp=$TIMESTAMP"
  echo "repo_root=$REPO_ROOT"
  echo "git_commit=$GIT_COMMIT"
  echo "ollama_version=$OLLAMA_VERSION"
  echo "kernel=$KERNEL"
  echo "gpu_name=$GPU_NAME"
  echo "gpu_vram_mib=$GPU_VRAM_MIB"
  echo "gpu_power_limit_w=$GPU_POWER_LIMIT_W"
  echo "gpu_power_max_limit_w=$GPU_POWER_MAX_LIMIT_W"
  echo "gpu_driver_version=$GPU_DRIVER_VERSION"
  echo "cuda_runtime_version=$CUDA_RUNTIME_VERSION"
  echo "nvcc_version=$NVCC_VERSION"
  echo "repeats=$REPEATS"
  echo "models_csv=$MODELS_CSV"
  echo "ollama_base_url=$OLLAMA_BASE_URL"
} > "$OUTPUT_DIR/environment.txt"

TELEMETRY_PID=
cleanup() {
  if [[ -n "${TELEMETRY_PID:-}" ]]; then
    kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

nvidia-smi \
  --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader,nounits \
  -l 1 > "$TELEMETRY" &
TELEMETRY_PID=$!

resolve_model_path() {
  local model=$1
  local from_line
  from_line=$(ollama show --modelfile "$model" | sed -n 's/^FROM //p' | head -n 1)
  if [[ -z "$from_line" ]]; then
    echo "failed to resolve model path for $model via ollama show --modelfile" >&2
    exit 1
  fi
  if [[ "$from_line" == /* ]]; then
    echo "$from_line"
    return
  fi
  if [[ -f "$HOME/.ollama/models/blobs/$from_line" ]]; then
    echo "$HOME/.ollama/models/blobs/$from_line"
    return
  fi
  if [[ -f "/usr/share/ollama/.ollama/models/blobs/$from_line" ]]; then
    echo "/usr/share/ollama/.ollama/models/blobs/$from_line"
    return
  fi
  echo "unable to map modelfile FROM path to local GGUF: $from_line" >&2
  exit 1
}

run_case() {
  local contract=$1
  local backend=$2
  local model_alias=$3
  local model_path=$4
  local prompt=$5
  shift 5
  local extra=("$@")

  local cmd=(
    cargo run -p psionic-serve --example qwen35_cuda_bench --
    --backend "$backend"
    --model-path "$model_path"
    --model-alias "$model_alias"
    --row-label "$contract"
    --prompt "$prompt"
    --max-output-tokens 128
    --repeats "$REPEATS"
    --jsonl-out "$JSONL"
  )
  if [[ "$backend" == "ollama" ]]; then
    cmd+=(
      --ollama-model "$model_alias"
      --ollama-base-url "$OLLAMA_BASE_URL"
    )
  fi
  cmd+=("${extra[@]}")

  {
    echo "===== $contract | $backend | $model_alias ====="
    printf 'command='
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >> "$RAW_LOG"
  "${cmd[@]}" | tee -a "$RAW_LOG"
}

IFS=',' read -r -a MODELS <<< "$MODELS_CSV"

for model_alias in "${MODELS[@]}"; do
  model_path=$(resolve_model_path "$model_alias")
  if [[ ! -f "$model_path" ]]; then
    echo "resolved model path does not exist: $model_path" >&2
    exit 1
  fi

  run_case "greedy" "ollama" "$model_alias" "$model_path" \
    "Explain what Psionic is in one sentence." \
    --decode greedy
  ollama stop "$model_alias" >/dev/null 2>&1 || true
  run_case "greedy" "psionic" "$model_alias" "$model_path" \
    "Explain what Psionic is in one sentence." \
    --decode greedy
  ollama stop "$model_alias" >/dev/null 2>&1 || true

  run_case "sampled_topk40" "ollama" "$model_alias" "$model_path" \
    "Respond with plain text only. Write 20 numbered one-sentence reasons why GPU decode throughput matters for local inference. Start at 1 and keep going until all 20 are written." \
    --decode sample --temperature 0.8 --top-k 40 --top-p 0.9 --repeat-penalty 1.0 --presence-penalty 0.0 --frequency-penalty 0.0 --seed 42
  ollama stop "$model_alias" >/dev/null 2>&1 || true
  run_case "sampled_topk40" "psionic" "$model_alias" "$model_path" \
    "Respond with plain text only. Write 20 numbered one-sentence reasons why GPU decode throughput matters for local inference. Start at 1 and keep going until all 20 are written." \
    --decode sample --temperature 0.8 --top-k 40 --top-p 0.9 --repeat-penalty 1.0 --presence-penalty 0.0 --frequency-penalty 0.0 --seed 42
  ollama stop "$model_alias" >/dev/null 2>&1 || true

  run_case "sampled_topk100" "ollama" "$model_alias" "$model_path" \
    "Explain what Psionic is in one sentence." \
    --decode sample --temperature 0.8 --top-k 100 --top-p 0.9 --min-p 0.05 --seed 42
  ollama stop "$model_alias" >/dev/null 2>&1 || true
  run_case "sampled_topk100" "psionic" "$model_alias" "$model_path" \
    "Explain what Psionic is in one sentence." \
    --decode sample --temperature 0.8 --top-k 100 --top-p 0.9 --min-p 0.05 --seed 42
  ollama stop "$model_alias" >/dev/null 2>&1 || true
done

kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
wait "$TELEMETRY_PID" 2>/dev/null || true
TELEMETRY_PID=

REPORT_ARGS=(
  --jsonl "$JSONL"
  --output-dir "$OUTPUT_DIR"
  --run-id "$RUN_ID"
  --raw-log "$RAW_LOG"
  --telemetry-csv "$TELEMETRY"
  --git-commit "$GIT_COMMIT"
  --ollama-version "$OLLAMA_VERSION"
  --gpu-name "$GPU_NAME"
  --gpu-vram-mib "$GPU_VRAM_MIB"
)
if [[ "$GPU_POWER_LIMIT_W" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  REPORT_ARGS+=(--gpu-power-limit-w "$GPU_POWER_LIMIT_W")
fi
if [[ "$GPU_POWER_MAX_LIMIT_W" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  REPORT_ARGS+=(--gpu-power-max-limit-w "$GPU_POWER_MAX_LIMIT_W")
fi

python3 "$SCRIPT_DIR/report-qwen35-vs-ollama.py" "${REPORT_ARGS[@]}" | tee "$OUTPUT_DIR/report_paths.txt"

echo "benchmark matrix completed"
echo "output_dir=$OUTPUT_DIR"
