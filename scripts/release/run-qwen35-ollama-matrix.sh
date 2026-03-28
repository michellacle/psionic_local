#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

json_escape() {
  local value="${1-}"
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  value=${value//$'\n'/\\n}
  value=${value//$'\r'/\\r}
  value=${value//$'\t'/\\t}
  printf '%s' "$value"
}

sanitize_id() {
  local value="${1-}"
  value="${value//:/_}"
  value="${value//\//_}"
  value="${value// /_}"
  printf '%s' "$value"
}

extract_number_field() {
  local field="$1"
  local path="$2"
  rg -m1 -o "\"${field}\": [0-9]+(\\.[0-9]+)?" "$path" | awk '{print $2}'
}

extract_output_tokens_csv() {
  local path="$1"
  rg -o '"output_tokens": [0-9]+' "$path" | awk '{print $2}' | paste -sd, -
}

require_model_path() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing ${label} model artifact: $path" >&2
    exit 1
  fi
}

export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"

host_label="${PSIONIC_QWEN35_MATRIX_HOST_LABEL:-$(hostname -s | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-')}"
repeats="${PSIONIC_QWEN35_MATRIX_REPEATS:-3}"
contracts_csv="${PSIONIC_QWEN35_MATRIX_CONTRACTS:-greedy,sampled_topk40,sampled_topk100}"
models_csv="${PSIONIC_QWEN35_MATRIX_MODELS:-qwen3.5:0.8b,qwen3.5:2b,qwen3.5:4b,qwen3.5:9b}"
ollama_base_url="${PSIONIC_QWEN35_MATRIX_OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
change_rationale="${PSIONIC_QWEN35_MATRIX_CHANGE_RATIONALE:-Michel follow-up rerun with repo-owned per-run evidence capture}"
ollama_change_rationale="${PSIONIC_QWEN35_MATRIX_OLLAMA_CHANGE_RATIONALE:-Local Ollama comparison on the same host and prompt contract}"

timestamp="$(date -u +%Y%m%d_%H%M%S)"
run_id="qwen35_ollama_matrix_${timestamp}_${host_label}"
matrix_dir="fixtures/qwen35/benchmarks"
report_dir="${matrix_dir}/reports/${run_id}"
row_report_dir="${report_dir}/rows"
matrix_path="${matrix_dir}/${run_id}.json"
summary_path="${report_dir}/one_page_summary.md"

mkdir -p "$row_report_dir"

declare -A model_paths=(
  ["qwen3.5:0.8b"]="${PSIONIC_QWEN35_08B_GGUF_PATH:-/home/christopherdavid/models/qwen3.5/qwen3.5-0.8b-q8_0.gguf}"
  ["qwen3.5:2b"]="${PSIONIC_QWEN35_2B_GGUF_PATH:-/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf}"
  ["qwen3.5:4b"]="${PSIONIC_QWEN35_4B_GGUF_PATH:-/home/christopherdavid/models/qwen3.5/qwen3.5-4b-q8_0-registry.gguf}"
  ["qwen3.5:9b"]="${PSIONIC_QWEN35_9B_GGUF_PATH:-/home/christopherdavid/models/qwen3.5/qwen3.5-9b-q4_k_m-registry.gguf}"
)

declare -A ollama_models=(
  ["qwen3.5:0.8b"]="${PSIONIC_QWEN35_08B_OLLAMA_MODEL:-qwen3.5:0.8b}"
  ["qwen3.5:2b"]="${PSIONIC_QWEN35_2B_OLLAMA_MODEL:-qwen3.5:2b}"
  ["qwen3.5:4b"]="${PSIONIC_QWEN35_4B_OLLAMA_MODEL:-qwen3.5:4b}"
  ["qwen3.5:9b"]="${PSIONIC_QWEN35_9B_OLLAMA_MODEL:-qwen3.5:9b}"
)

IFS=',' read -r -a contracts <<< "$contracts_csv"
IFS=',' read -r -a models <<< "$models_csv"

for model in "${models[@]}"; do
  require_model_path "$model" "${model_paths[$model]-}"
done

psionic_commit="$(git rev-parse HEAD)"
ollama_version="$(
  if command -v ollama >/dev/null 2>&1; then
    ollama --version 2>/dev/null || true
  fi
)"

gpu_name=""
gpu_memory_total=""
gpu_power_limit=""
gpu_power_default_limit=""
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_query="$(nvidia-smi --query-gpu=name,memory.total,power.limit,power.default_limit --format=csv,noheader 2>/dev/null | head -n1 || true)"
  if [[ -n "$gpu_query" ]]; then
    IFS=',' read -r gpu_name gpu_memory_total gpu_power_limit gpu_power_default_limit <<< "$gpu_query"
    gpu_name="${gpu_name## }"
    gpu_memory_total="${gpu_memory_total## }"
    gpu_power_limit="${gpu_power_limit## }"
    gpu_power_default_limit="${gpu_power_default_limit## }"
  fi
fi

{
  printf '# Qwen35 Native CUDA vs Ollama Matrix Summary\n\n'
  printf 'Run ID: `%s`\n\n' "$run_id"
  printf 'Generated at: `%s`\n\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Psionic commit: `%s`\n\n' "$psionic_commit"
  printf 'Ollama version: `%s`\n\n' "${ollama_version:-unknown}"
  printf 'Host label: `%s`\n\n' "$host_label"
  printf 'Host GPU: `%s`\n\n' "${gpu_name:-unknown}"
  printf 'Host GPU memory total: `%s`\n\n' "${gpu_memory_total:-unknown}"
  printf 'Host power limit: `%s`\n\n' "${gpu_power_limit:-unknown}"
  printf 'Host default power limit: `%s`\n\n' "${gpu_power_default_limit:-unknown}"
  printf 'Repeats per row: `%s`\n\n' "$repeats"
  printf 'CARGO_INCREMENTAL: `%s`\n\n' "$CARGO_INCREMENTAL"
  printf 'Change rationale: `%s`\n\n' "$change_rationale"
  printf 'Ollama comparison rationale: `%s`\n\n' "$ollama_change_rationale"
  printf '| Contract | Model | Psionic tok/s | Ollama tok/s | Ratio | Psionic output tokens | Ollama output tokens | Comparable |\n'
  printf '| --- | --- | ---: | ---: | ---: | --- | --- | --- |\n'
} > "$summary_path"

manifest_tmp="${matrix_path}.tmp"
{
  printf '{\n'
  printf '  "schema_version": 1,\n'
  printf '  "report_kind": "qwen35_ollama_matrix",\n'
  printf '  "run_id": "%s",\n' "$(json_escape "$run_id")"
  printf '  "generated_at_utc": "%s",\n' "$(json_escape "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
  printf '  "psionic_commit": "%s",\n' "$(json_escape "$psionic_commit")"
  printf '  "ollama_version": "%s",\n' "$(json_escape "${ollama_version:-unknown}")"
  printf '  "change_rationale": "%s",\n' "$(json_escape "$change_rationale")"
  printf '  "ollama_change_rationale": "%s",\n' "$(json_escape "$ollama_change_rationale")"
  printf '  "repeats_per_row": %s,\n' "$repeats"
  printf '  "cargo_incremental": "%s",\n' "$(json_escape "$CARGO_INCREMENTAL")"
  printf '  "host": {\n'
  printf '    "label": "%s",\n' "$(json_escape "$host_label")"
  printf '    "gpu_name": "%s",\n' "$(json_escape "${gpu_name:-unknown}")"
  printf '    "gpu_memory_total": "%s",\n' "$(json_escape "${gpu_memory_total:-unknown}")"
  printf '    "gpu_power_limit": "%s",\n' "$(json_escape "${gpu_power_limit:-unknown}")"
  printf '    "gpu_power_default_limit": "%s",\n' "$(json_escape "${gpu_power_default_limit:-unknown}")"
  printf '    "ollama_base_url": "%s"\n' "$(json_escape "$ollama_base_url")"
  printf '  },\n'
  printf '  "rows": [\n'
} > "$manifest_tmp"

row_sep=""
for contract in "${contracts[@]}"; do
  contract="${contract## }"
  contract="${contract%% }"
  case "$contract" in
    greedy)
      prompt="Explain what Psionic is in one sentence."
      contract_args=(--decode greedy --max-output-tokens 128)
      ;;
    sampled_topk40)
      prompt="Respond with plain text only. Write 20 numbered one-sentence reasons why GPU decode throughput matters for local inference. Start at 1 and keep going until all 20 are written."
      contract_args=(
        --decode sample
        --temperature 0.8
        --top-k 40
        --top-p 0.9
        --repeat-penalty 1.0
        --presence-penalty 0.0
        --frequency-penalty 0.0
        --seed 42
        --max-output-tokens 128
      )
      ;;
    sampled_topk100)
      prompt="Explain what Psionic is in one sentence."
      contract_args=(
        --decode sample
        --temperature 0.8
        --top-k 100
        --top-p 0.9
        --min-p 0.05
        --seed 42
        --max-output-tokens 128
      )
      ;;
    *)
      echo "unknown contract: $contract" >&2
      exit 1
      ;;
  esac

  for model in "${models[@]}"; do
    model="${model## }"
    model="${model%% }"
    row_id="$(sanitize_id "${contract}_${model}")"
    model_path="${model_paths[$model]}"
    ollama_model="${ollama_models[$model]}"
    psionic_report="${row_report_dir}/${row_id}_psionic.json"
    ollama_report="${row_report_dir}/${row_id}_ollama.json"
    psionic_rel="${psionic_report#$repo_root/}"
    ollama_rel="${ollama_report#$repo_root/}"

    cargo run --release -p psionic-serve --example qwen35_cuda_bench -- \
      --backend psionic \
      --model-path "$model_path" \
      --prompt "$prompt" \
      --repeats "$repeats" \
      --json-out "$psionic_report" \
      "${contract_args[@]}"

    cargo run --release -p psionic-serve --example qwen35_cuda_bench -- \
      --backend ollama \
      --model-path "$model_path" \
      --ollama-model "$ollama_model" \
      --ollama-base-url "$ollama_base_url" \
      --prompt "$prompt" \
      --repeats "$repeats" \
      --json-out "$ollama_report" \
      "${contract_args[@]}"

    psionic_mean="$(extract_number_field mean_decode_tok_s "$psionic_report")"
    ollama_mean="$(extract_number_field mean_decode_tok_s "$ollama_report")"
    psionic_tokens="$(extract_output_tokens_csv "$psionic_report")"
    ollama_tokens="$(extract_output_tokens_csv "$ollama_report")"
    comparable="no"
    if [[ "$psionic_tokens" == "$ollama_tokens" ]]; then
      comparable="yes"
    fi
    ratio="$(awk -v ps="$psionic_mean" -v ol="$ollama_mean" 'BEGIN { if (ol == 0) { print "0.00" } else { printf "%.2f", ps / ol } }')"

    printf '| `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` |\n' \
      "$contract" \
      "$model" \
      "$psionic_mean" \
      "$ollama_mean" \
      "$ratio" \
      "$psionic_tokens" \
      "$ollama_tokens" \
      "$comparable" >> "$summary_path"

    printf '%s' "$row_sep" >> "$manifest_tmp"
    row_sep=$',\n'
    {
      printf '    {\n'
      printf '      "contract": "%s",\n' "$(json_escape "$contract")"
      printf '      "model": "%s",\n' "$(json_escape "$model")"
      printf '      "model_path": "%s",\n' "$(json_escape "$model_path")"
      printf '      "ollama_model": "%s",\n' "$(json_escape "$ollama_model")"
      printf '      "psionic_report_path": "%s",\n' "$(json_escape "$psionic_rel")"
      printf '      "ollama_report_path": "%s",\n' "$(json_escape "$ollama_rel")"
      printf '      "output_tokens_match": %s,\n' "$([[ "$comparable" == "yes" ]] && printf true || printf false)"
      printf '      "psionic_report": '
      cat "$psionic_report"
      printf ',\n'
      printf '      "ollama_report": '
      cat "$ollama_report"
      printf '\n    }'
    } >> "$manifest_tmp"
  done
done

{
  printf '\n  ]\n'
  printf '}\n'
} >> "$manifest_tmp"

mv "$manifest_tmp" "$matrix_path"

printf '\nArtifacts:\n- %s\n- %s\n' "$matrix_path" "$summary_path"
