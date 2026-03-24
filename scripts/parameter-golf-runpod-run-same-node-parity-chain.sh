#!/usr/bin/env bash

set -euo pipefail

psionic_repo_root=""
upstream_repo_root=""
run_root=""
trainer_pid=""
upstream_log=""
output_dir=""
status_log=""
git_remote="origin"
git_ref="main"
git_ssh_key=""
poll_seconds="60"

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-runpod-run-same-node-parity-chain.sh --psionic-repo-root <path> --upstream-repo-root <path> --run-root <path> [options]

Options:
  --trainer-pid <pid>        Wait for one live trainer PID to exit before continuing.
  --upstream-log <path>      Upstream train_gpt.py log path. Default: /workspace/parameter_golf_train_gpt_reference_<timestamp>.log
  --output-dir <path>        Same-node parity artifact directory. Default: /workspace/parameter_golf_same_node_parity_<timestamp>
  --status-log <path>        Status log path. Default: /workspace/parameter_golf_same_node_parity_chain.status.log
  --git-remote <name>        Git remote to fast-forward before the upstream run. Default: origin.
  --git-ref <ref>            Git ref to fast-forward before the upstream run. Default: main.
  --git-ssh-key <path>       Optional SSH key for the git fetch/pull step.
  --poll-seconds <n>         Poll interval while waiting for the trainer/report. Default: 60.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --psionic-repo-root)
      psionic_repo_root="$2"
      shift 2
      ;;
    --upstream-repo-root)
      upstream_repo_root="$2"
      shift 2
      ;;
    --run-root)
      run_root="$2"
      shift 2
      ;;
    --trainer-pid)
      trainer_pid="$2"
      shift 2
      ;;
    --upstream-log)
      upstream_log="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --status-log)
      status_log="$2"
      shift 2
      ;;
    --git-remote)
      git_remote="$2"
      shift 2
      ;;
    --git-ref)
      git_ref="$2"
      shift 2
      ;;
    --git-ssh-key)
      git_ssh_key="$2"
      shift 2
      ;;
    --poll-seconds)
      poll_seconds="$2"
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

if [[ -z "${psionic_repo_root}" || -z "${upstream_repo_root}" || -z "${run_root}" ]]; then
  echo "error: --psionic-repo-root, --upstream-repo-root, and --run-root are required" >&2
  usage
  exit 1
fi

if [[ -z "${upstream_log}" ]]; then
  upstream_log="/workspace/parameter_golf_train_gpt_reference_$(date -u +%Y%m%dT%H%M%SZ).log"
fi
if [[ -z "${output_dir}" ]]; then
  output_dir="/workspace/parameter_golf_same_node_parity_$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "${status_log}" ]]; then
  status_log="/workspace/parameter_golf_same_node_parity_chain.status.log"
fi

if [[ ! -d "${psionic_repo_root}" ]]; then
  echo "error: psionic repo root does not exist: ${psionic_repo_root}" >&2
  exit 1
fi
if [[ ! -d "${upstream_repo_root}" ]]; then
  echo "error: upstream repo root does not exist: ${upstream_repo_root}" >&2
  exit 1
fi
if [[ ! -d "${run_root}" ]]; then
  echo "error: run root does not exist: ${run_root}" >&2
  exit 1
fi

mkdir -p "$(dirname -- "${status_log}")"
exec >>"${status_log}" 2>&1

report_path="${run_root}/parameter_golf_single_h100_training.json"

printf 'chain_start started_at_utc=%s run_root=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_root}"

if [[ -n "${trainer_pid}" ]]; then
  while kill -0 "${trainer_pid}" 2>/dev/null; do
    sleep "${poll_seconds}"
  done
  printf 'trainer_exit_observed observed_at_utc=%s trainer_pid=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${trainer_pid}"
else
  until [[ -f "${report_path}" ]]; do
    sleep "${poll_seconds}"
  done
fi

if [[ ! -f "${report_path}" ]]; then
  printf 'missing_psionic_report path=%s\n' "${report_path}"
  exit 0
fi

cd "${psionic_repo_root}"
if [[ -n "${git_ssh_key}" ]]; then
  export GIT_SSH_COMMAND="ssh -i ${git_ssh_key} -o IdentitiesOnly=yes"
fi
git fetch "${git_remote}"
git pull --ff-only "${git_remote}" "${git_ref}"

bash scripts/parameter-golf-runpod-run-train-gpt-reference.sh \
  --repo-root "${upstream_repo_root}" \
  --log "${upstream_log}"

bash scripts/parameter-golf-build-same-node-parity.sh \
  --psionic-report "${report_path}" \
  --upstream-log "${upstream_log}" \
  --output-dir "${output_dir}"

printf 'chain_complete completed_at_utc=%s parity_dir=%s upstream_log=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${output_dir}" "${upstream_log}"
