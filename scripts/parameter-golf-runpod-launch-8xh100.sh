#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

launch_file="${repo_root}/fixtures/parameter_golf/runpod/parameter_golf_runpod_8xh100_launch_profiles_v1.json"
preflight_script="${repo_root}/scripts/parameter-golf-runpod-operator-preflight.sh"

profile_id=""
run_id=""
pod_host=""
pod_port=""
ssh_user="root"
ssh_key_path=""
manifest_only=false
stop_after="finalize"
repo_dir=""

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-runpod-launch-8xh100.sh [options]

Options:
  --profile <profile_id>   Launch profile from the committed RunPod authority file.
  --run-id <run_id>        Stable run identifier for the manifest.
  --pod-host <host>        RunPod pod host or IP for later real execution.
  --pod-port <port>        SSH port for later real execution.
  --ssh-user <user>        SSH user for later real execution. Default: root.
  --ssh-key <path>         SSH private key path for later real execution.
  --repo-dir <path>        Remote repo checkout path. Default: <workspace_root>/psionic.
  --stop-after <phase>     Stop after remote_preflight, pre_training, execution, or finalize.
  --manifest-only          Emit the launch manifest without contacting a pod.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      profile_id="$2"
      shift 2
      ;;
    --run-id)
      run_id="$2"
      shift 2
      ;;
    --pod-host)
      pod_host="$2"
      shift 2
      ;;
    --pod-port)
      pod_port="$2"
      shift 2
      ;;
    --ssh-user)
      ssh_user="$2"
      shift 2
      ;;
    --ssh-key)
      ssh_key_path="$2"
      shift 2
      ;;
    --repo-dir)
      repo_dir="$2"
      shift 2
      ;;
    --stop-after)
      stop_after="$2"
      shift 2
      ;;
    --manifest-only)
      manifest_only=true
      shift
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

if [[ -z "${profile_id}" ]]; then
  echo "error: --profile is required" >&2
  usage
  exit 1
fi

if [[ -z "${run_id}" ]]; then
  run_id="parameter-golf-runpod-8xh100-$(date -u +%Y%m%dT%H%M%SZ)"
fi

case "${stop_after}" in
  remote_preflight|pre_training|execution|finalize)
    ;;
  *)
    echo "error: --stop-after must be one of remote_preflight, pre_training, execution, or finalize" >&2
    exit 1
    ;;
esac

if [[ "${manifest_only}" != "true" && ( -z "${pod_host}" || -z "${pod_port}" ) ]]; then
  echo "error: real RunPod launch requires --pod-host and --pod-port" >&2
  exit 1
fi

preflight_json="$("${preflight_script}" --profile "${profile_id}")"
profile_json="$(jq -c --arg profile_id "${profile_id}" '.profiles[] | select(.profile_id == $profile_id)' "${launch_file}")"
if [[ -z "${profile_json}" ]]; then
  echo "error: launch profile ${profile_id} was not found" >&2
  exit 1
fi

workspace_root="$(jq -r '.workspace_root' "${launch_file}")"
default_branch="$(jq -r '.default_branch' "${launch_file}")"
descriptor_uri="$(jq -r '.default_input_package_descriptor_uri' "${launch_file}")"
repo_clone_url="$(jq -r '.repo_clone_url' "${launch_file}")"
run_root="${workspace_root}/parameter-golf-runpod/${run_id}"
if [[ -z "${repo_dir}" ]]; then
  repo_dir="${workspace_root}/psionic"
fi
submission_dir="${run_root}/exported_submission/records/track_non_record_16mb/2026-03-18_psionic_local_reference_runtime_replay_v2"
manifest_path="${run_root}/parameter_golf_runpod_8xh100_launch_manifest.json"
launch_receipt_path="${run_root}/parameter_golf_runpod_8xh100_launch_receipt.json"
remote_preflight_log_path="${run_root}/remote_preflight.log"
pre_training_log_path="${run_root}/pre_training.log"
execution_log_path="${run_root}/execution.log"
finalizer_log_path="${run_root}/finalizer.log"

manifest_json="$(
jq -n \
  --arg schema_version "parameter_golf.runpod_8xh100_launch_manifest.v1" \
  --arg created_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg run_id "${run_id}" \
  --arg profile_id "${profile_id}" \
  --arg profile_label "$(jq -r '.profile_label' <<<"${profile_json}")" \
  --arg trainer_lane_id "$(jq -r '.trainer_lane_id' <<<"${profile_json}")" \
  --arg expected_execution_backend "$(jq -r '.expected_execution_backend' <<<"${profile_json}")" \
  --arg branch "${default_branch}" \
  --arg repo_clone_url "${repo_clone_url}" \
  --arg workspace_root "${workspace_root}" \
  --arg pod_host "${pod_host}" \
  --arg pod_port "${pod_port}" \
  --arg ssh_user "${ssh_user}" \
  --arg ssh_key_path "${ssh_key_path}" \
  --arg repo_dir "${repo_dir}" \
  --arg descriptor_uri "${descriptor_uri}" \
  --arg descriptor_artifact_ref "$(jq -r '.default_input_package_descriptor_artifact_ref' "${launch_file}")" \
  --arg submission_dir "${submission_dir}" \
  --arg run_root "${run_root}" \
  --arg manifest_path "${manifest_path}" \
  --arg launch_receipt_path "${launch_receipt_path}" \
  --arg remote_preflight_log_path "${remote_preflight_log_path}" \
  --arg pre_training_log_path "${pre_training_log_path}" \
  --arg execution_log_path "${execution_log_path}" \
  --arg finalizer_log_path "${finalizer_log_path}" \
  --arg pod_shape "$(jq -r '.pod_shape' <<<"${profile_json}")" \
  --arg runtime_image_posture "$(jq -r '.runtime_image_posture' <<<"${profile_json}")" \
  --arg workspace_mount_posture "$(jq -r '.workspace_mount_posture' <<<"${profile_json}")" \
  --arg accelerator_type "$(jq -r '.accelerator_type' <<<"${profile_json}")" \
  --argjson accelerator_count "$(jq -r '.accelerator_count' <<<"${profile_json}")" \
  --argjson require_non_mig "$(jq -r '.require_non_mig' <<<"${profile_json}")" \
  --argjson world_size "$(jq -r '.world_size' <<<"${profile_json}")" \
  --argjson grad_accum_steps "$(jq -r '.grad_accum_steps' <<<"${profile_json}")" \
  --arg pre_training_command "$(jq -r '.pre_training_command' <<<"${profile_json}")" \
  --arg execution_entrypoint_command "$(jq -r '.execution_entrypoint_command' <<<"${profile_json}")" \
  --arg finalizer_command "$(jq -r '.finalizer_command' <<<"${profile_json}")" \
  --argjson expected_receipt_paths "$(jq -c '.expected_receipt_paths' <<<"${profile_json}")" \
  --argjson preflight "${preflight_json}" \
  --arg manifest_only "${manifest_only}" \
  --arg stop_after "${stop_after}" \
  '{
    schema_version: $schema_version,
    created_at_utc: $created_at_utc,
    provider: "runpod",
    run_id: $run_id,
    profile_id: $profile_id,
    profile_label: $profile_label,
    trainer_lane_id: $trainer_lane_id,
    expected_execution_backend: $expected_execution_backend,
    git_ref: {
      repo_clone_url: $repo_clone_url,
      branch: $branch
    },
    operator_endpoint: {
      ssh_user: $ssh_user,
      pod_host: (if $pod_host == "" then null else $pod_host end),
      pod_port: (if $pod_port == "" then null else ($pod_port | tonumber) end),
      ssh_key_path: (if $ssh_key_path == "" then null else $ssh_key_path end)
    },
    launcher: {
      manifest_only: ($manifest_only == "true"),
      stop_after_phase: (if $manifest_only == "true" then null else $stop_after end)
    },
    machine: {
      pod_shape: $pod_shape,
      runtime_image_posture: $runtime_image_posture,
      workspace_mount_posture: $workspace_mount_posture
    },
    topology: {
      accelerator_type: $accelerator_type,
      accelerator_count: $accelerator_count,
      require_non_mig: $require_non_mig,
      world_size: $world_size,
      grad_accum_steps: $grad_accum_steps
    },
    input_package: {
      descriptor_uri: $descriptor_uri,
      descriptor_artifact_ref: $descriptor_artifact_ref
    },
    run_roots: {
      workspace_root: $workspace_root,
      repo_dir: $repo_dir,
      run_root: $run_root,
      submission_dir: $submission_dir
    },
    commands: {
      remote_preflight_command: "mkdir -p \"$PGOLF_RUN_ROOT\" && printf \"repo_dir=%s\\nsubmission_dir=%s\\n\" \"$PGOLF_REPO_DIR\" \"$PGOLF_SUBMISSION_DIR\" && command -v bash && command -v git && if command -v cargo >/dev/null 2>&1; then cargo --version; elif [[ -x \"$HOME/.cargo/bin/cargo\" ]]; then \"$HOME/.cargo/bin/cargo\" --version; else exit 1; fi && python3 --version && command -v nvidia-smi && nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader",
      pre_training_command: $pre_training_command,
      execution_entrypoint_command: $execution_entrypoint_command,
      finalizer_command: $finalizer_command
    },
    retained_paths: {
      manifest_path: $manifest_path,
      launch_receipt_path: $launch_receipt_path,
      remote_preflight_log_path: $remote_preflight_log_path,
      pre_training_log_path: $pre_training_log_path,
      execution_log_path: $execution_log_path,
      finalizer_log_path: $finalizer_log_path
    },
    expected_receipt_paths: $expected_receipt_paths,
    preflight: $preflight,
    manifest_only: ($manifest_only == "true")
  }'
)"

if [[ "${manifest_only}" == "true" ]]; then
  printf '%s\n' "${manifest_json}"
  exit 0
fi

ssh_cmd=(ssh -o StrictHostKeyChecking=no -p "${pod_port}")
if [[ -n "${ssh_key_path}" ]]; then
  ssh_cmd+=(-i "${ssh_key_path}")
fi
ssh_cmd+=("${ssh_user}@${pod_host}")

phase_results='[]'
launch_status="completed"
failed_phase=""
failed_exit_code=0
manifest_json_b64="$(printf '%s' "${manifest_json}" | base64 -w0)"

append_phase_result() {
  local phase_id="$1"
  local command="$2"
  local log_path="$3"
  local started_at_utc="$4"
  local finished_at_utc="$5"
  local exit_code="$6"
  local status="$7"

  phase_results="$(
    jq -c \
      --arg phase_id "${phase_id}" \
      --arg command "${command}" \
      --arg log_path "${log_path}" \
      --arg started_at_utc "${started_at_utc}" \
      --arg finished_at_utc "${finished_at_utc}" \
      --argjson exit_code "${exit_code}" \
      --arg status "${status}" \
      '. + [{
        phase_id: $phase_id,
        command: $command,
        log_path: $log_path,
        started_at_utc: $started_at_utc,
        finished_at_utc: $finished_at_utc,
        exit_code: $exit_code,
        status: $status
      }]' <<<"${phase_results}"
  )"
}

run_remote_phase() {
  local phase_id="$1"
  local phase_command="$2"
  local log_path="$3"
  local started_at_utc finished_at_utc exit_code status workdir phase_command_b64

  if [[ "${phase_id}" == "remote_preflight" ]]; then
    workdir="${run_root}"
  else
    workdir="${repo_dir}"
  fi

  started_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  phase_command_b64="$(printf '%s' "${phase_command}" | base64 -w0)"
  set +e
  "${ssh_cmd[@]}" bash -s -- \
    "${phase_id}" \
    "${log_path}" \
    "${repo_dir}" \
    "${repo_clone_url}" \
    "${default_branch}" \
    "${run_root}" \
    "${submission_dir}" \
    "${manifest_path}" \
    "${manifest_json_b64}" \
    "${phase_command_b64}" \
    "${workdir}" <<'REMOTE'
set -euo pipefail

phase_id="$1"
phase_log="$2"
repo_dir="$3"
repo_clone_url="$4"
repo_branch="$5"
run_root="$6"
submission_dir="$7"
manifest_path="$8"
manifest_b64="$9"
phase_command_b64="${10}"
phase_workdir="${11}"
phase_command="$(printf '%s' "${phase_command_b64}" | base64 -d)"

mkdir -p "${run_root}" "$(dirname -- "${repo_dir}")"
printf '%s' "${manifest_b64}" | base64 -d > "${manifest_path}"

if ! command -v cargo >/dev/null 2>&1 && [[ -d "${HOME}/.cargo/bin" ]]; then
  export PATH="${HOME}/.cargo/bin:${PATH}"
fi

if [[ "${phase_id}" != "remote_preflight" && ! -d "${repo_dir}/.git" ]]; then
  git clone "${repo_clone_url}" "${repo_dir}"
fi

export PGOLF_REPO_DIR="${repo_dir}"
export PGOLF_BRANCH="${repo_branch}"
export PGOLF_RUN_ROOT="${run_root}"
export PGOLF_SUBMISSION_DIR="${submission_dir}"
export PGOLF_LAUNCH_MANIFEST="${manifest_path}"
export PGOLF_LAUNCH_RECEIPT="${run_root}/parameter_golf_runpod_8xh100_launch_receipt.json"

printf 'phase_start phase=%s started_at_utc=%s\n' "${phase_id}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${phase_log}"
(
  cd "${phase_workdir}"
  bash -lc "${phase_command}"
) 2>&1 | tee -a "${phase_log}"
phase_exit_code=${PIPESTATUS[0]}
printf 'phase_exit phase=%s exit_code=%s finished_at_utc=%s\n' "${phase_id}" "${phase_exit_code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${phase_log}"
exit "${phase_exit_code}"
REMOTE
  exit_code=$?
  set -e
  finished_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "${exit_code}" -eq 0 ]]; then
    status="completed"
  else
    status="failed"
  fi
  append_phase_result "${phase_id}" "${phase_command}" "${log_path}" "${started_at_utc}" "${finished_at_utc}" "${exit_code}" "${status}"
  return "${exit_code}"
}

remote_preflight_command="$(jq -r '.commands.remote_preflight_command' <<<"${manifest_json}")"
pre_training_command="$(jq -r '.commands.pre_training_command' <<<"${manifest_json}")"
execution_command="$(jq -r '.commands.execution_entrypoint_command' <<<"${manifest_json}")"
finalizer_command="$(jq -r '.commands.finalizer_command' <<<"${manifest_json}")"

if ! run_remote_phase "remote_preflight" "${remote_preflight_command}" "${remote_preflight_log_path}"; then
  launch_status="failed"
  failed_phase="remote_preflight"
  failed_exit_code="$(jq -r 'last.exit_code' <<<"${phase_results}")"
fi

if [[ "${launch_status}" == "completed" && "${stop_after}" != "remote_preflight" ]]; then
  if ! run_remote_phase "pre_training" "${pre_training_command}" "${pre_training_log_path}"; then
    launch_status="failed"
    failed_phase="pre_training"
    failed_exit_code="$(jq -r 'last.exit_code' <<<"${phase_results}")"
  fi
fi

if [[ "${launch_status}" == "completed" && ( "${stop_after}" == "execution" || "${stop_after}" == "finalize" ) ]]; then
  if ! run_remote_phase "execution" "${execution_command}" "${execution_log_path}"; then
    launch_status="failed"
    failed_phase="execution"
    failed_exit_code="$(jq -r 'last.exit_code' <<<"${phase_results}")"
  fi
fi

if [[ "${launch_status}" == "completed" && "${stop_after}" == "finalize" ]]; then
  if ! run_remote_phase "finalize" "${finalizer_command}" "${finalizer_log_path}"; then
    launch_status="failed"
    failed_phase="finalize"
    failed_exit_code="$(jq -r 'last.exit_code' <<<"${phase_results}")"
  fi
fi

launch_receipt_json="$(
  jq -n \
    --arg schema_version "parameter_golf.runpod_8xh100_launch_receipt.v1" \
    --arg runner "scripts/parameter-golf-runpod-launch-8xh100.sh" \
    --arg created_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg status "${launch_status}" \
    --arg failed_phase "${failed_phase}" \
    --argjson failed_exit_code "${failed_exit_code}" \
    --argjson manifest "$(jq -c '.' <<<"${manifest_json}")" \
    --argjson phase_results "${phase_results}" \
    '{
      schema_version: $schema_version,
      runner: $runner,
      created_at_utc: $created_at_utc,
      status: $status,
      failed_phase: (if $failed_phase == "" then null else $failed_phase end),
      failed_exit_code: (if $failed_exit_code == 0 then null else $failed_exit_code end),
      manifest: $manifest,
      phases: $phase_results
    }'
)"

"${ssh_cmd[@]}" "mkdir -p $(printf '%q' "${run_root}") && cat > $(printf '%q' "${launch_receipt_path}")" <<<"${launch_receipt_json}"
printf '%s\n' "${launch_receipt_json}"

if [[ "${launch_status}" != "completed" ]]; then
  exit "${failed_exit_code}"
fi
