#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
POLICY_FILE="${REPO_ROOT}/fixtures/psion/google/psion_google_checkpoint_archive_policy_v1.json"
PILOT_OUTPUT_DIR="${REPO_ROOT}/target/psion_reference_pilot_bundle"
MANIFEST_OUT=""

usage() {
  cat <<'EOF'
Usage: psion-google-archive-reference-pilot-checkpoint.sh [options] [pilot_output_dir]

Options:
  --manifest-out <path>      Write the generated archive manifest to one local path.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest-out)
      MANIFEST_OUT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      PILOT_OUTPUT_DIR="$1"
      shift
      ;;
  esac
done

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

compute_sha256() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  else
    shasum -a 256 "${path}" | awk '{print $1}'
  fi
}

wait_for_object() {
  local object_path="$1"
  local attempt
  for attempt in 1 2 3 4 5; do
    if gcloud storage ls "${object_path}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "error: object ${object_path} did not become visible" >&2
  exit 1
}

sanitize_component() {
  sed -E 's/[^A-Za-z0-9.-]+/-/g; s/^-+//; s/-+$//' <<<"$1"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "error: required file not found: ${path}" >&2
    exit 1
  fi
}

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

if [[ ! -d "${PILOT_OUTPUT_DIR}" ]]; then
  echo "error: pilot output directory not found: ${PILOT_OUTPUT_DIR}" >&2
  exit 1
fi

checkpoint_file="${PILOT_OUTPUT_DIR}/psion_reference_pilot_checkpoint.safetensors"
checkpoint_manifest_file="${PILOT_OUTPUT_DIR}/psion_reference_pilot_checkpoint_manifest.json"
optimizer_state_file="${PILOT_OUTPUT_DIR}/psion_reference_pilot_optimizer_state.json"
stage_receipt_file="${PILOT_OUTPUT_DIR}/psion_reference_pilot_stage_receipt.json"
observability_receipt_file="${PILOT_OUTPUT_DIR}/psion_reference_pilot_observability_receipt.json"

require_file "${checkpoint_file}"
require_file "${checkpoint_manifest_file}"
require_file "${optimizer_state_file}"
require_file "${stage_receipt_file}"
require_file "${observability_receipt_file}"

project_id="$(jq -r '.project_id' "${POLICY_FILE}")"
bucket_url="$(jq -r '.bucket_url' "${POLICY_FILE}")"
archive_prefix="$(jq -r '.checkpoint_archive_prefix' "${POLICY_FILE}")"
archive_manifest_name="$(jq -r '.archive_manifest_name' "${POLICY_FILE}")"
input_package_descriptor_uri="$(jq -r '.input_package_descriptor_uri' "${POLICY_FILE}")"
recovery_mode="$(jq -r '.recovery_mode' "${POLICY_FILE}")"
storage_profile_json="$(jq '.storage_profile' "${POLICY_FILE}")"
created_at_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
git_revision="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
git_revision_short="$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)"

run_id="$(jq -r '.run_id' "${stage_receipt_file}")"
stage_id="$(jq -r '.stage_id' "${stage_receipt_file}")"
checkpoint_ref="$(jq -r '.checkpoint_ref' "${checkpoint_manifest_file}")"
checkpoint_family="$(jq -r '.checkpoint_family' "${checkpoint_manifest_file}")"
checkpoint_lineage_digest="$(jq -r '.checkpoint_lineage.checkpoint_lineage_digest' "${stage_receipt_file}")"
checkpoint_object_digest="$(jq -r '.checkpoint_lineage.promoted_checkpoint.object_digest' "${stage_receipt_file}")"
checkpoint_manifest_digest="$(jq -r '.checkpoint_lineage.promoted_checkpoint.manifest_digest' "${stage_receipt_file}")"
stage_receipt_digest="$(jq -r '.receipt_digest' "${stage_receipt_file}")"
observability_digest="$(jq -r '.observability_digest' "${observability_receipt_file}")"

checkpoint_ref_component="$(sanitize_component "${checkpoint_ref}")"
archive_id="psion-google-reference-checkpoint-${git_revision_short}-$(date -u '+%Y%m%dt%H%M%Sz' | tr '[:upper:]' '[:lower:]')"
checkpoint_prefix="${archive_prefix}/${run_id}/${checkpoint_ref_component}"
archive_manifest_uri="${checkpoint_prefix}/archive/${archive_manifest_name}"
artifact_records_file="${tmpdir}/artifact_records.jsonl"
: > "${artifact_records_file}"

upload_artifact() {
  local artifact_kind="$1"
  local local_path="$2"
  local remote_path="$3"
  local sha256
  sha256="$(compute_sha256 "${local_path}")"
  gcloud storage cp --quiet "${local_path}" "${remote_path}" >/dev/null
  wait_for_object "${remote_path}"
  jq -nc \
    --arg artifact_kind "${artifact_kind}" \
    --arg local_path "${local_path}" \
    --arg remote_uri "${remote_path}" \
    --arg sha256 "${sha256}" \
    --arg byte_length "$(wc -c < "${local_path}" | tr -d ' ')" \
    '{
      artifact_kind: $artifact_kind,
      local_path: $local_path,
      remote_uri: $remote_uri,
      sha256: $sha256,
      byte_length: ($byte_length | tonumber)
    }' >> "${artifact_records_file}"
}

upload_artifact "checkpoint_weights" "${checkpoint_file}" "${checkpoint_prefix}/dense/psion_reference_pilot_checkpoint.safetensors"
upload_artifact "checkpoint_manifest" "${checkpoint_manifest_file}" "${checkpoint_prefix}/manifests/psion_reference_pilot_checkpoint_manifest.json"
upload_artifact "optimizer_state" "${optimizer_state_file}" "${checkpoint_prefix}/manifests/psion_reference_pilot_optimizer_state.json"
upload_artifact "stage_receipt" "${stage_receipt_file}" "${checkpoint_prefix}/receipts/psion_reference_pilot_stage_receipt.json"
upload_artifact "observability_receipt" "${observability_receipt_file}" "${checkpoint_prefix}/receipts/psion_reference_pilot_observability_receipt.json"

artifacts_json="$(jq -s '.' "${artifact_records_file}")"
archive_manifest_file="${tmpdir}/${archive_manifest_name}"

jq -n \
  --arg schema_version "psion.google_reference_checkpoint_archive_manifest.v1" \
  --arg archive_id "${archive_id}" \
  --arg created_at_utc "${created_at_utc}" \
  --arg project_id "${project_id}" \
  --arg bucket_url "${bucket_url}" \
  --arg checkpoint_prefix "${checkpoint_prefix}" \
  --arg archive_manifest_uri "${archive_manifest_uri}" \
  --arg repo_git_revision "${git_revision}" \
  --arg input_package_descriptor_uri "${input_package_descriptor_uri}" \
  --arg run_id "${run_id}" \
  --arg stage_id "${stage_id}" \
  --arg checkpoint_ref "${checkpoint_ref}" \
  --arg checkpoint_family "${checkpoint_family}" \
  --arg checkpoint_lineage_digest "${checkpoint_lineage_digest}" \
  --arg checkpoint_object_digest "${checkpoint_object_digest}" \
  --arg checkpoint_manifest_digest "${checkpoint_manifest_digest}" \
  --arg stage_receipt_digest "${stage_receipt_digest}" \
  --arg observability_digest "${observability_digest}" \
  --arg recovery_mode "${recovery_mode}" \
  --argjson storage_profile "${storage_profile_json}" \
  --argjson artifacts "${artifacts_json}" \
  '{
    schema_version: $schema_version,
    archive_id: $archive_id,
    created_at_utc: $created_at_utc,
    project_id: $project_id,
    bucket_url: $bucket_url,
    checkpoint_prefix: $checkpoint_prefix,
    archive_manifest_uri: $archive_manifest_uri,
    repo_git_revision: $repo_git_revision,
    input_package_descriptor_uri: $input_package_descriptor_uri,
    run_id: $run_id,
    stage_id: $stage_id,
    checkpoint_ref: $checkpoint_ref,
    checkpoint_family: $checkpoint_family,
    checkpoint_lineage_digest: $checkpoint_lineage_digest,
    checkpoint_object_digest: $checkpoint_object_digest,
    checkpoint_manifest_digest: $checkpoint_manifest_digest,
    stage_receipt_digest: $stage_receipt_digest,
    observability_digest: $observability_digest,
    recovery_mode: $recovery_mode,
    local_disk_authority: false,
    storage_profile: $storage_profile,
    artifacts: $artifacts,
    detail: "Reference pilot checkpoint archive stores weights, manifest, optimizer state, and bound stage receipts in GCS so the dense checkpoint can be restored under resume_from_last_stable_checkpoint posture."
  }' > "${archive_manifest_file}"

gcloud storage cp --quiet "${archive_manifest_file}" "${archive_manifest_uri}" >/dev/null
wait_for_object "${archive_manifest_uri}"

if [[ -n "${MANIFEST_OUT}" ]]; then
  cp "${archive_manifest_file}" "${MANIFEST_OUT}"
fi

echo "checkpoint archive manifest:"
cat "${archive_manifest_file}"
