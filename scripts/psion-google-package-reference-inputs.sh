#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
POLICY_FILE="${REPO_ROOT}/fixtures/psion/google/psion_google_reference_input_package_policy_v1.json"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

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

compute_sha256() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  else
    shasum -a 256 "${path}" | awk '{print $1}'
  fi
}

record_file_artifact() {
  local artifact_file="$1"
  local kind="$2"
  local package_path="$3"
  local overlay_target="${4:-}"
  local digest
  digest="$(compute_sha256 "${artifact_file}")"
  jq -nc \
    --arg artifact_id "${package_path}" \
    --arg kind "${kind}" \
    --arg package_path "${package_path}" \
    --arg overlay_target "${overlay_target}" \
    --arg sha256 "${digest}" \
    '{
      artifact_id: $artifact_id,
      kind: $kind,
      package_path: $package_path,
      overlay_target: (if $overlay_target == "" then null else $overlay_target end),
      sha256: $sha256
    }'
}

copy_path_into_package() {
  local src_path="$1"
  local package_root="$2"
  local list_file="$3"
  if [[ -d "${REPO_ROOT}/${src_path}" ]]; then
    while IFS= read -r file_path; do
      local relative_path="${file_path#${REPO_ROOT}/}"
      local dest_path="${package_root}/repo_overlay/${relative_path}"
      mkdir -p "$(dirname "${dest_path}")"
      cp "${file_path}" "${dest_path}"
      record_file_artifact "${dest_path}" "repo_overlay_file" "repo_overlay/${relative_path}" "${relative_path}" >> "${list_file}"
    done < <(find "${REPO_ROOT}/${src_path}" -type f | sort)
  else
    local dest_path="${package_root}/repo_overlay/${src_path}"
    mkdir -p "$(dirname "${dest_path}")"
    cp "${REPO_ROOT}/${src_path}" "${dest_path}"
    record_file_artifact "${dest_path}" "repo_overlay_file" "repo_overlay/${src_path}" "${src_path}" >> "${list_file}"
  fi
}

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

corpus_build_dir="${tmpdir}/reference_corpus"
pilot_build_dir="${tmpdir}/reference_pilot"
package_root="${tmpdir}/input_package"
artifact_list_file="${tmpdir}/artifact_list.jsonl"
package_manifest_file="${package_root}/psion_google_reference_input_package_manifest.json"
descriptor_file="${tmpdir}/psion_google_reference_input_package_descriptor.json"

bucket_url="$(jq -r '.bucket_url' "${POLICY_FILE}")"
descriptor_uri="$(jq -r '.descriptor_uri' "${POLICY_FILE}")"
archive_prefix="$(jq -r '.archive_prefix' "${POLICY_FILE}")"
package_id_prefix="$(jq -r '.package_id_prefix' "${POLICY_FILE}")"
materialization_mode="$(jq -r '.materialization.mode' "${POLICY_FILE}")"
max_local_working_set_gb="$(jq -r '.materialization.max_local_working_set_gb' "${POLICY_FILE}")"
overlay_target="$(jq -r '.materialization.overlay_target' "${POLICY_FILE}")"
git_revision="$(git rev-parse HEAD)"
created_at_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
package_id="${package_id_prefix}-$(git rev-parse --short=12 HEAD)-$(date -u '+%Y%m%dt%H%M%Sz' | tr '[:upper:]' '[:lower:]')"

mkdir -p "${package_root}/repo_overlay" "${package_root}/derived/reference_corpus" "${package_root}/derived/reference_pilot"
: > "${artifact_list_file}"

cargo run -p psionic-data --example psion_reference_corpus_build -- "${corpus_build_dir}" >/dev/null
cargo run -p psionic-train --example psion_reference_pilot -- "${pilot_build_dir}" >/dev/null

while IFS= read -r overlay_path; do
  copy_path_into_package "${overlay_path}" "${package_root}" "${artifact_list_file}"
done < <(jq -r '.repo_overlay_paths[]' "${POLICY_FILE}")

while IFS= read -r file_path; do
  relative_name="$(basename "${file_path}")"
  dest_path="${package_root}/derived/reference_corpus/${relative_name}"
  cp "${file_path}" "${dest_path}"
  record_file_artifact "${dest_path}" "derived_reference_corpus_file" "derived/reference_corpus/${relative_name}" >> "${artifact_list_file}"
done < <(find "${corpus_build_dir}" -maxdepth 2 -type f | sort)

stage_config_src="${pilot_build_dir}/psion_reference_pilot_stage_config.json"
stage_config_dest="${package_root}/derived/reference_pilot/psion_reference_pilot_stage_config.json"
cp "${stage_config_src}" "${stage_config_dest}"
record_file_artifact "${stage_config_dest}" "derived_stage_config" "derived/reference_pilot/psion_reference_pilot_stage_config.json" >> "${artifact_list_file}"

artifacts_json="$(jq -s '.' "${artifact_list_file}")"
raw_source_manifest_sha256="$(compute_sha256 "${package_root}/derived/reference_corpus/psion_raw_source_manifest_v1.json")"
tokenizer_bundle_sha256="$(compute_sha256 "${package_root}/derived/reference_corpus/psion_tokenizer_artifact_bundle_v1.json")"
tokenized_corpus_manifest_sha256="$(compute_sha256 "${package_root}/derived/reference_corpus/psion_tokenized_corpus_manifest_v1.json")"
stage_config_sha256="$(compute_sha256 "${stage_config_dest}")"
benchmark_catalog_sha256="$(compute_sha256 "${package_root}/repo_overlay/fixtures/psion/benchmarks/psion_benchmark_catalog_v1.json")"
acceptance_matrix_sha256="$(compute_sha256 "${package_root}/repo_overlay/fixtures/psion/acceptance/psion_acceptance_matrix_v1.json")"

jq -n \
  --arg schema_version "psion.google_training_input_package_manifest.v1" \
  --arg package_id "${package_id}" \
  --arg created_at_utc "${created_at_utc}" \
  --arg git_revision "${git_revision}" \
  --arg repo_clone_url "https://github.com/OpenAgentsInc/psionic.git" \
  --arg materialization_mode "${materialization_mode}" \
  --arg overlay_target "${overlay_target}" \
  --argjson max_local_working_set_gb "${max_local_working_set_gb}" \
  --argjson artifacts "${artifacts_json}" \
  --argjson benchmark_execution_posture "$(jq '.benchmark_execution_posture' "${POLICY_FILE}")" \
  --arg raw_source_manifest_sha256 "${raw_source_manifest_sha256}" \
  --arg tokenizer_bundle_sha256 "${tokenizer_bundle_sha256}" \
  --arg tokenized_corpus_manifest_sha256 "${tokenized_corpus_manifest_sha256}" \
  --arg stage_config_sha256 "${stage_config_sha256}" \
  --arg benchmark_catalog_sha256 "${benchmark_catalog_sha256}" \
  --arg acceptance_matrix_sha256 "${acceptance_matrix_sha256}" \
  '{
    schema_version: $schema_version,
    package_id: $package_id,
    created_at_utc: $created_at_utc,
    git_revision: $git_revision,
    repo_clone_url: $repo_clone_url,
    materialization: {
      mode: $materialization_mode,
      max_local_working_set_gb: $max_local_working_set_gb,
      overlay_target: $overlay_target
    },
    benchmark_execution_posture: $benchmark_execution_posture,
    key_digests: {
      raw_source_manifest_sha256: $raw_source_manifest_sha256,
      tokenizer_bundle_sha256: $tokenizer_bundle_sha256,
      tokenized_corpus_manifest_sha256: $tokenized_corpus_manifest_sha256,
      stage_config_sha256: $stage_config_sha256,
      benchmark_catalog_sha256: $benchmark_catalog_sha256,
      acceptance_matrix_sha256: $acceptance_matrix_sha256
    },
    artifacts: $artifacts
  }' > "${package_manifest_file}"

archive_path="${tmpdir}/${package_id}.tar.gz"
(
  cd "${package_root}"
  tar -czf "${archive_path}" .
)
archive_sha256="$(compute_sha256 "${archive_path}")"
manifest_sha256="$(compute_sha256 "${package_manifest_file}")"
archive_uri="${archive_prefix}/${package_id}.tar.gz"

jq -n \
  --arg schema_version "psion.google_training_input_package_descriptor.v1" \
  --arg package_id "${package_id}" \
  --arg created_at_utc "${created_at_utc}" \
  --arg git_revision "${git_revision}" \
  --arg descriptor_uri "${descriptor_uri}" \
  --arg archive_uri "${archive_uri}" \
  --arg archive_sha256 "${archive_sha256}" \
  --arg manifest_path_in_archive "psion_google_reference_input_package_manifest.json" \
  --arg manifest_sha256 "${manifest_sha256}" \
  --arg materialization_mode "${materialization_mode}" \
  --argjson max_local_working_set_gb "${max_local_working_set_gb}" \
  --argjson benchmark_execution_posture "$(jq '.benchmark_execution_posture' "${POLICY_FILE}")" \
  --argjson key_digests "$(jq '.key_digests' "${package_manifest_file}")" \
  '{
    schema_version: $schema_version,
    package_id: $package_id,
    created_at_utc: $created_at_utc,
    git_revision: $git_revision,
    descriptor_uri: $descriptor_uri,
    archive_uri: $archive_uri,
    archive_sha256: $archive_sha256,
    manifest_path_in_archive: $manifest_path_in_archive,
    manifest_sha256: $manifest_sha256,
    materialization: {
      mode: $materialization_mode,
      max_local_working_set_gb: $max_local_working_set_gb
    },
    benchmark_execution_posture: $benchmark_execution_posture,
    key_digests: $key_digests
  }' > "${descriptor_file}"

gcloud storage cp --quiet "${archive_path}" "${archive_uri}" >/dev/null
gcloud storage cp --quiet "${descriptor_file}" "${descriptor_uri}" >/dev/null
wait_for_object "${archive_uri}"
wait_for_object "${descriptor_uri}"

echo "input package descriptor:"
cat "${descriptor_file}"
