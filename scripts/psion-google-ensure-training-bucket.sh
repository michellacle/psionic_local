#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
BUCKET_NAME="${BUCKET_NAME:-openagentsgemini-psion-train-us-central1}"
BUCKET_URL="gs://${BUCKET_NAME}"
LOCATION="${LOCATION:-us-central1}"
DEFAULT_STORAGE_CLASS="${DEFAULT_STORAGE_CLASS:-STANDARD}"
RETENTION_PERIOD="${RETENTION_PERIOD:-7d}"
SOFT_DELETE_DURATION="${SOFT_DELETE_DURATION:-14d}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
LIFECYCLE_FILE="${REPO_ROOT}/fixtures/psion/google/psion_google_training_bucket_lifecycle_v1.json"
STORAGE_PROFILE_FILE="${REPO_ROOT}/fixtures/psion/google/psion_google_training_storage_profile_v1.json"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

if gcloud storage buckets describe "${BUCKET_URL}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "bucket already exists: ${BUCKET_URL}"
else
  echo "creating bucket: ${BUCKET_URL}"
  gcloud storage buckets create "${BUCKET_URL}" \
    --project="${PROJECT_ID}" \
    --location="${LOCATION}" \
    --default-storage-class="${DEFAULT_STORAGE_CLASS}" \
    --public-access-prevention \
    --uniform-bucket-level-access \
    --retention-period="${RETENTION_PERIOD}" \
    --soft-delete-duration="${SOFT_DELETE_DURATION}"
fi

echo "applying bucket settings: ${BUCKET_URL}"
gcloud storage buckets update "${BUCKET_URL}" \
  --project="${PROJECT_ID}" \
  --default-storage-class="${DEFAULT_STORAGE_CLASS}" \
  --public-access-prevention \
  --uniform-bucket-level-access \
  --versioning \
  --retention-period="${RETENTION_PERIOD}" \
  --soft-delete-duration="${SOFT_DELETE_DURATION}" \
  --lifecycle-file="${LIFECYCLE_FILE}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

cp "${STORAGE_PROFILE_FILE}" "${tmpdir}/psion_google_training_storage_profile_v1.json"

for prefix in runs checkpoints receipts logs manifests; do
  printf 'Psion Google training bucket prefix: %s\n' "${prefix}" > "${tmpdir}/${prefix}.keep"
  gcloud storage cp --quiet "${tmpdir}/${prefix}.keep" "${BUCKET_URL}/${prefix}/.keep" >/dev/null
done

gcloud storage cp \
  --quiet \
  "${tmpdir}/psion_google_training_storage_profile_v1.json" \
  "${BUCKET_URL}/manifests/psion_google_training_storage_profile_v1.json" >/dev/null

echo "bucket summary:"
gcloud storage buckets describe "${BUCKET_URL}" --project="${PROJECT_ID}" --format=json | \
  jq '{
    name,
    location,
    default_storage_class,
    public_access_prevention,
    soft_delete_policy,
    retention_policy,
    versioning_enabled,
    uniform_bucket_level_access,
    lifecycle_config
  }'
