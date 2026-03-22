#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
REGION="${REGION:-us-central1}"
NETWORK="${NETWORK:-oa-lightning}"
SUBNETWORK="${SUBNETWORK:-oa-lightning-us-central1}"
ROUTER="${ROUTER:-oa-nat-router-us-central1}"
EXPECTED_NAT="${EXPECTED_NAT:-oa-nat-us-central1}"
FIREWALL_RULE="${FIREWALL_RULE:-oa-allow-psion-train-iap-ssh}"
TARGET_TAG="${TARGET_TAG:-psion-train-host}"
SOURCE_RANGE="${SOURCE_RANGE:-35.235.240.0/20}"
BUCKET_URL="${BUCKET_URL:-gs://openagentsgemini-psion-train-us-central1}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
NETWORK_POSTURE_FILE="${REPO_ROOT}/fixtures/psion/google/psion_google_network_posture_v1.json"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

wait_for_firewall_rule() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if gcloud compute firewall-rules describe "${FIREWALL_RULE}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "error: firewall rule ${FIREWALL_RULE} did not become visible" >&2
  exit 1
}

wait_for_manifest_upload() {
  local manifest_path="${BUCKET_URL}/manifests/psion_google_network_posture_v1.json"
  local attempt
  for attempt in 1 2 3 4 5; do
    if gcloud storage ls "${manifest_path}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "error: network posture manifest upload did not become visible" >&2
  exit 1
}

router_json="$(
  gcloud compute routers describe "${ROUTER}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --format=json
)"
router_network="$(jq -r '.network | split("/") | last' <<<"${router_json}")"
nat_name="$(jq -r '.nats[0].name // empty' <<<"${router_json}")"
source_subnets="$(jq -r '.nats[0].sourceSubnetworkIpRangesToNat // empty' <<<"${router_json}")"

if [[ "${router_network}" != "${NETWORK}" ]]; then
  echo "error: router ${ROUTER} is attached to ${router_network}, expected ${NETWORK}" >&2
  exit 1
fi

if [[ "${nat_name}" != "${EXPECTED_NAT}" ]]; then
  echo "error: router ${ROUTER} is missing expected NAT ${EXPECTED_NAT}" >&2
  exit 1
fi

if [[ "${source_subnets}" != "ALL_SUBNETWORKS_ALL_IP_RANGES" ]]; then
  echo "error: router ${ROUTER} does not NAT all subnetwork IP ranges" >&2
  exit 1
fi

if gcloud compute firewall-rules describe "${FIREWALL_RULE}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute firewall-rules update "${FIREWALL_RULE}" \
    --project="${PROJECT_ID}" \
    --priority=1000 \
    --allow=tcp:22 \
    --source-ranges="${SOURCE_RANGE}" \
    --target-tags="${TARGET_TAG}" >/dev/null
else
  gcloud compute firewall-rules create "${FIREWALL_RULE}" \
    --project="${PROJECT_ID}" \
    --network="${NETWORK}" \
    --direction=INGRESS \
    --priority=1000 \
    --allow=tcp:22 \
    --source-ranges="${SOURCE_RANGE}" \
    --target-tags="${TARGET_TAG}" \
    --description="IAP SSH for Psion single-node Google training hosts." >/dev/null
fi

gcloud storage cp --quiet \
  "${NETWORK_POSTURE_FILE}" \
  "${BUCKET_URL}/manifests/psion_google_network_posture_v1.json" >/dev/null

wait_for_firewall_rule
wait_for_manifest_upload

echo "network posture summary:"
gcloud compute firewall-rules describe "${FIREWALL_RULE}" --project="${PROJECT_ID}" --format=json | \
  jq '{name, network: (.network | split("/") | last), direction, priority, sourceRanges, allowed, targetTags}'

echo "nat summary:"
jq '{name, region, network: (.network | split("/") | last), nat: .nats[0].name, sourceSubnetworkIpRangesToNat: .nats[0].sourceSubnetworkIpRangesToNat}' <<<"${router_json}"
