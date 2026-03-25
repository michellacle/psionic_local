#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
POLICY_FILE="${POLICY_FILE:-${REPO_ROOT}/fixtures/psion/google/psion_google_two_node_swarm_impairment_policy_v1.json}"
ACTION="${ACTION:-apply}"
PROFILE_ID="${PROFILE_ID:-}"
HOST_ROLE="${HOST_ROLE:-}"
RUN_ID="${RUN_ID:-}"
INTERFACE_NAME="${INTERFACE_NAME:-}"
RECEIPT_OUT="${RECEIPT_OUT:-}"
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: psion-google-two-node-swarm-impair.sh [options]

Options:
  --action <apply|clear|show>         Operation to perform.
  --profile <profile_id>              Admitted impairment profile id.
  --host-role <coordinator|contributor>
                                      Local host role for role-specific asymmetric settings.
  --run-id <run_id>                   Run identifier to carry into the receipt.
  --interface <name>                  Override the default-route interface.
  --receipt-out <path>                Write one machine-readable receipt to this path.
  --dry-run                           Render the exact receipt and commands without mutating `tc`.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --action)
      ACTION="$2"
      shift 2
      ;;
    --profile)
      PROFILE_ID="$2"
      shift 2
      ;;
    --host-role)
      HOST_ROLE="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --interface)
      INTERFACE_NAME="$2"
      shift 2
      ;;
    --receipt-out)
      RECEIPT_OUT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

timestamp_utc() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

resolve_interface() {
  if [[ -n "${INTERFACE_NAME}" ]]; then
    printf '%s' "${INTERFACE_NAME}"
    return 0
  fi
  ip route show default | awk '/default/ {print $5; exit}'
}

require_root_if_needed() {
  if [[ "${DRY_RUN}" == "true" || "${ACTION}" == "show" ]]; then
    return 0
  fi
  if [[ "${EUID}" -ne 0 ]]; then
    echo "error: ${ACTION} requires root so the helper can call tc" >&2
    exit 1
  fi
}

ensure_tooling() {
  local required_commands=(jq hostname)
  if [[ -z "${INTERFACE_NAME}" || "${DRY_RUN}" != "true" || "${ACTION}" == "show" ]]; then
    required_commands+=(ip)
  fi
  if [[ "${DRY_RUN}" != "true" || "${ACTION}" == "show" ]]; then
    required_commands+=(tc)
  fi
  for required_command in "${required_commands[@]}"; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
      echo "error: ${required_command} is required" >&2
      exit 1
    fi
  done
}

build_netem_args() {
  local delay_ms="$1"
  local jitter_ms="$2"
  local loss_percent="$3"
  local duplicate_percent="$4"
  local reorder_percent="$5"
  local rate_mbit="$6"
  local args=()

  if [[ "${delay_ms}" != "0" || "${jitter_ms}" != "0" ]]; then
    args+=(delay "${delay_ms}ms")
    if [[ "${jitter_ms}" != "0" ]]; then
      args+=("${jitter_ms}ms" distribution normal)
    fi
  fi
  if [[ "${loss_percent}" != "0" && "${loss_percent}" != "0.0" ]]; then
    args+=(loss "${loss_percent}%")
  fi
  if [[ "${duplicate_percent}" != "0" && "${duplicate_percent}" != "0.0" ]]; then
    args+=(duplicate "${duplicate_percent}%")
  fi
  if [[ "${reorder_percent}" != "0" && "${reorder_percent}" != "0.0" ]]; then
    args+=(reorder "${reorder_percent}%")
  fi
  if [[ "${rate_mbit}" != "0" ]]; then
    args+=(rate "${rate_mbit}mbit")
  fi
  printf '%s\0' "${args[@]}"
}

apply_profile() {
  local interface_name="$1"
  local ports="$2"
  local delay_ms="$3"
  local jitter_ms="$4"
  local loss_percent="$5"
  local duplicate_percent="$6"
  local reorder_percent="$7"
  local rate_mbit="$8"
  local root_handle target_flowid netem_handle netem_args
  root_handle="$(jq -r '.qdisc_handles.root' "${POLICY_FILE}")"
  target_flowid="$(jq -r '.qdisc_handles.target_flowid' "${POLICY_FILE}")"
  netem_handle="$(jq -r '.qdisc_handles.netem' "${POLICY_FILE}")"

  tc qdisc del dev "${interface_name}" root 2>/dev/null || true
  tc qdisc replace dev "${interface_name}" root handle "${root_handle}" prio bands 4
  mapfile -d '' netem_args < <(build_netem_args "${delay_ms}" "${jitter_ms}" "${loss_percent}" "${duplicate_percent}" "${reorder_percent}" "${rate_mbit}")
  if [[ "${#netem_args[@]}" -gt 0 ]]; then
    tc qdisc replace dev "${interface_name}" parent "${target_flowid}" handle "${netem_handle}" netem "${netem_args[@]}"
  fi
  for port in ${ports}; do
    tc filter replace dev "${interface_name}" protocol ip parent "${root_handle}" prio 1 u32 \
      match ip protocol 6 0xff \
      match ip sport "${port}" 0xffff flowid "${target_flowid}"
    tc filter replace dev "${interface_name}" protocol ip parent "${root_handle}" prio 1 u32 \
      match ip protocol 6 0xff \
      match ip dport "${port}" 0xffff flowid "${target_flowid}"
  done
}

clear_profile() {
  local interface_name="$1"
  tc qdisc del dev "${interface_name}" root 2>/dev/null || true
}

emit_receipt() {
  local interface_name="$1"
  local interface_selection_mode="$2"
  local profile_json="$3"
  local role_parameters_json="$4"
  local affected_ports_json="$5"
  local command_lines_json="$6"
  local detail="$7"
  local qdisc_show="$8"
  local filter_show="$9"
  local host_internal_ipv4
  host_internal_ipv4=""
  if command -v ip >/dev/null 2>&1; then
    host_internal_ipv4="$(
      ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i + 1); exit}}'
    )"
  fi
  jq -n \
    --arg schema_version "psion.google_two_node_swarm_impairment_receipt.v1" \
    --arg created_at_utc "$(timestamp_utc)" \
    --arg action "${ACTION}" \
    --arg profile_id "${PROFILE_ID}" \
    --arg run_id "${RUN_ID}" \
    --arg host_role "${HOST_ROLE}" \
    --arg interface "${interface_name}" \
    --arg interface_selection_mode "${interface_selection_mode}" \
    --arg hostname_short "$(hostname -s)" \
    --arg hostname_fqdn "$(hostname -f 2>/dev/null || hostname)" \
    --arg host_internal_ipv4 "${host_internal_ipv4}" \
    --arg cluster_namespace "$(jq -r '.cluster_namespace' "${POLICY_FILE}")" \
    --arg detail "${detail}" \
    --arg qdisc_show "${qdisc_show}" \
    --arg filter_show "${filter_show}" \
    --argjson dry_run "${DRY_RUN}" \
    --argjson affected_ports "${affected_ports_json}" \
    --argjson validation_targets "$(jq '.validation_targets' <<<"${profile_json}")" \
    --argjson profile_parameters "${role_parameters_json}" \
    --argjson qdisc_handles "$(jq '.qdisc_handles' "${POLICY_FILE}")" \
    --argjson commands "${command_lines_json}" \
    '{
      schema_version: $schema_version,
      created_at_utc: $created_at_utc,
      action: $action,
      profile_id: $profile_id,
      run_id: (if $run_id == "" then null else $run_id end),
      host_role: $host_role,
      interface: $interface,
      interface_selection_mode: $interface_selection_mode,
      host_identity: {
        hostname_short: $hostname_short,
        hostname_fqdn: $hostname_fqdn,
        internal_ipv4: (if $host_internal_ipv4 == "" then null else $host_internal_ipv4 end)
      },
      cluster_namespace: $cluster_namespace,
      affected_ports: $affected_ports,
      validation_targets: $validation_targets,
      profile_parameters: $profile_parameters,
      qdisc_handles: $qdisc_handles,
      dry_run: $dry_run,
      commands: $commands,
      verification: {
        qdisc_show: (if $qdisc_show == "" then null else $qdisc_show end),
        filter_show: (if $filter_show == "" then null else $filter_show end)
      },
      detail: $detail
    }'
}

if [[ -z "${PROFILE_ID}" ]]; then
  echo "error: --profile is required" >&2
  exit 1
fi
if [[ -z "${HOST_ROLE}" ]]; then
  echo "error: --host-role is required" >&2
  exit 1
fi
if [[ "${HOST_ROLE}" != "coordinator" && "${HOST_ROLE}" != "contributor" ]]; then
  echo "error: host role must be coordinator or contributor" >&2
  exit 1
fi
if [[ "${ACTION}" != "apply" && "${ACTION}" != "clear" && "${ACTION}" != "show" ]]; then
  echo "error: action must be apply, clear, or show" >&2
  exit 1
fi

ensure_tooling
require_root_if_needed

profile_json="$(jq -c --arg profile_id "${PROFILE_ID}" '.profiles[] | select(.profile_id == $profile_id)' "${POLICY_FILE}")"
if [[ -z "${profile_json}" ]]; then
  echo "error: unknown impairment profile ${PROFILE_ID}" >&2
  exit 1
fi
role_parameters_json="$(jq -c --arg host_role "${HOST_ROLE}" '.role_parameters[$host_role]' <<<"${profile_json}")"
if [[ -z "${role_parameters_json}" || "${role_parameters_json}" == "null" ]]; then
  echo "error: profile ${PROFILE_ID} is missing role parameters for ${HOST_ROLE}" >&2
  exit 1
fi

interface_selection_mode="explicit"
resolved_interface="$(resolve_interface)"
if [[ -z "${INTERFACE_NAME}" ]]; then
  interface_selection_mode="default_route_device"
fi
if [[ -z "${resolved_interface}" ]]; then
  echo "error: failed to resolve the target interface" >&2
  exit 1
fi

affected_ports_json="$(jq -c '.cluster_ports' "${POLICY_FILE}")"
affected_ports_string="$(jq -r '.cluster_ports | join(" ")' "${POLICY_FILE}")"
delay_ms="$(jq -r '.delay_ms' <<<"${role_parameters_json}")"
jitter_ms="$(jq -r '.jitter_ms' <<<"${role_parameters_json}")"
loss_percent="$(jq -r '.loss_percent' <<<"${role_parameters_json}")"
duplicate_percent="$(jq -r '.duplicate_percent' <<<"${role_parameters_json}")"
reorder_percent="$(jq -r '.reorder_percent' <<<"${role_parameters_json}")"
rate_mbit="$(jq -r '.rate_mbit' <<<"${role_parameters_json}")"
profile_mode="$(jq -r '.mode' <<<"${profile_json}")"

command_lines_json="$(jq -nc \
  --arg interface "${resolved_interface}" \
  --arg ports "${affected_ports_string}" \
  --arg delay_ms "${delay_ms}" \
  --arg jitter_ms "${jitter_ms}" \
  --arg loss_percent "${loss_percent}" \
  --arg duplicate_percent "${duplicate_percent}" \
  --arg reorder_percent "${reorder_percent}" \
  --arg rate_mbit "${rate_mbit}" \
  '[
    "tc qdisc del dev " + $interface + " root",
    "tc qdisc replace dev " + $interface + " root handle 1: prio bands 4",
    "tc qdisc replace dev " + $interface + " parent 1:3 handle 30: netem delay " + $delay_ms + "ms " + $jitter_ms + "ms distribution normal loss " + $loss_percent + "% duplicate " + $duplicate_percent + "% reorder " + $reorder_percent + "% rate " + $rate_mbit + "mbit",
    "tc filter replace dev " + $interface + " protocol ip parent 1: prio 1 u32 match ip protocol 6 0xff match ip sport <cluster_port> 0xffff flowid 1:3",
    "tc filter replace dev " + $interface + " protocol ip parent 1: prio 1 u32 match ip protocol 6 0xff match ip dport <cluster_port> 0xffff flowid 1:3"
  ]'
)"

detail="$(jq -r '.detail' <<<"${profile_json}")"
qdisc_show=""
filter_show=""

case "${ACTION}" in
  show)
    qdisc_show="$(tc qdisc show dev "${resolved_interface}" 2>/dev/null || true)"
    filter_show="$(tc filter show dev "${resolved_interface}" parent "$(jq -r '.qdisc_handles.root' "${POLICY_FILE}")" 2>/dev/null || true)"
    detail="Current tc state for the Google two-node swarm impairment surface on ${resolved_interface}."
    ;;
  clear)
    detail="Cleared any Google two-node swarm impairment qdisc from ${resolved_interface} so the cluster ports return to baseline."
    if [[ "${DRY_RUN}" != "true" ]]; then
      clear_profile "${resolved_interface}"
      qdisc_show="$(tc qdisc show dev "${resolved_interface}" 2>/dev/null || true)"
      filter_show="$(tc filter show dev "${resolved_interface}" parent "$(jq -r '.qdisc_handles.root' "${POLICY_FILE}")" 2>/dev/null || true)"
    fi
    ;;
  apply)
    if [[ "${profile_mode}" == "clear" ]]; then
      detail="Applied the clean baseline by clearing any Google swarm impairment qdisc from ${resolved_interface}."
      if [[ "${DRY_RUN}" != "true" ]]; then
        clear_profile "${resolved_interface}"
        qdisc_show="$(tc qdisc show dev "${resolved_interface}" 2>/dev/null || true)"
        filter_show="$(tc filter show dev "${resolved_interface}" parent "$(jq -r '.qdisc_handles.root' "${POLICY_FILE}")" 2>/dev/null || true)"
      fi
    else
      if [[ "${DRY_RUN}" != "true" ]]; then
        apply_profile \
          "${resolved_interface}" \
          "${affected_ports_string}" \
          "${delay_ms}" \
          "${jitter_ms}" \
          "${loss_percent}" \
          "${duplicate_percent}" \
          "${reorder_percent}" \
          "${rate_mbit}"
        qdisc_show="$(tc qdisc show dev "${resolved_interface}")"
        filter_show="$(tc filter show dev "${resolved_interface}" parent "$(jq -r '.qdisc_handles.root' "${POLICY_FILE}")")"
      fi
    fi
    ;;
esac

receipt_json="$(emit_receipt \
  "${resolved_interface}" \
  "${interface_selection_mode}" \
  "${profile_json}" \
  "${role_parameters_json}" \
  "${affected_ports_json}" \
  "${command_lines_json}" \
  "${detail}" \
  "${qdisc_show}" \
  "${filter_show}")"

if [[ -n "${RECEIPT_OUT}" ]]; then
  mkdir -p "$(dirname "${RECEIPT_OUT}")"
  printf '%s\n' "${receipt_json}" > "${RECEIPT_OUT}"
fi

printf '%s\n' "${receipt_json}"
