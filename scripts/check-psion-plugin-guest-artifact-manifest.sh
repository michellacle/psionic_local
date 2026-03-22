#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cargo run -q -p psionic-runtime --example psion_plugin_guest_artifact_manifest

jq -e '
  .schema_version == "psionic.psion.plugin_guest_artifact_manifest.v1"
  and .packet_abi_version == "packet.v1"
  and .guest_export_name == "handle_packet"
  and (.artifact_digest | type == "string")
  and (.artifact_digest | length) == 64
  and .publication_posture == "operator_internal_only_publication_blocked"
' fixtures/psion/plugins/guest_artifact/psion_plugin_guest_artifact_manifest_v1.json >/dev/null

jq -e '
  .schema_version == "psionic.psion.plugin_guest_artifact_identity.v1"
  and .packet_abi_version == "packet.v1"
  and .guest_export_name == "handle_packet"
  and (.artifact_digest | type == "string")
  and (.artifact_digest | length) == 64
  and .publication_posture == "operator_internal_only_publication_blocked"
' fixtures/psion/plugins/guest_artifact/psion_plugin_guest_artifact_identity_v1.json >/dev/null

echo "guest-artifact manifest contract fixtures are present and bounded"
