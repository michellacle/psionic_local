#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cargo run -q -p psionic-runtime --example psion_plugin_guest_artifact_runtime_loading

jq -e '
  .schema_version == "psionic.psion.plugin_guest_artifact_runtime_loading.v1"
  and .success_case.status == "loaded"
  and (.success_case.loaded_artifact.host_owned_capability_mediation == true)
  and (.success_case.loaded_artifact.publication_blocked == true)
  and (.refusal_cases | length) >= 3
' fixtures/psion/plugins/guest_artifact/psion_plugin_guest_artifact_runtime_loading_v1.json >/dev/null

echo "guest-artifact runtime loading bundle is present and bounded"
