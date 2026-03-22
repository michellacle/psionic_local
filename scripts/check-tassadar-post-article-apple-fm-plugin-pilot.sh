#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-apple-fm --example tassadar_post_article_apple_fm_plugin_session_pilot_bundle
cargo test -p psionic-apple-fm starter_plugin_apple_fm_ -- --nocapture
cargo test -p psionic-apple-fm apple_fm_plugin_session_ -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article.apple_fm_plugin_session_pilot.bundle.v1"
  and ((.tool_definition_rows | length) == 4)
  and ((.case_rows | length) == 2)
  and (.case_rows | any(.case_id == "apple_fm_plugin_session_success" and (.step_rows | length) == 5 and .session_token_binding_preserved == true))
  and (.case_rows | any(.case_id == "apple_fm_plugin_session_fetch_refusal" and .typed_refusal_preserved == true and (.step_rows | any(.projected_result.status == "refusal"))))
' fixtures/tassadar/runs/tassadar_post_article_apple_fm_plugin_session_pilot_v1/tassadar_post_article_apple_fm_plugin_session_pilot_bundle.json >/dev/null
