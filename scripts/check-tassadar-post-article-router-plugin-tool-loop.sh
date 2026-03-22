#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-serve --example tassadar_post_article_router_plugin_tool_loop_pilot_bundle
cargo test -p psionic-router starter_plugin_tool_loop_ -- --nocapture
cargo test -p psionic-serve router_plugin_tool_loop_ -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article.router_plugin_tool_loop_pilot.bundle.v1"
  and ((.tool_definition_rows | length) == 4)
  and ((.case_rows | length) == 2)
  and (.case_rows | any(.case_id == "router_plugin_tool_loop_success" and .bounded_step_count_preserved == true and (.receipt_rows | length) == 5 and .served_seed.tool_call_names[0] == "plugin_text_url_extract"))
  and (.case_rows | any(.case_id == "router_plugin_tool_loop_fetch_refusal" and .typed_refusal_preserved == true and (.receipt_rows | any(.status == "refusal"))))
  and (.continuation_row.conversation_revision == 2)
  and (.continuation_row.replayed_prompt_messages > 0)
' fixtures/tassadar/runs/tassadar_post_article_router_plugin_tool_loop_pilot_v1/tassadar_post_article_router_plugin_tool_loop_pilot_bundle.json >/dev/null
