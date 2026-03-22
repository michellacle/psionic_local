#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_plugin_feed_rss_atom_parse_bundle
cargo test -p psionic-runtime feed_parse_ -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article.plugin_feed_rss_atom_parse.runtime_bundle.v1"
  and .plugin_id == "plugin.feed.rss_atom_parse"
  and .plugin_version == "v1"
  and .packet_abi_version == "packet.v1"
  and .mount_envelope_id == "mount.plugin.feed.rss_atom_parse.no_capabilities.v1"
  and .tool_projection.tool_name == "plugin_feed_rss_atom_parse"
  and ((.negative_claim_ids | length) == 3)
  and ((.case_rows | length) == 5)
  and (.case_rows | any(.case_id == "rss_parse_success" and .status == "exact_success"))
  and (.case_rows | any(.case_id == "atom_parse_success" and .status == "exact_success"))
  and (.case_rows | any(.case_id == "schema_invalid_missing_feed_text" and .response_or_refusal_schema_id == "plugin.refusal.schema_invalid.v1"))
  and (.case_rows | any(.case_id == "unsupported_feed_format_refusal" and .response_or_refusal_schema_id == "plugin.refusal.unsupported_feed_format.v1"))
  and (.case_rows | any(.case_id == "input_too_large_refusal" and .response_or_refusal_schema_id == "plugin.refusal.input_too_large.v1"))
  and .composition_case.case_id == "fetch_then_parse_feed"
  and .composition_case.green == true
  and .composition_case.schema_repair_allowed == false
  and .composition_case.hidden_host_parsing_allowed == false
' fixtures/tassadar/runs/tassadar_post_article_plugin_feed_rss_atom_parse_v1/tassadar_post_article_plugin_feed_rss_atom_parse_bundle.json >/dev/null
