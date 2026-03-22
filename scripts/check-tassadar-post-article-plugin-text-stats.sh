#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_plugin_text_stats_bundle
cargo test -p psionic-runtime text_stats_ -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article.plugin_text_stats.runtime_bundle.v1"
  and .plugin_id == "plugin.text.stats"
  and .plugin_version == "v1"
  and .packet_abi_version == "packet.v1"
  and .mount_envelope_id == "mount.plugin.text.stats.no_capabilities.v1"
  and .tool_projection.tool_name == "plugin_text_stats"
  and ((.negative_claim_ids | length) == 4)
  and ((.case_rows | length) == 4)
  and (.case_rows | any(.case_id == "text_stats_success" and .status == "exact_success"))
  and (.case_rows | any(.case_id == "schema_invalid_missing_text" and .response_or_refusal_schema_id == "plugin.refusal.schema_invalid.v1"))
  and (.case_rows | any(.case_id == "packet_too_large_refusal" and .response_or_refusal_schema_id == "plugin.refusal.packet_too_large.v1"))
  and (.case_rows | any(.case_id == "unsupported_codec_refusal" and .response_or_refusal_schema_id == "plugin.refusal.unsupported_codec.v1"))
' fixtures/tassadar/runs/tassadar_post_article_plugin_text_stats_v1/tassadar_post_article_plugin_text_stats_bundle.json >/dev/null
