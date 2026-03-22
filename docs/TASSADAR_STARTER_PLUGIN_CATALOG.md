# Tassadar Starter Plugin Catalog

This document tracks the runtime-owned starter-plugin catalog bundle and the
downstream catalog, eval, summary, and provider-reporting surfaces above it.

The boundary is narrow on purpose:

- the starter catalog is operator-curated and operator-only
- each cataloged plugin carries an explicit descriptor, fixture bundle, and
  sample mount-envelope sidecar
- capability class, replay posture, trust tier, and negative claims stay
  machine-legible
- the catalog does not imply public plugin publication, arbitrary external
  plugin admission, or a public plugin marketplace

## Implemented

- runtime bundle:
  `fixtures/tassadar/runs/tassadar_post_article_starter_plugin_catalog_v1/tassadar_post_article_starter_plugin_catalog_bundle.json`
- catalog report:
  `fixtures/tassadar/reports/tassadar_post_article_starter_plugin_catalog_report.json`
- eval report:
  `fixtures/tassadar/reports/tassadar_post_article_starter_plugin_catalog_eval_report.json`
- research summary:
  `fixtures/tassadar/reports/tassadar_post_article_starter_plugin_catalog_summary.json`
- checker:
  `scripts/check-tassadar-post-article-starter-plugin-catalog.sh`

The current starter catalog freezes five bounded starter-plugin entries:

- `plugin.text.url_extract`
- `plugin.text.stats`
- `plugin.http.fetch_text`
- `plugin.html.extract_readable`
- `plugin.feed.rss_atom_parse`

Capability posture stays explicit:

- four local deterministic entries
- one read-only network entry
- two bounded composition flows

`plugin.text.stats` is now part of the cataloged starter-plugin surface with
its own descriptor, fixture bundle, and capability-free sample mount envelope,
which means the repo can now point to one machine-legible path from runtime
plugin truth to catalog, eval, summary, and provider-report truth for a
user-added starter plugin.

## Planned

- controller-level composition with a user-added starter plugin lands
  separately in the controller issue wave
- weighted-controller admission for user-added starter plugins lands
  separately in the canonical weighted-lane issue
