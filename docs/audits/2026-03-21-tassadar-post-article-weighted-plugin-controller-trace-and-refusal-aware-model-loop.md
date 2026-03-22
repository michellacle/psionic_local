# Tassadar Post-Article Weighted Plugin Controller Trace And Refusal-Aware Model Loop

## Scope

`TAS-204` closes the bounded weighted plugin controller trace on the canonical
post-article machine identity.

The canonical machine-readable artifacts are:

- `fixtures/tassadar/runs/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_v1/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_bundle.json`
- `fixtures/tassadar/reports/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_report.json`
- `fixtures/tassadar/reports/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_eval_report.json`
- `fixtures/tassadar/reports/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_summary.json`
- `crates/psionic-provider/src/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop.rs`
- `scripts/check-tassadar-post-article-weighted-plugin-controller-trace-and-refusal-aware-model-loop.sh`

## What Is Now Frozen

The bounded post-article plugin lane now has one explicit weighted controller
trace that:

- keeps plugin selection, export selection, and packet-argument construction
  model-owned
- keeps multi-step continuation, retry, refusal, and stop conditions
  model-owned
- returns typed refusals back into the model loop instead of hiding them behind
  runtime retry or downgrade behavior
- makes determinism class, sampling policy, temperature, randomness, and
  external-signal boundaries explicit
- keeps the host on validate-and-execute duties without letting it become the
  planner
- keeps hidden host sequencing, auto-retry, fallback export selection,
  heuristic ranking, schema auto-repair, cached result substitution, candidate
  precomputation, hidden top-k filtering, helper substitution, and runtime
  policy drift on explicit blocked lanes
- now also admits one shared-registry user-added capability-free starter
  plugin, `plugin.text.stats`, through one bounded admission row and one
  explicit model-selected success trace on the canonical weighted lane

## What This Still Does Not Claim

This tranche does not claim:

- plugin authority or trust-tier promotion
- plugin publication rights
- served or public plugin universality
- arbitrary public software execution

The remaining frontier moves to `TAS-205`, which is where authority,
promotion, publication, and trust-tier gates must be made true.
