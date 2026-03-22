# Tassadar Post-Article Plugin Result-Binding, Schema-Stability, And Composition

## Scope

`TAS-203A` closes the bounded plugin result-binding, schema-stability, and
composition contract above the post-article conformance harness.

The canonical machine-readable artifacts are:

- `fixtures/tassadar/runs/tassadar_post_article_plugin_result_binding_schema_stability_and_composition_v1/tassadar_post_article_plugin_result_binding_schema_stability_and_composition_bundle.json`
- `fixtures/tassadar/reports/tassadar_post_article_plugin_result_binding_schema_stability_and_composition_report.json`
- `fixtures/tassadar/reports/tassadar_post_article_plugin_result_binding_schema_stability_and_composition_summary.json`
- `crates/psionic-provider/src/tassadar_post_article_plugin_result_binding_schema_stability_and_composition.rs`
- `scripts/check-tassadar-post-article-plugin-result-binding-schema-stability-and-composition.sh`

## What Is Now Frozen

The bounded post-article plugin lane now has one explicit result-binding
contract that:

- keeps schema evolution explicit under exact identity or declared
  backward-compatible rules
- binds admitted plugin-output digests to the next model-visible state digest
  explicitly
- preserves typed refusal and failure classes through reinjection instead of
  collapsing them into one generic adapter-defined error
- keeps proof-carrying result guarantees distinct from observational result
  audits
- keeps multi-step chaining semantically closed and non-lossy for admitted
  compositions
- keeps lossy coercion, schema auto-repair, ambiguous composition, and
  semantically incomplete reinjection on explicit fail-closed lanes

## What This Still Does Not Claim

This tranche does not claim:

- weighted plugin selection, sequencing, retries, or stop conditions
- plugin publication rights
- served or public plugin universality
- arbitrary public software execution

The remaining frontier moves to `TAS-204`, which is where the weighted
controller trace and refusal-aware model loop must be made true.
