# Tassadar Post-Article Plugin Conformance Sandbox And Benchmark Harness

## Scope

`TAS-203` closes the first bounded plugin conformance and benchmark harness
above the post-article admissibility contract.

The canonical machine-readable artifacts are:

- `fixtures/tassadar/runs/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_v1/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_bundle.json`
- `fixtures/tassadar/reports/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_report.json`
- `fixtures/tassadar/reports/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_eval_report.json`
- `fixtures/tassadar/reports/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_summary.json`
- `crates/psionic-provider/src/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness.rs`
- `scripts/check-tassadar-post-article-plugin-conformance-sandbox-and-benchmark-harness.sh`

## What Is Now Frozen

The bounded post-article plugin lane now has one explicit conformance and
benchmark harness that:

- keeps conformance traces static and host-scripted rather than model-owned
  sequencing
- keeps roundtrip, malformed-packet refusal, capability denial, timeout,
  memory-limit, packet-size refusal, digest-mismatch refusal, replay, and
  hot-swap compatibility rows explicit
- keeps multi-plugin workflow integrity, refusal propagation, envelope
  intersection, hot-swap inside composed workflows, and replay under partial
  cancellation explicit
- keeps per-plugin, per-step, and per-workflow failure-domain isolation
  explicit
- keeps shared-cache, shared-store, timing-channel, and covert-channel
  negatives explicit
- keeps cold, warm, pooled, queued, cancelled, evidence-overhead, and
  timeout-enforcement benchmark evidence explicit

## What This Still Does Not Claim

This tranche does not claim:

- weighted plugin sequencing
- plugin publication rights
- served or public plugin universality
- arbitrary public software execution

The remaining frontier stays on later result-binding, controller, authority,
and platform issues.
