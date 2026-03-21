# Tassadar Post-Article Plugin Runtime API And Engine Abstraction

`TAS-200` closes the next bounded plugin-runtime tranche above the canonical
post-`TAS-186` machine by freezing one host-owned runtime API and one
backend-neutral engine abstraction on top of the already-closed packet ABI and
manifest contract.

The committed runtime bundle,
`fixtures/tassadar/runs/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_v1/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_bundle.json`,
now states machine-readably that the runtime must:

- load only declared plugin artifacts
- verify artifact digests before instantiate or pool admission
- keep instantiate, invoke, mount, cancel, and usage collection behind one
  host-owned runtime API
- keep backend-specific engine details below that API
- enforce explicit timeout, memory, queue, pool, and concurrency bounds
- keep queue depth, retries, runtime cost, and wall-clock time hidden from the
  model
- keep scheduling semantics fixed and cost-model invariance explicit
- keep per-plugin, per-step, and per-workflow failure isolation explicit

The sandbox-owned report,
`fixtures/tassadar/reports/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_report.json`,
binds that bundle to the same canonical bridge machine identity, canonical
route, computational-model statement, and packet ABI contract as the earlier
plugin tranche. It also cites the supporting import-policy, async-lifecycle,
and simulator-effect artifacts so bounded runtime behavior is not claimed by
implication from the bundle alone.

The operator summary, provider projection, served publication, and checker now
live at:

- `fixtures/tassadar/reports/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_summary.json`
- `crates/psionic-provider/src/tassadar_post_article_plugin_runtime_api_and_engine_abstraction.rs`
- `fixtures/tassadar/reports/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_publication.json`
- `scripts/check-tassadar-post-article-plugin-runtime-api-and-engine-abstraction.sh`

Claim boundary:

- this tranche allows the rebased machine claim to carry one bounded
  host-owned plugin runtime API and engine abstraction
- it does not by itself allow weighted plugin control
- it does not allow plugin publication
- it does not widen served/public universality
- it does not imply arbitrary software capability

The deferred frontier now moves to `TAS-201`, where invocation receipts and
replay classes must be frozen explicitly.
