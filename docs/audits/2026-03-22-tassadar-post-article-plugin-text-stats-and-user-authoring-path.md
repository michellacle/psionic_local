# 2026-03-22 Tassadar Post-Article Plugin Text Stats And User Authoring Path

This audit records one practical follow-up exercise on top of the landed
starter-plugin runtime:

> Can a repo user add one more real host-native plugin the way the current
> Psionic runtime expects, then run it through the same bounded packet and
> receipt path the existing starter plugins use?

The short answer is yes, but the current path is still a repository-authoring
path, not a polished external plugin SDK.

## Landed Surfaces

- `psionic-runtime` now owns a real `plugin.text.stats` runtime path in
  `crates/psionic-runtime/src/tassadar_post_article_starter_plugin_runtime.rs`
- the same runtime writes committed fixture truth to
  `fixtures/tassadar/runs/tassadar_post_article_plugin_text_stats_v1/tassadar_post_article_plugin_text_stats_bundle.json`
- `crates/psionic-runtime/examples/tassadar_post_article_plugin_text_stats_bundle.rs`
  now writes the canonical runtime bundle for the new plugin
- `crates/psionic-runtime/examples/tassadar_post_article_plugin_text_stats_demo.rs`
  now acts as the direct user-run example for invoking the plugin with one JSON
  packet and printing the typed response plus receipt identity
- `scripts/check-tassadar-post-article-plugin-text-stats.sh` now acts as the
  dedicated checker over the runtime bundle and targeted tests
- `docs/TASSADAR_STARTER_PLUGIN_RUNTIME.md` now tracks `plugin.text.stats` as a
  first-class capability-free starter-runtime entry

## What Was Actually Implemented

The new plugin is intentionally small and deterministic:

- canonical plugin id: `plugin.text.stats`
- stable tool projection: `plugin_text_stats`
- input schema id: `plugin.text.stats.input.v1`
- output schema id: `plugin.text.stats.output.v1`
- mount envelope id: `mount.plugin.text.stats.no_capabilities.v1`
- replay class: `deterministic_replayable`

Input shape:

- `{ "text": string }`

Output shape:

- `byte_count`
- `unicode_scalar_count`
- `line_count`
- `non_empty_line_count`
- `word_count`

The counting rules stay explicit and bounded:

- bytes use Rust string byte length
- scalar count uses `chars()`
- line count uses `lines()`
- word count uses `split_whitespace()`

Typed refusals are also explicit:

- `plugin.refusal.schema_invalid.v1`
- `plugin.refusal.packet_too_large.v1`
- `plugin.refusal.unsupported_codec.v1`

Negative claims stay explicit:

- no tokenizer truth
- no language detection truth
- no sentence-boundary truth
- no semantic-structure truth

## What Was Actually Run

The implementation was not left at the type-definition level.
The following real commands were run in the repo checkout:

- `cargo test -p psionic-runtime text_stats_ -- --nocapture`
- `cargo run -p psionic-runtime --example tassadar_post_article_plugin_text_stats_demo`
- `cargo run -p psionic-runtime --example tassadar_post_article_plugin_text_stats_bundle`
- `./scripts/check-tassadar-post-article-plugin-text-stats.sh`

The direct demo command is the clearest statement of the current user path:

```bash
cargo run -p psionic-runtime --example tassadar_post_article_plugin_text_stats_demo
```

That run produced:

- one success response with deterministic count fields
- one receipt id:
  `receipt.plugin.text.stats.72ccc4ddaca2729d.v1`
- one receipt digest:
  `94985e020227e97bc4074a3d32b743b11a4a834bd6480e011185b1c498063dfd`

The bundle writer also produced the committed runtime artifact:

- bundle path:
  `fixtures/tassadar/runs/tassadar_post_article_plugin_text_stats_v1/tassadar_post_article_plugin_text_stats_bundle.json`
- bundle digest:
  `3da14ca2372c133187c6b969077c45256c082dcad251e269777ce7262890b14a`

## What This Proves

This exercise proves several practical things:

- the starter-plugin runtime is not frozen to the original four plugins
- a fifth real host-native plugin can be added without inventing a second
  runtime contract
- the packet ABI, typed-refusal posture, and receipt-binding pattern are
  reusable for another capability-free local transform
- a repo user can run the new plugin directly through the runtime using one
  narrow example binary instead of wiring it through router or Apple FM first

That matters because it shows the runtime contract is reusable in code, not
just in the original starter-plugin tranche.

## What This Does Not Yet Prove

This exercise does not close the broader user-authoring problem.

It does **not** yet prove:

- a drop-in external plugin folder or manifest workflow
- a generated starter-plugin scaffold
- automatic registration of new plugins into the shared tool bridge
- automatic registration into router-owned tool loops
- automatic registration into Apple FM tool callbacks
- automatic publication into the starter-plugin catalog
- a stable public plugin SDK outside this repo

The honest current authoring path is still:

1. add a host-native plugin implementation in the runtime source
2. add its runtime bundle writer
3. add its targeted tests
4. add its checker script
5. add its user-run demo example
6. update the runtime documentation

So this is a real user-authoring path for a repository contributor, but not
yet a polished plugin-authoring product surface.

## Why The Current Path Is Still Acceptable

For the current state of the repo, this is still a good next step because it
keeps the truth explicit:

- the plugin is real code, not a mock
- the user path is executable and reproducible
- typed refusals and receipts stay first-class
- the capability-free boundary stays narrow
- no false claim is made that Psionic already has a generalized plugin SDK

That is better than pretending the bridge, router, Apple FM, and catalog layers
can already absorb arbitrary new plugins automatically.

## Immediate Next Steps

The next logical steps are straightforward and dependency-ordered.

### 1. Factor starter-plugin registration into one central table

Right now each starter plugin is hand-registered in multiple places. The repo
should pull the shared plugin metadata into one runtime-owned registration
table so that:

- tool projection
- bundle writing
- bridge exposure
- catalog exposure

all derive from the same source of truth.

This is the most valuable next cleanup because it turns “add a fifth plugin”
from a multi-file surgery into one bounded registration change plus one plugin
implementation.

### 2. Extend the shared tool bridge to include `plugin.text.stats`

The new plugin currently runs directly through the runtime and the demo
example, but it is not yet projected through:

- `docs/TASSADAR_STARTER_PLUGIN_TOOL_BRIDGE.md`
- router `/v1/responses`
- Apple FM tool definitions

The immediate next integration step is to add `plugin_text_stats` to the shared
bridge and refresh the bridge bundle and checks honestly.

### 3. Extend the starter-plugin catalog and downstream reports

The starter-plugin catalog still freezes the previous four-plugin set.
Once the central registration path exists, the repo should:

- add `plugin.text.stats` descriptor, fixture, and mount-envelope sidecars
- refresh the catalog bundle
- refresh the catalog report, eval report, summary, and provider receipt

That closes the gap between “runtime plugin exists” and “cataloged starter
plugin exists.”

### 4. Add one controller-level use of `plugin.text.stats`

After the bridge is updated, the repo should prove at least one composition
case using the new plugin, for example:

- `plugin_http_fetch_text -> plugin_text_stats`
- `plugin_html_extract_readable -> plugin_text_stats`

This would keep the plugin useful in the orchestration wave instead of leaving
it as an isolated runtime example.

### 5. Publish a starter-plugin authoring template

Once registration is centralized, the repo should add one explicit template or
guide that shows:

- required request and response structs
- required tool projection
- refusal conventions
- bundle writer shape
- targeted test expectations
- checker script pattern

That would turn the current authoring pattern from “read source and imitate it”
into a documented contract.

### 6. Consider a narrow scaffold generator only after the template is stable

A generator is useful only if the contract is already stable enough.
The repo should not generate starter-plugin scaffolds before the central
registration path and template exist; otherwise it would just automate today’s
manual duplication.

### 7. Keep capability-free plugins separate from networked plugins

`plugin.text.stats` is a good example of the low-risk class:

- local
- deterministic
- capability-free

The repo should continue to expand this class first before widening the
networked plugin family, because it keeps replay, receipts, and refusal truth
much easier to reason about.

## Strategic Next Frontier

The broader strategic objective is not “add many more ad hoc plugins.”

The real frontier is:

- make starter-plugin authoring cheaper
- keep runtime truth explicit
- and let bridge, catalog, router, and Apple FM lanes derive from one shared
  registration source

If that lands, Psionic can support more real host-native plugins without
turning each new plugin into a cross-repo bookkeeping exercise.

That is the honest next step from this audit, and it is the right sequencing
before any stronger public claim about generalized plugin authoring.
