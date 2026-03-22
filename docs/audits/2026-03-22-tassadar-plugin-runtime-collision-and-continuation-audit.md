# 2026-03-22 Tassadar Plugin Runtime Collision And Continuation Audit

## Scope

This audit was written after fetching `origin/main` and moving to a fresh
post-pull checkout at commit `c1d76f97`
(`Implement TAS-220 feed parse starter runtime`).

The goal is narrow:

- record whether the paused local starter-plugin scratch work now collides with
  landed `origin/main` work
- classify whether those collisions are mechanical, semantic, or both
- state which checkout is authoritative
- state who should continue the plugin lane next

## Fresh `origin/main` Truth

`origin/main` now carries a complete starter-plugin runtime tranche, not just
the older starter-catalog shell:

- `ddd68407` `Implement TAS-217 url extract starter runtime`
- `fc68cd11` `Implement TAS-218 fetch text starter runtime`
- `b38eaa49` `Implement TAS-219 readable HTML starter runtime`
- `c1d76f97` `Implement TAS-220 feed parse starter runtime`

The corresponding GitHub issues are already closed:

- `#348` `TAS-217: Canonicalize URL Extract As A Deterministic Starter Plugin`
- `#347` `TAS-218: Add HTTP Fetch Text As A Read-Only Starter Plugin`
- `#346` `TAS-219: Add HTML Extract Readable As A Deterministic Starter Plugin`
- `#349` `TAS-220: Add RSS And Atom Parse As A Deterministic Starter Plugin`

The landed implementation direction is now explicit:

- one shared runtime surface:
  `crates/psionic-runtime/src/tassadar_post_article_starter_plugin_runtime.rs`
- one shared doc:
  `docs/TASSADAR_STARTER_PLUGIN_RUNTIME.md`
- four real runtime-owned starter plugin bundles:
  `plugin.text.url_extract`, `plugin.http.fetch_text`,
  `plugin.html.extract_readable`, and `plugin.feed.rss_atom_parse`
- current open continuation lane:
  `#355` / `TAS-221` umbrella and `#350` / `TAS-222` shared plugin-to-tool
  projection bridge

## Local Scratch State Observed At Audit Time

Three local checkouts existed when this audit was written:

1. `/home/christopherdavid/code/psionic`
   `main` was dirty and behind `origin/main`, with broad article-equivalence and
   interpreter-breadth work in progress. It was not a safe landing checkout for
   any plugin follow-on work.
2. `/home/christopherdavid/code/psionic-tas-clean`
   `tas-clean` held an isolated but unmerged starter-plugin scratch set built
   around a pre-runtime interpretation of `TAS-217`, plus a few unstaged fixes
   made while validating nested checker scripts.
3. `/home/christopherdavid/code/psionic-tas-seq`
   `tas-seq` held an older mixed scratch checkout with much broader unrelated
   dirt and was not safe for continuation.

Only the fresh post-pull checkout based on `origin/main` was authoritative for
new decision-making.

## Collision Assessment

### Direct file overlap

The paused `tas-clean` starter-plugin scratch and the landed plugin-runtime
tranche directly overlap on these files:

- `crates/psionic-runtime/src/lib.rs`
- `docs/ARCHITECTURE.md`
- `docs/ROADMAP_TASSADAR.md`
- `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md`

These are real merge collisions, not theoretical ones.

### Semantic overlap and supersession

The bigger collision is architectural, not textual.

The paused local scratch was built around the older assumption that `TAS-217`
was still open and should be closed by adding a dedicated
`tassadar_post_article_url_extract_starter_plugin_*` report or eval or research
or provider lane above the starter-catalog artifacts.

`origin/main` has now closed `TAS-217` through `TAS-220` differently:

- the first-class implementation surface is the shared
  `tassadar_post_article_starter_plugin_runtime.rs` runtime
- the first-class evidence surface is one runtime bundle per real starter
  plugin, not one catalog-derived URL-extract-only closure stack
- the first-class documentation surface is
  `docs/TASSADAR_STARTER_PLUGIN_RUNTIME.md` plus per-plugin audit notes dated
  `2026-03-22`

That means the paused TAS-217 scratch is now semantically superseded even where
it does not touch the exact same filenames.

### Issue-state collision

The paused scratch was still treating `TAS-217` as active work.

That assumption is no longer true:

- `TAS-217` is closed on GitHub
- `TAS-218` is closed on GitHub
- `TAS-219` is closed on GitHub
- `TAS-220` is closed on GitHub

Any agent continuing from the paused TAS-217 scratch without first rebasing
onto `origin/main` would be continuing from stale issue state.

## Risk Classification

### Safe to reuse

- narrow implementation ideas that do not conflict with the new shared runtime
  abstraction
- bounded refusal wording or test cases if they still match the landed runtime
  contracts
- documentation language only after checking it against the current
  `TASSADAR_STARTER_PLUGIN_RUNTIME.md` framing

### Not safe to merge as-is

- the paused `tas-clean` or `tas-seq` starter-plugin branches
- any commit that reopens `TAS-217` through `TAS-220` as unfinished
- any doc change that re-centers the plugin lane on the old catalog-derived
  URL-extract-only closure stack instead of the landed shared runtime
- any staged artifact set produced from the dirty scratch worktrees rather than
  from fresh `origin/main`

## Authority Decision

The authoritative line is now:

- `origin/main`
- commits `ddd68407` through `c1d76f97`
- the shared starter-plugin runtime abstraction in
  `crates/psionic-runtime/src/tassadar_post_article_starter_plugin_runtime.rs`

The paused local starter-plugin scratch work should be treated as superseded
operator scratch, not as the branch to continue or merge.

## Continuation Recommendation

The agent who landed the runtime tranche on `origin/main` should continue.

Reason:

- that agent already owns the current shared runtime abstraction
- that agent already closed the four starter-plugin runtime issues in sequence
- the next open work depends directly on that exact abstraction, not on the
  older catalog-derived TAS-217 interpretation

The concrete next issue to continue is:

- `#350` / `TAS-222: Shared Plugin-To-Tool Projection And Receipt Bridge`

and it should be executed under:

- `#355` / `TAS-221: Publish The First Real-Run Multi-Plugin Orchestration Wave`

## Operator Instruction

Do not continue starter-plugin work from:

- `/home/christopherdavid/code/psionic`
- `/home/christopherdavid/code/psionic-tas-clean`
- `/home/christopherdavid/code/psionic-tas-seq`

Continue only from a fresh checkout or worktree created from current
`origin/main`.

If any content is harvested from the paused scratch work, do it selectively and
only after mapping it onto the landed shared runtime surfaces and the now-open
`TAS-222` bridge scope.
