# TAS-159 Existing Substrate Inventory And Reuse Boundary

`TAS-159` freezes which current Psionic surfaces are real reusable substrate
for the article-equivalence closure wave and which surfaces still remain
blocking, research-only, or insufficient.

This issue does not implement the owned article-equivalence Transformer stack.
It freezes the overlap so later work can build on existing substrate instead of
rebuilding it blindly.

## What Landed

- one committed eval artifact at
  `fixtures/tassadar/reports/tassadar_existing_substrate_inventory_report.json`
- one committed research summary at
  `fixtures/tassadar/reports/tassadar_existing_substrate_inventory_summary.json`
- one audit note at
  `docs/audits/2026-03-20-tassadar-existing-substrate-inventory.md`

## Inventory Boundary

The canonical inventory now freezes ten substrate rows across:

- `psionic-core`
- `psionic-array`
- `psionic-nn`
- `psionic-transformer`
- `psionic-models`
- `psionic-runtime`

Each row is machine-labeled as one of:

- `reusable_as_is`
- `reusable_with_extension`
- `research_only`
- `not_sufficient_for_article_closure`

Each row also carries an explicit blocker or non-blocker label.

## Current Counts

The committed inventory records:

- `surface_count=10`
- `blocker_surface_count=5`
- `non_blocker_surface_count=5`
- `reusable_as_is=4`
- `reusable_with_extension=4`
- `research_only=1`
- `not_sufficient_for_article_closure=1`

## Closure-Gate Tie

The inventory is explicitly tied to the final article-equivalence acceptance
gate through `TAS-159`.

Current bounded truth:

- `TAS-159` is now green inside the acceptance gate
- the gate itself remains `blocked`
- article equivalence itself remains red

This means the repo now has one canonical reusable-substrate inventory without
pretending the canonical article route is already complete.

## Validation

- `cargo test -p psionic-eval existing_substrate_inventory -- --nocapture`
- `cargo test -p psionic-research existing_substrate_inventory_summary -- --nocapture`

## Claim Boundary

This inventory is a reuse map, not a closure claim. It does not imply that the
owned article-equivalence Transformer stack, the canonical article model
artifact, or the final article-equivalence proof route already exist.

## Audit Statement

`Psionic now has one canonical machine-readable existing-substrate inventory for the article-equivalence closure wave, with explicit reusable, extension, research-only, and insufficient rows tied directly to the final acceptance gate, while the overall article-equivalence verdict remains blocked.`
