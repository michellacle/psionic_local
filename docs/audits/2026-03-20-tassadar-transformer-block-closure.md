# TAS-162 Transformer Block Closure

`TAS-162` lands the reusable Transformer block layer inside
`psionic-transformer`.

This issue does not close the paper-faithful article model or the final
article-equivalence route. It freezes the reusable block composition that the
later article-model and route work must consume.

## What Landed

- one reusable block-composition implementation in
  `crates/psionic-transformer/src/blocks.rs`
- one committed eval artifact at
  `fixtures/tassadar/reports/tassadar_transformer_block_closure_report.json`
- one committed research summary at
  `fixtures/tassadar/reports/tassadar_transformer_block_closure_summary.json`
- one audit note at
  `docs/audits/2026-03-20-tassadar-transformer-block-closure.md`

## Block Scope

The owned reusable path now covers:

- token embedding plus positional binding
- explicit train versus eval dropout posture
- multi-head projection and merge
- position-wise feed-forward composition
- residual paths and layer norm
- reusable decoder-block execution

These pieces now live in `psionic-transformer` above `psionic-nn` instead of
remaining implicit or drifting into per-model code.

## Boundary Statement

The crate boundary is explicit:

- `psionic-transformer` now depends directly on `psionic-nn`
- `psionic-transformer` still does not depend directly on `psionic-models` or
  `psionic-runtime`
- `psionic-nn` now stays below `psionic-train`, `psionic-models`, and
  `psionic-runtime`
- optimizer interop stays split above the lower-level layer substrate in
  `psionic-nn-optimizers`
- `psionic-models` still does not define or re-own the reusable block symbols

This keeps reusable block composition at the canonical Transformer layer while
preserving an acyclic dependency direction.

## Closure-Gate Tie

The new artifact ties directly into the final article-equivalence acceptance
gate through `TAS-162`.

Current bounded truth:

- `TAS-162` is now green inside the acceptance gate
- the gate itself remains `blocked`
- article equivalence itself remains red

## Validation

- `cargo test -p psionic-transformer -- --nocapture`
- `cargo test -p psionic-nn-optimizers -- --nocapture`
- `cargo test -p psionic-eval transformer_block_closure -- --nocapture`
- `cargo test -p psionic-research transformer_block_closure_summary -- --nocapture`

## Claim Boundary

This issue closes only the reusable Transformer block layer. It does not imply
that the paper-faithful article model artifact, the Transformer-backed weight
lineage, the reference-linear proof route, or the final article-equivalence
verdict are already complete.

## Audit Statement

Psionic now has one canonical reusable Transformer block layer in
`psionic-transformer`, tied directly to the final article-equivalence gate,
while the overall article-equivalence verdict remains blocked.
