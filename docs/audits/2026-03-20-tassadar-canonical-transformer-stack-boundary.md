# TAS-160 Canonical Transformer Stack Boundary

`TAS-160` freezes the one owned Transformer stack boundary that the final
article-equivalence route must use.

This issue does not claim that the final article-equivalence route is already
complete. It fixes the ownership split above the existing Psionic substrate so
later work cannot close a weaker ad hoc route and call it article-equivalent.

## What Landed

- one canonical boundary spec at
  `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md`
- one committed eval artifact at
  `fixtures/tassadar/reports/tassadar_canonical_transformer_stack_boundary_report.json`
- one committed research summary at
  `fixtures/tassadar/reports/tassadar_canonical_transformer_stack_boundary_summary.json`
- one audit note at
  `docs/audits/2026-03-20-tassadar-canonical-transformer-stack-boundary.md`

## Boundary Decision

The canonical article route is not one new monolithic crate. It is one
explicit boundary spanning existing crates:

- `psionic-core` plus `psionic-array` own tensor and bounded array ops
- `psionic-nn` plus `psionic-transformer` own layer, parameter-state, and
  reusable Transformer-architecture interfaces
- `psionic-models` owns the canonical article-model wrapper, model artifact
  format, and forward-pass trace hook surface
- `psionic-runtime` owns replay, trace ABI, proof identity, and receipt hooks

`psionic-transformer` is the architecture anchor inside that boundary.

## Required Interface Coverage

The committed boundary report freezes five exact interface families:

- `tensor_array_ops`
- `layer_and_parameter_state`
- `model_artifact_format`
- `forward_pass_trace_hooks`
- `replay_and_receipt_hooks`

The report also carries one ownership diagram and real module refs for each
crate-level owner.

## Dependency Direction

The committed boundary report checks the real `Cargo.toml` dependency shape:

- `psionic-models` depends on `psionic-transformer`
- `psionic-transformer` does not depend on `psionic-models`
- `psionic-transformer` does not depend on `psionic-runtime`
- `psionic-runtime` does not depend on `psionic-models`
- `psionic-models` does not depend on `psionic-eval` or `psionic-research`

## Closure-Gate Tie

The boundary is explicitly tied into the final article-equivalence acceptance
gate through `TAS-160`.

Current bounded truth:

- `TAS-160` is now green inside the acceptance gate
- the gate itself remains `blocked`
- article equivalence itself remains red

This means the canonical ownership split is now frozen without pretending the
full article route, article-model lineage, or final closure proof already
exists.

## Validation

- `cargo test -p psionic-eval canonical_transformer_stack_boundary -- --nocapture`
- `cargo test -p psionic-research canonical_transformer_stack_boundary_summary -- --nocapture`

## Claim Boundary

This issue freezes the route boundary only. It does not imply that the
canonical article model artifact is already Transformer-backed, that the
reference-linear article route is complete, or that final article equivalence
is already green.

## Audit Statement

Psionic now has one canonical machine-readable Transformer stack boundary for
the article-equivalence closure wave, with `psionic-transformer` as the
architecture anchor inside one explicit multi-crate ownership split tied
directly to the final acceptance gate, while the overall article-equivalence
verdict remains blocked.
