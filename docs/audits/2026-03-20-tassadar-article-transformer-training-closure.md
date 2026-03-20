# TAS-164 Article Transformer Training Closure

`TAS-164` lands the first bounded training lane for the canonical
paper-faithful article-Transformer stack.

This issue does not close full article-model training, replay receipts,
artifact-backed weight lineage, Hungarian or Sudoku benchmark parity, or the
final article-equivalence verdict. It closes the narrower claim that the owned
stack can train and restore on bounded tasks without leaving the
`psionic-transformer` route.

## What Landed

- one bounded training module at
  `crates/psionic-train/src/tassadar_article_transformer_training.rs`
- one committed evidence bundle at
  `fixtures/tassadar/runs/tassadar_article_transformer_training_v1/article_transformer_training_evidence_bundle.json`
- one committed eval artifact at
  `fixtures/tassadar/reports/tassadar_article_transformer_training_closure_report.json`
- one committed research summary at
  `fixtures/tassadar/reports/tassadar_article_transformer_training_closure_summary.json`
- one audit note at
  `docs/audits/2026-03-20-tassadar-article-transformer-training-closure.md`

## Training Scope

The new bounded lane now covers:

- label-smoothed cross-entropy over the canonical article wrapper
- Adam with inverse-square-root warmup
- tiny-batch overfit on two toy selector tasks
- finite-difference gradient checks over the bounded trainable parameter set
- deterministic checkpoint export and restore via Psionic-native
  `safetensors`

The lane is intentionally narrow. It proves that the owned article stack is a
real trainable model path, not just static forward math.

## Boundary Statement

The route boundary stays explicit:

- `psionic-transformer` continues to own the reusable encoder-decoder stack
- `psionic-models` continues to own the canonical article wrapper and
  trainable parameter surface
- `psionic-train` owns the optimizer loop, step receipts, and checkpoint
  export for the bounded training lane
- `psionic-eval` and `psionic-research` consume the committed evidence bundle
  rather than restating the training recipe by hand

This keeps the trained path rooted in `psionic-transformer` instead of
creating a disconnected fixture-only training route.

## Closure-Gate Tie

The new artifact ties directly into the final article-equivalence acceptance
gate through `TAS-164`.

Current bounded truth:

- `TAS-164` is now green inside the acceptance gate
- the gate itself remains `blocked`
- blocker family `BEQ-003` remains open because later owned-stack,
  artifact-lineage, exactness, and audit tranches are still incomplete

## Validation

- `cargo test -p psionic-train tassadar_article_transformer_training -- --nocapture`
- `cargo test -p psionic-eval article_transformer_training_closure -- --nocapture`
- `cargo test -p psionic-research article_transformer_training_closure_summary -- --nocapture`

## Claim Boundary

This issue closes only the bounded training recipe and restore lane for the
owned article-Transformer stack. It does not imply full article-model
training, benchmark parity, generic interpreter-in-weights closure, or final
article-equivalence green status.

## Audit Statement

Psionic now has one bounded, machine-legible article-Transformer training lane
with finite gradients, tiny-task overfit, and deterministic checkpoint restore
on the canonical owned stack, while the broader article-equivalence verdict
remains blocked.
