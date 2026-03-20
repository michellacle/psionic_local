# Tassadar Transformer Generalization Gate

`TAS-171B` closes the next post-exactness tranche on the owned
Transformer-backed reference-linear route.

The repository now commits one deterministic generalization and
anti-memorization gate at
`fixtures/tassadar/reports/tassadar_article_transformer_generalization_gate_report.json`
plus the operator summary at
`fixtures/tassadar/reports/tassadar_article_transformer_generalization_summary.json`.

That gate ties back to the committed reference-linear exactness prerequisite,
then requires one explicit held-out and adversarial article-envelope suite with:

- randomized bounded-family Sudoku and Hungarian cases
- adversarial article-scale Sudoku and Hungarian variants
- explicit size and structure scaling coverage across bounded and article-scale
  families
- mixed-order curriculum runs across the same held-out case pool
- out-of-distribution fingerprints against both the declared article corpus and
  the relevant family corpora

The current closure statement is narrow:

- the owned reference-linear route is now machine-readably green on this
  deterministic held-out and adversarial article-envelope generalization suite
- the gate keeps mismatch, refusal, and overlap posture explicit if any row
  drifts or falls back onto seen workload structure

The gate still does not certify:

- dataset contamination independence
- fast-route promotion
- article demo or benchmark equivalence
- single-run no-spill closure
- clean-room weight causality or interpreter ownership
- route minimality
- final article-equivalence green status
