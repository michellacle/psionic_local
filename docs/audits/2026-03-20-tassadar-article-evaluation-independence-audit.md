# Tassadar Article Evaluation Independence Audit

`TAS-171C` closes the dataset-contamination and evaluation-independence tranche
on top of the owned Transformer-backed reference-linear route.

The repository now commits one explicit audit at
`fixtures/tassadar/reports/tassadar_article_evaluation_independence_audit_report.json`
plus the operator summary at
`fixtures/tassadar/reports/tassadar_article_evaluation_independence_summary.json`.

That audit is anchored to the committed `TAS-171B` generalization suite and the
committed trained lineage contract, then requires:

- exact case-id exclusion between training and evaluation rows
- exact source-token, target-token, and sequence-digest exclusion
- prefix-window near-duplicate checks across the training and evaluation sets
- generator-level separation through distinct generator ids and rule digests
- feature-distribution review so the evaluation suite is not just a relabeled
  replay of the training slice

The current closure statement is narrow:

- the held-out and adversarial article-envelope evaluation suite is now
  machine-readably separated from the current bounded trained article slice by
  explicit overlap, near-duplicate, generator, and feature-separation audits
- the gate stays red if evaluation success can be explained by leaked cases,
  overlapping token windows, or procedurally shared generators rather than
  independent generalization

The audit still does not certify:

- fast-route promotion
- article demo or benchmark equivalence
- single-run no-spill closure
- clean-room weight causality or interpreter ownership
- route minimality
- final article-equivalence green status
