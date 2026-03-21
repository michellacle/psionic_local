# TAS-191 Audit

`TAS-191` closes the witness-suite reissue tranche for the post-`TAS-186`
universality bridge.

The canonical artifact is now
`fixtures/tassadar/reports/tassadar_post_article_universality_witness_suite_reissue_report.json`.
It reissues the older seven-row witness suite onto the declared bridge machine
identity, canonical model artifact, and canonical route id while keeping the
same exact versus refusal-boundary families explicit.

The report keeps helper substitution, hidden cache-owned control flow, and
resume-only cheating as explicit negative rows instead of letting those
surfaces satisfy the canonical-route suite by implication.

The operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_universality_witness_suite_reissue_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_universality_witness_suite_reissue.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-universality-witness-suite-reissue.sh`.

This tranche is intentionally narrower than the later canonical-route
universal-substrate gate or verdict publication. It does not yet enable the
canonical-route universal-substrate gate, publish the rebased
theory/operator/served verdict split, admit served/public universality, admit
weighted plugin control, or admit arbitrary software capability.
