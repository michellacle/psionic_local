# TAS-189 Audit

`TAS-189` closes the carrier-publication tranche for the post-`TAS-186`
universality bridge.

The canonical artifact is now
`fixtures/tassadar/reports/tassadar_post_article_carrier_split_contract_report.json`.
It binds the direct article-equivalent truths and the resumable bounded
universality truths to different carriers on the same bridge machine identity,
states machine-readably that transfer by implication is blocked, and keeps the
reserved capability plane explicit instead of leaving later widening
unspecified.

The operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_carrier_split_contract_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_carrier_split_contract.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-carrier-split-contract.sh`.

This tranche is intentionally narrower than a full rebased universality
approval. It does not yet rebind the historical universal-machine proof onto
the post-`TAS-186` machine identity, reissue the witness suite on that
identity, publish the rebased verdict split, admit served/public universality,
admit weighted plugin control, or admit arbitrary software capability.
