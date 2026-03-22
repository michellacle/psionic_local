# Tassadar Post-Article Canonical Computational Model Statement

`TAS-208` publishes one runtime-owned computational-model statement for the
post-article canonical machine.

The canonical machine-readable artifacts are:

- `fixtures/tassadar/reports/tassadar_post_article_canonical_computational_model_statement_report.json`
- `fixtures/tassadar/reports/tassadar_post_article_canonical_computational_model_statement_summary.json`
- `crates/psionic-transformer/src/tassadar_post_article_canonical_computational_model_contract.rs`
- `crates/psionic-provider/src/tassadar_post_article_canonical_computational_model_statement.rs`
- `scripts/check-tassadar-post-article-canonical-computational-model-statement.sh`

This tranche makes three things explicit and machine-readable:

1. The canonical post-article machine is one owned direct
   `tassadar.article_route.direct_hull_cache_runtime.v1` Transformer route on
   the closed article-equivalence model and weight lineage.
2. Resumable continuation semantics and effect boundaries attach to that same
   machine only through the historical green `TCM.v1` runtime contract.
3. The plugin layer sits above that machine as a bounded software-capability
   overlay; it does not redefine the machine substrate, continuation carrier,
   or publication posture.

This statement does not yet prove execution-semantics proof transport, it does
not publish the final claim-bearing canonical machine closure bundle, and it
does not turn plugin publication, served/public universality, or arbitrary
software capability green.

The bridge frontier therefore moves forward to `TAS-209`, while the final
closure-bundle separation still remains at `TAS-215`.
