# TAS-188 Audit

The public repo now has one machine-readable post-article canonical-route
semantic-preservation audit.

That audit closes a narrower tranche than full rebased universality. It proves
that the declared continuation mechanics now preserve canonical model identity,
canonical route identity, declared state ownership, and declared semantics on
the bridge machine identity instead of merely reproducing a subset of outputs.

The public repo now says, machine-readably:

- canonical model id, route id, route digest, and continuation-contract digest
  stay fixed across the audited continuation mechanics
- host may execute declared resume, spill, migration, and rollback mechanics,
  but host does not own workflow decisions
- workflow authority stays weight-owned while ephemeral, resumed, and durable
  state remain separate non-authoritative classes
- any workflow-affecting state outside those declared classes is refused
- session-process continuation, spill/tape continuation, and installed-process
  lifecycle each preserve declared semantics with exact parity and typed
  refusal boundaries
- proof-carrying artifacts remain distinct from observational March 20 audits

This is stronger than saying the repo merely has direct article closure plus
older continuation artifacts. It freezes one bounded proof-carrying closeout
for state ownership and semantic preservation on the post-`TAS-186` bridge
machine identity.

It is still not a full rebased universality claim.

The audit does not say:

- that branch, retry, and stop decision provenance is already proved
- that the final direct-versus-resumable carrier split is already published as
  its own machine-readable contract
- that the rebased Turing-completeness claim is already admitted on the
  canonical route
- that weighted plugin control is already in scope
- that served/public universality is already allowed
- that arbitrary software capability is already allowed

Canonical artifacts for this tranche:

- `fixtures/tassadar/reports/tassadar_post_article_canonical_route_semantic_preservation_audit_report.json`
- `fixtures/tassadar/reports/tassadar_post_article_canonical_route_semantic_preservation_summary.json`
- `crates/psionic-provider/src/tassadar_post_article_canonical_route_semantic_preservation.rs`
- `scripts/check-tassadar-post-article-canonical-route-semantic-preservation-audit.sh`
