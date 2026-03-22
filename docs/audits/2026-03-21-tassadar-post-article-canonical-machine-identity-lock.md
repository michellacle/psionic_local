# Tassadar Post-Article Canonical Machine Identity Lock

`TAS-207` closes the first explicit canonical-machine anti-drift lock above the
post-`TAS-186` bridge, route audits, proof and witness surfaces, plugin
receipts, controller traces, conformance harnesses, authority posture, and the
bounded weighted plugin-platform closeout.

The new machine-readable contract lives at
`fixtures/tassadar/reports/tassadar_post_article_canonical_machine_identity_lock_report.json`.
It is anchored by the transformer-owned contract in
`crates/psionic-transformer/src/tassadar_post_article_canonical_machine_identity_lock_contract.rs`,
which names one globally named machine tuple and one invalidation lattice over
model identity, weight identity, route identity, continuation identity,
carrier-class recomposition, and publication overread.

The lock is honest about historical artifact shape. Older reports that only
carry part of the tuple do not get silently upgraded by prose; they stay green
only because the new lock binds them machine-readably by source ref and digest.
Reports that already carry the full tuple stay green as direct projections of
that same canonical machine.

This tranche therefore freezes one clear current boundary:

- one canonical post-article machine tuple is now explicitly named
- route, proof, witness, receipt, controller, and bounded platform surfaces are
  explicitly bound to that tuple
- mixed direct-versus-resumable carrier recomposition remains fail-closed
- plugin publication, served/public universality, and arbitrary software
  capability remain out of scope
- the final claim-bearing canonical machine closure bundle still remains
  separate for `TAS-215`

The companion operator summary lives at
`fixtures/tassadar/reports/tassadar_post_article_canonical_machine_identity_lock_summary.json`,
the provider receipt lives at
`crates/psionic-provider/src/tassadar_post_article_canonical_machine_identity_lock.rs`,
and the dedicated checker lives at
`scripts/check-tassadar-post-article-canonical-machine-identity-lock.sh`.
