# Psion Capability Withdrawal

> Status: canonical `PSION-29` / `#385` capability-withdrawal and
> regression-rollback contract, written 2026-03-22 after landing the first
> served capability, evidence, and claim-posture artifacts.

This document freezes the first explicit rollback contract for the `Psion`
learned-model lane.

Promotion is not one-way.

Published checkpoints, capability matrices, and served claim surfaces now have
one machine-readable downgrade path when rights posture changes, contamination
is discovered, deterministic replay drifts, or route or refusal behavior falls
outside the accepted band.

## Canonical Artifacts

- `docs/PSION_CAPABILITY_WITHDRAWAL.md` is the canonical human-readable
  contract.
- `crates/psionic-serve/src/psion_capability_withdrawal.rs` owns the typed
  rollback receipt, history validation, and digest logic.
- `crates/psionic-serve/examples/psion_capability_withdrawal_fixtures.rs`
  writes the canonical example receipts.
- `fixtures/psion/withdrawal/psion_capability_withdrawal_rights_change_v1.json`
  demonstrates source-rights narrowing that withdraws the served checkpoint and
  depublishes the direct claim surface.
- `fixtures/psion/withdrawal/psion_capability_withdrawal_contamination_v1.json`
  demonstrates contamination-triggered rollback to a prior clean checkpoint
  target plus retraining and depublication follow-on analysis.
- `fixtures/psion/withdrawal/psion_capability_withdrawal_replay_failure_v1.json`
  demonstrates deterministic replay drift that withdraws publication and rolls
  serving back to a replay-clean checkpoint target.
- `fixtures/psion/withdrawal/psion_capability_withdrawal_route_regression_v1.json`
  demonstrates route regression that removes the exactness route from the
  current endpoint and converts the served surface to typed refusal.
- `fixtures/psion/withdrawal/psion_capability_withdrawal_refusal_regression_v1.json`
  demonstrates refusal regression that clamps the exactness route to explicit
  refusal until recalibration succeeds.

The stable schema version is `psion.capability_withdrawal_receipt.v1`.

## Trigger Families

The first contract recognizes five rollback triggers:

- `rights_change`
- `contamination_discovered`
- `replay_failure`
- `route_regression`
- `refusal_regression`

The trigger is not free text.

Each one binds to a specific evidence shape:

- source-impact analysis plus corrected source-manifest reference for rights or
  contamination changes
- replay-verification drift for deterministic replay failure
- observed route-calibration plus route-class evaluation artifact for route
  regression
- observed refusal-calibration receipt for refusal regression

## Rollback Surface

Each withdrawal receipt now carries four explicit downgrade lanes:

- one published capability-matrix reference with stable matrix digest
- one checkpoint plan that either withdraws the served checkpoint or rolls back
  to one prior target
- one capability-matrix history block
- one served-claim history block

Matrix history is explicit on purpose.

The first contract only allows:

- `withdraw_publication`
- `downgrade_region_posture`

That means the repo can now withdraw a stale matrix or tighten one or more
regions without pretending the old matrix never existed.

Served-claim history is equally explicit.

The first contract only allows:

- `depublish`
- `narrow_visible_claims`
- `change_behavior_visibility`

This keeps rollback visible on the serving surface itself instead of only in a
governance note.

## Source-Driven Rollback

Rights or contamination triggers must cite one
`PsionSourceImpactAnalysisReceipt`.

That receipt must already require:

- `capability_matrix_review`
- `depublication_review`

and may also require:

- `retraining_review`
- `benchmark_invalidation_review`

The rollback receipt then turns those lifecycle actions into concrete publication
effects:

- checkpoint withdrawal or rollback
- matrix withdrawal or downgrade
- served-claim depublication or narrowing
- follow-on analysis references

Rights and contamination triggers also require an explicit
`source_manifest_correction` follow-on analysis so publication recovery stays
bound to corrected source lineage rather than to oral history.

## Regression-Driven Rollback

Replay, route, and refusal regressions validate against the same accepted band
that originally justified publication.

The first contract therefore rejects:

- replay verification that did not actually drift
- route receipts that stayed inside the accepted route band
- refusal receipts that stayed inside the accepted refusal band

Regression rollback is not allowed to be metadata-only.

The receipt must also include both:

- one matrix-history downgrade
- one served-claim-history downgrade

so route or refusal regressions cannot be waved away while the published matrix
and serving surface stay unchanged.

## Follow-On Analyses

Rollback receipts now carry explicit follow-on analyses rather than only one
summary paragraph.

The first contract recognizes:

- `source_manifest_correction`
- `bounded_retraining_analysis`
- `depublication_analysis`

Each one cites the artifact that triggered it and may also cite the planned
analysis artifact id and digest.

This is the `Psion` answer to "what happens after withdrawal?"

The answer is now typed:

- source-driven rollback points back to corrected source lineage
- checkpoint or replay regressions point forward to bounded retraining review
- publication-impacting rollback points forward to explicit depublication review

## Why This Matters

This closes the first downgrade-governance step for the learned lane:

- `Psion` now has one typed rollback receipt instead of ad hoc operator notes
- rights, contamination, replay, route, and refusal regressions all map to
  explicit downgrade actions
- matrix history and serving-surface claim history are preserved together
- bounded retraining and depublication analysis can now be triggered directly
  from recorded rollback artifacts
