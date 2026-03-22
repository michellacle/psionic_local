# Psion Decentralized Contribution

> Status: canonical `PSION-30` / `#386` bounded decentralized-contribution
> contract, written 2026-03-22 after landing the first Psion-specific bundle on
> top of the existing adapter-window reference substrate.

This document freezes the first bounded decentralized contribution lane for
`Psion`.

It does not claim arbitrary public full-model all-reduce, arbitrary public
synchronous training, hidden contributor trust, or automatic served promotion
of contributed outputs.

It records one bounded adapter-delta contribution lane that reuses the repo's
existing cluster-backed window substrate, then binds those contributed outputs
back to the same trusted-cluster, reasoning-SFT, acceptance, capability, and
rollback discipline as the main `Psion` lane.

## Canonical Artifacts

- `docs/PSION_DECENTRALIZED_CONTRIBUTION.md` is the canonical human-readable
  contract.
- `crates/psionic-train/src/psion_decentralized_contribution.rs` owns the
  typed bundle, contributor-summary projection, and digest logic.
- `crates/psionic-train/examples/psion_decentralized_contribution_fixtures.rs`
  regenerates the canonical bundle.
- `fixtures/psion/decentralized/psion_decentralized_contribution_bundle_v1.json`
  is the canonical machine-readable bundle.

The stable schema version is `psion.decentralized_contribution_bundle.v1`.

## First Admitted Mode

The first admitted contribution mode is:

- `adapter_delta_window`

That choice is deliberate.

The repo already has a bounded decentralized adapter control plane with typed:

- contributor membership receipts
- window plans
- artifact receipts
- provenance-security receipts
- replay-aware window summaries
- sealed-window aggregation receipts

So the honest next step for `Psion` is to bind that bounded substrate back to
the main learned-model governance lane instead of skipping straight to a
stronger and unsupported full-model public-training claim.

## What The Bundle Freezes

The first bundle now binds:

- one trusted-cluster run artifact from `docs/PSION_TRUSTED_CLUSTER_RUN.md`
- one reasoning-SFT run artifact from `docs/PSION_REASONING_SFT.md`
- one `Psion` acceptance binding that names the current promoted decision and
  the later publication phase contributed outputs must still clear
- one `Psion` capability binding that names the current capability matrix and
  the rollback schema contributed outputs must still honor
- one reused two-window decentralized adapter reference report with contributor
  churn, replay-checked work, and bounded local policy promotion
- one contributor summary lane that keeps artifact receipts, security receipts,
  aggregation acceptance, and replay-checked window posture explicit per
  contribution

That means the repo now has one machine-readable answer to:

> what is the first decentralized contribution path that `Psion` is actually
> willing to name?

## Reused Substrate, New Binding

The embedded reference program is the already-existing bounded adapter-window
reference program from `psionic-train`.

That is not a hidden substitution.

It is the point of the bundle:

- reuse the real bounded decentralized contribution substrate
- keep the contribution mode explicit as `adapter_delta_window`
- bind the result back to `Psion` governance instead of pretending local
  window promotion is the same thing as main-lane publication

The local adapter-window promotion receipts therefore remain local window truth.

They do not become:

- a `Psion` phase-promotion decision
- a served capability publication
- a bypass around rollback or downgrade discipline

## Promotion Discipline

Contributed outputs still inherit the main-lane acceptance contract.

The first bundle records:

- one current promoted `Psion` decision reference
- one later required publication phase

That keeps the decentralized contribution lane honest:

- local window aggregation may promote a bounded local policy revision
- any contributed output that wants service or publication still has to clear
  the same `Psion` acceptance-matrix discipline as the main lane

The first bundle sets that required publication phase to `internal_serving`.

## Capability And Rollback Discipline

Contributed outputs also inherit the same capability and downgrade posture as
the main lane.

The first bundle therefore binds:

- one current capability-matrix id and version
- one required rollback schema version
- one reference rollback receipt from `docs/PSION_CAPABILITY_WITHDRAWAL.md`

This is the repo's explicit statement that decentralized contribution is not a
reason to weaken downgrade discipline.

If a contributed output later reaches service, it still has to live inside the
same capability matrix and the same rollback contract as the rest of `Psion`.

## Contributor Truth

The contributor summary lane now keeps each bounded contribution tied to:

- contribution id
- window id
- worker id
- staged artifact id plus artifact receipt digest
- security receipt id plus security receipt digest
- artifact disposition
- security disposition
- aggregation acceptance
- replay-checked window posture

That is intentionally narrower than "everyone trained the model."

It is enough to make contributor receipts and window constraints explicit
without widening the lane into unsupported public all-reduce or hidden trust
assumptions.

## Non-Goals

The first bundle explicitly does not claim:

- arbitrary public full-model all-reduce
- arbitrary public synchronous training over the full `Psion` checkpoint
- hidden contributor trust or hidden topology assumptions
- automatic served promotion from a bounded local window promotion receipt
- any bypass around `Psion` acceptance, capability, or rollback discipline

## Why This Matters

This closes the first bounded decentralized contribution step for the learned
lane:

- `Psion` now has one named decentralized contribution bundle
- contributor receipts and window constraints are explicit artifacts
- the lane stays bounded to adapter-delta windows instead of unsupported
  full-model public all-reduce claims
- promotion and rollback of contributed outputs are now explicitly tied back to
  the same acceptance, capability, and withdrawal contracts as the main lane
