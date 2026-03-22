# Psion Capability Matrix

> Status: canonical `PSION-6` / `#362` capability-publication contract, written
> 2026-03-22 after landing `docs/PSION_ACCEPTANCE_MATRIX.md`.

This document freezes the first served capability matrix for the `Psion`
learned-model lane and the contract for publishing or updating it.

It is not generic product copy.

It is the explicit publication surface that says what the served learned lane
supports directly, what must route elsewhere, what must refuse, and what
remains unsupported.

## Canonical Artifacts

- `docs/PSION_CAPABILITY_MATRIX.md` is the canonical human-readable contract.
- `fixtures/psion/capability/psion_capability_matrix_v1.json` is the canonical
  machine-readable artifact.
- `crates/psionic-serve/src/psion_capability_matrix.rs` owns the typed served
  artifact and publication validation.

## v1 Surface

Capability-matrix `v1` makes four posture classes explicit:

- `supported`
- `route_required`
- `refusal_required`
- `unsupported`

The initial artifact includes all four on purpose. The served lane is only
honest if route-required and refusal-required regions stay visible instead of
being collapsed into vague “partial support” language.

`PSION-26` widens the refusal-required side of the matrix with two extra
regions:

- `unsupported_exact_execution_without_executor_surface`
- `underspecified_design_without_required_constraints`

Those sit beside the earlier over-context and freshness rows so refusal
calibration can cite concrete unsupported regions instead of only a generic
unsupported-request percentage.

## Published Envelope

The matrix publishes two top-level envelopes:

- one context envelope with a direct-service window, a route-required boundary,
  and a hard-refusal boundary
- one latency envelope with bounded first-token and end-to-end bands

Those limits belong in the matrix itself because later capability withdrawal or
narrowing needs a stable baseline artifact, not oral history.

## Publication Contract

The served matrix validates against:

- one named `Psion` acceptance-matrix id and version
- one promoted phase-decision receipt
- every benchmark receipt referenced by that promoted decision
- replay, checkpoint, contamination, route-calibration, and refusal-calibration
  receipt refs from the same decision

Publication is rejected when:

- the linked acceptance decision is not `promoted`
- the receipt references drift from the linked decision
- the matrix omits route-required or refusal-required regions
- the context or latency envelope is internally inconsistent
- a region posture is contradictory, such as a supported row with refusal
  reasons or a route-required row without explicit handoff

The refusal-required rows now deliberately separate:

- exactness requests that must route through an explicit executor surface
- exactness requests that refuse because no such surface is published
- underspecified design asks that refuse because required constraints are
  missing
- context overflow, freshness-sensitive asks, and generic assistant asks

## Why This Matters

This closes the next governance step for the learned lane:

- one versioned served capability matrix now exists for `Psion`
- context and latency constraints are part of the published envelope
- publication links back to the acceptance-matrix version and the exact receipt
  set that justified it

Later issues still own richer serving behavior, public UI copy, broader route
selection, and capability withdrawal after real runs. This document only freezes
the first explicit capability publication contract.

`PSION-27` now binds per-output served provenance back to this matrix through
`docs/PSION_SERVED_EVIDENCE.md` and the example bundles in
`fixtures/psion/serve/`, so route-required, refusal-required, and supported
region ids do not have to be restated ad hoc in later serve or provider
surfaces.
