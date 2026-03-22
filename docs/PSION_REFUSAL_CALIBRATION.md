# Psion Refusal Calibration

> Status: canonical `PSION-26` / `#382` refusal-calibration contract,
> written 2026-03-22 after landing the first unsupported-envelope refusal
> package and capability-matrix-bound receipt for `Psion`.

This document freezes the first dedicated refusal-calibration package for the
`Psion` learned-model lane.

It is distinct from route selection. The package is meant to prove that the
lane refuses unsupported exactness asks, underspecified design asks, context
overflow, freshness-sensitive requests, and open-ended assistant chat with
named reason codes instead of vague hedging.

## Canonical Artifacts

- `fixtures/psion/benchmarks/psion_benchmark_catalog_v1.json` contains the
  canonical package row `psion_unsupported_request_refusal_benchmark_v1`.
- `fixtures/psion/benchmarks/psion_benchmark_receipt_set_v1.json` contains the
  canonical package receipt for that benchmark package.
- `fixtures/psion/benchmarks/psion_benchmark_label_generation_receipt_set_v1.json`
  contains the canonical label-generation receipt for that package.
- `fixtures/psion/refusal/psion_refusal_calibration_receipt_v1.json` contains
  the canonical refusal-calibration receipt that later acceptance and
  capability-publication work can consume directly.
- `fixtures/psion/capability/psion_capability_matrix_v1.json` now carries the
  explicit unsupported exactness and missing-constraints regions that anchor
  the receipt.
- `crates/psionic-train/src/psion_benchmark_packages.rs` owns the typed
  refusal item payloads and exact refusal graders.
- `crates/psionic-train/src/psion_refusal_calibration.rs` owns the refusal
  receipt plus the minimal capability-matrix view used to validate it.

## Package Identity

The first dedicated package is:

- package id: `psion_unsupported_request_refusal_benchmark_v1`
- package family: `refusal_evaluation`
- package digest:
  `73274857b551ca89edd96a200c60d25992a0aaa62a5cb25a1c369f518e8c42c4`

## Typed Coverage

The canonical package now covers five explicit refusal probes:

- `unsupported_exactness_request`
- `missing_constraints`
- `over_context_envelope`
- `freshness_or_hidden_artifact_dependency`
- `unsupported_general_assistant_chat`

Each refusal item now preserves:

- `expected_reason_code`
- `refusal_boundary_ref`
- `probe_kind`
- `capability_region_id`
- `unsupported_region_evidence_ref`
- `claim_boundary_required`

This keeps refusal evaluation tied to one declared capability boundary instead
of turning it into a generic “safe answer” score.

## Refusal Receipt

The refusal receipt records one row per unsupported-envelope probe and keeps
four things explicit:

- the capability region the probe calibrates against
- the expected refusal reason code for that region
- the observed refusal accuracy for the unsupported request
- the observed reason-code match rate for that request

The aggregate receipt also carries one supported-control over-refusal rate and
one refusal-regression value, so acceptance and serving policy can cite the
same receipt without losing the false-positive side of the story.

## Capability Binding

The first capability matrix now includes two extra refusal-required regions:

- `unsupported_exact_execution_without_executor_surface`
- `underspecified_design_without_required_constraints`

Those sit beside the earlier over-context, freshness, and open-ended assistant
rows so the refusal benchmark calibrates against the published capability
matrix instead of an undocumented safety vibe.

## Acceptance Binding

`Psion` acceptance-matrix `v1` now binds the unsupported-request refusal gate
directly to the concrete package above for every phase that requires the
family.

The pilot and later promotion decisions can therefore cite one concrete
benchmark package digest and one concrete refusal-calibration receipt when they
justify capability publication or refusal-boundary preservation.
