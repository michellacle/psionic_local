# Psion Normative Spec-Reading Benchmark

> Status: canonical `PSION-22` / `#378` normative-reading benchmark contract,
> written 2026-03-22 after landing the first typed normative spec-reading
> package and its direct acceptance binding for `Psion`.

This document freezes the first dedicated normative spec-reading benchmark
package for the `Psion` learned-model lane.

It is distinct from engineering interpretation. The package is meant to test
what the source text explicitly says, not what an implementation engineer might
 infer from it later.

## Canonical Artifacts

- `fixtures/psion/benchmarks/psion_benchmark_catalog_v1.json` contains the
  canonical package row `psion_normative_spec_benchmark_v1`.
- `fixtures/psion/benchmarks/psion_benchmark_receipt_set_v1.json` contains the
  canonical package receipt for that benchmark package.
- `fixtures/psion/benchmarks/psion_benchmark_label_generation_receipt_set_v1.json`
  contains the canonical label-generation receipt for that package.
- `fixtures/psion/acceptance/psion_acceptance_matrix_v1.json` now binds the
  normative-reading requirements directly to
  `psion_normative_spec_benchmark_v1` instead of using the older blended
  spec/manual family.
- `crates/psionic-train/src/psion_benchmark_packages.rs` owns the typed
  normative item payloads inside the shared package contract.
- `crates/psionic-train/src/psion_acceptance_matrix.rs` owns the direct
  benchmark-artifact binding used by the acceptance matrix.

## Package Identity

The first dedicated package is:

- package id: `psion_normative_spec_benchmark_v1`
- package family: `normative_spec_reading`
- package digest:
  `dd5741c92863fe4525d67eadd513b955b01f97adb9aa0f1cfa22619e8b273b32`

## Typed Coverage

The canonical package now covers three normative probe kinds explicitly:

- exact definition
- named edge condition
- named guarantee

Each normative item now preserves:

- `normative_source_ref`
- `required_section_anchor`
- `probe_kind`
- `expected_fact`
- `grounded_reading_required`
- `engineering_inference_forbidden`

This keeps the package focused on source-grounded reading discipline instead of
accepting persuasive paraphrase or speculative implementation advice.

## Labels And Receipts

The normative package currently stays exact-label on the label side:

- the package receipt stays on the shared `PsionBenchmarkPackageReceipt`
  contract
- the label-generation receipt pins exact-truth bindings for the normative
  items and keeps the parent source ids explicit
- the acceptance-ready evidence now lands under the dedicated
  `normative_spec_reading` family instead of the previous blended
  spec/manual family

## Contamination Attachment

The package carries contamination attachment through the committed
contamination-input bundle:

- benchmark-visible source ids remain explicit
- held-out and training-excluded review inputs remain explicit
- the near-duplicate review reference remains explicit
- the package still only counts as green at promotion time when the phase gate
  carries a clean contamination review

## Acceptance Binding

`Psion` acceptance-matrix `v1` now binds the normative-reading gate directly to
the concrete benchmark package artifact above.

This keeps grounded reading separate from engineering inference in two ways:

- the normative package has its own acceptance family
- the engineering interpretation package is no longer reused as the acceptance
  proxy for source-grounded reading
