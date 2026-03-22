# PSION Plugin Sequencing And Multi-Call Benchmark

> Status: canonical `PSION_PLUGIN-10` sequencing-and-multi-call benchmark
> package for the bounded host-native plugin tranche, written 2026-03-22 after
> landing the first repo-owned sequence-plan package and receipt on top of the
> shared `psion_plugin` benchmark contract.

This document freezes the first benchmark family for plugin sequencing and
multi-call planning.

The package is mixed on purpose:

- benchmark-authored cases cover serial and parallelizable plans the current
  held-out split does not yet span by itself
- one held-out lineage-backed case keeps the typed-runtime-refusal stop
  boundary receipt-backed instead of benchmark-authored

## Canonical Artifacts

- `docs/PSION_PLUGIN_SEQUENCING_BENCHMARK.md` is the canonical human-readable
  contract.
- `crates/psionic-train/src/psion_plugin_sequencing_benchmark.rs` owns the
  package and receipt builders.
- `crates/psionic-train/examples/psion_plugin_sequencing_benchmark.rs` writes
  the canonical bundle.
- `fixtures/psion/benchmarks/psion_plugin_sequencing_benchmark_v1/` carries
  the first committed bundle.

## Coverage

The first package covers:

- serial fetch-then-transform plans
- serial fetch-then-parse plans
- parallelizable independent two-tool work
- continuation-until-exhausted full-pipeline plans
- one typed-runtime-refusal stop boundary tied to held-out receipts

This is enough to close the first sequencing package without claiming argument
grading, unsupported-plugin refusal, or guest-artifact sequencing coverage.

## Sequence Boundary

Each item now preserves:

- the admitted tool set
- the expected tool sequence
- whether the work is serial or parallelizable
- whether the model should stop after the goal, continue through the full
  requested bounded pipeline, or stop on typed runtime refusal

That keeps sequence correctness separate from argument correctness and makes
continuation behavior explicit instead of hand-waved.

## Provenance Boundary

Benchmark-authored items cite:

- the shared contamination bundle ref and digest
- one explicit authored benchmark prompt ref
- no runtime receipt requirement

The held-out refusal-stop item cites:

- one held-out parent-lineage row from the contamination bundle
- the source-case id from that row
- the full receipt set required to score the stop boundary honestly

That keeps receipt and continuation truth visible in evaluation output without
pretending the current held-out split already spans every desired sequence
shape.

## Receipt Surface

The package emits one shared plugin benchmark receipt with:

- sequence-plan accuracy
- continuation-boundary accuracy
- typed runtime-refusal accuracy

That keeps sequence order, stop/continue posture, and refusal-stop evidence as
separate scored dimensions.
