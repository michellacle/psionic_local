# PSION Plugin Argument Construction Benchmark

> Status: canonical `PSION_PLUGIN-9` argument-construction benchmark package
> for the bounded host-native plugin tranche, written 2026-03-22 after landing
> the first repo-owned packet-schema-aware package and receipt on top of the
> shared `psion_plugin` benchmark contract.

This document freezes the first benchmark family for typed plugin argument
construction.

The package is intentionally mixed:

- most cases are benchmark-authored because the current held-out split does not
  cover every missing-input and malformed-structure path the package needs
- one case is held-out lineage-backed so typed runtime refusal remains tied to
  real execution receipts instead of benchmark fiction

## Canonical Artifacts

- `docs/PSION_PLUGIN_ARGUMENT_CONSTRUCTION_BENCHMARK.md` is the canonical
  human-readable contract.
- `crates/psionic-train/src/psion_plugin_argument_construction_benchmark.rs`
  owns the package and receipt builders.
- `crates/psionic-train/examples/psion_plugin_argument_construction_benchmark.rs`
  writes the canonical bundle.
- `fixtures/psion/benchmarks/psion_plugin_argument_construction_benchmark_v1/`
  carries the first committed bundle.

## Coverage

The first package covers:

- schema-correct single-field argument construction
- schema-correct multi-field argument construction
- missing required input that should trigger request-for-structure
- wrong-type single-field input that should trigger corrected structure
- malformed multi-field structure that should trigger corrected structure
- one held-out typed runtime-refusal case with execution evidence

This is enough to close the first argument-construction package without
claiming sequencing, discovery, unsupported-plugin refusal, or guest-artifact
coverage.

## Packet-Schema Boundary

The shared argument grader now preserves:

- the target tool name
- required argument paths
- required JSON value types for those paths
- forbidden argument paths
- whether request-for-structure is an allowed outcome

That keeps argument grading bound to typed packet contracts instead of raw
string snippets.

## Provenance Boundary

Benchmark-authored items cite:

- the shared contamination bundle ref and digest
- one explicit authored benchmark prompt ref
- no runtime receipt requirement

The held-out runtime-refusal item cites:

- one held-out parent-lineage row from the contamination bundle
- the source-case id from that row
- the execution receipt ref required to score the typed refusal

That keeps runtime-refusal evidence honest while still allowing package breadth
beyond the current held-out split.

## Receipt Surface

The package emits one shared plugin benchmark receipt with:

- route accuracy
- argument-schema accuracy
- request-for-structure accuracy
- typed runtime-refusal accuracy

That keeps packet correctness, structure requests, and execution-backed refusal
evidence visible as separate dimensions.
