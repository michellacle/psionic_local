# PSION Plugin Guest Capability-Boundary Benchmark

> Status: canonical `PSION_PLUGIN-27` guest-plugin benchmark package for the
> bounded mixed plugin-conditioned tranche, written 2026-03-22 after landing
> the first repo-owned guest capability-boundary package and receipt on top of
> the shared `psion_plugin` benchmark contract.

This document freezes the first guest-plugin benchmark family for the Psion
plugin convergence program.

The package is benchmark-authored on purpose.

The current mixed lane has one admitted guest-artifact train example, but it
does not yet have a held-out guest-artifact benchmark lineage. This package
therefore keeps authored prompt provenance explicit instead of pretending those
items came from held-out execution truth.

## Canonical Artifacts

- `docs/PSION_PLUGIN_GUEST_PLUGIN_BENCHMARK.md` is the canonical
  human-readable contract.
- `crates/psionic-train/src/psion_plugin_guest_plugin_benchmark.rs` owns the
  package and receipt builders.
- `crates/psionic-train/examples/psion_plugin_guest_plugin_benchmark.rs`
  writes the canonical bundle.
- `fixtures/psion/benchmarks/psion_plugin_guest_plugin_benchmark_v1/`
  carries the first committed bundle.

## Coverage

The first package covers:

- admitted digest-bound guest-artifact use
- unsupported guest-artifact load claims
- guest-artifact publication overclaims
- arbitrary-binary overclaims
- served-public universality overclaims

This is enough to close the first guest capability-boundary package without
claiming held-out guest execution evals, guest-artifact result interpretation,
or any broader guest publication posture.

## Boundary

Each item now preserves:

- the expected route
- expected delegated plugin ids when the admitted guest plugin is in scope
- accepted refusal reason codes for unsupported claims
- required capability-boundary ids that must remain explicit in the answer
- forbidden capability-boundary ids that must stay absent
- the guest-capability scenario kind being tested

That keeps admitted guest use separate from unsupported loading, publication,
arbitrary-binary, and served-universality claims instead of flattening them
into one vague “guest plugins exist” story.

## Receipt Surface

The package emits one shared plugin benchmark receipt with:

- guest admitted-use accuracy
- unsupported guest-load refusal accuracy
- guest publication-boundary accuracy
- arbitrary-binary boundary accuracy
- served-universality boundary accuracy

That keeps the bounded guest lane machine-readable enough for the later mixed
capability-matrix issue.
