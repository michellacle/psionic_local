# Tassadar Wasm Flow Status Audit

Date: March 18, 2026

## Result

The bounded Tassadar Wasm flow is operational on the current `main` checkout.

The repo-owned WAT/Wasm ingress, conformance, module-scale, and trap/refusal
surfaces all reproduced and matched committed truth. The one local failure was
the optional canonical C-to-Wasm compile step: this machine's `clang 17.0.0`
does not provide a working `wasm32-unknown-unknown` target, so the receipt
refused with a typed `toolchain_failure`.

That local toolchain refusal does not invalidate the current bounded Wasm lane.
It means this machine cannot reproduce the canonical C-source compile success
without a different local toolchain.

## Commands Run

- `cargo run -p psionic-runtime --example tassadar_c_to_wasm_compile_receipt`
- `cargo run -p psionic-eval --example tassadar_compile_pipeline_matrix_report`
- `cargo run -p psionic-eval --example tassadar_wasm_module_ingress_report`
- `cargo run -p psionic-eval --example tassadar_wasm_conformance_report`
- `cargo run -p psionic-eval --example tassadar_module_scale_workload_suite_report`
- `cargo run -p psionic-eval --example tassadar_trap_exception_report`
- `cargo test -p psionic-eval wasm_module_ingress -- --nocapture`
- `cargo test -p psionic-eval wasm_conformance -- --nocapture`
- `cargo test -p psionic-eval module_scale_workload_suite -- --nocapture`
- `cargo test -p psionic-eval trap_exception -- --nocapture`

## Decisive Outcomes

### 1. Canonical C-source compile receipt refused locally

`fixtures/tassadar/reports/tassadar_c_to_wasm_compile_receipt.json` rebuilt to
a typed refusal on this machine:

- `compiler_family = clang`
- `compiler_version = 17.0.0`
- refusal kind: `toolchain_failure`
- stderr excerpt: no available targets are compatible with
  `wasm32-unknown-unknown`

I restored the committed receipt after the run so the repo does not absorb a
machine-local toolchain fact as canonical truth.

### 2. The bounded compile-pipeline matrix is green where it should be green

`fixtures/tassadar/reports/tassadar_compile_pipeline_matrix_report.json`
reproduced with:

- `wasm_text_multi_export_arithmetic_exact = lowered_exact`
- `wasm_text_memory_lookup_exact = lowered_exact`
- `wasm_text_param_abi_lowering_refusal = lowering_refused`
- `c_missing_toolchain_refusal = compile_refused`

This is the right split. The WAT-driven bounded lowering lane is operational,
and the unsupported parameter ABI plus missing-toolchain edges remain explicit.

### 3. Real-binary ingress is still bounded, synthetic bounded ingress is exact

`fixtures/tassadar/reports/tassadar_wasm_module_ingress_report.json`
reproduced with:

- `canonical_micro_wasm_kernel = lowering_refused`
- refusal kind: `unsupported_param_count`
- `seeded_multi_function_module = lowered_exact`

That means the real committed binary is still useful as a parse and normalize
fixture, but the current bounded lowering lane still only admits zero-parameter
exports.

### 4. Differential conformance is green on the supported subset

`fixtures/tassadar/reports/tassadar_wasm_conformance_report.json`
reproduced with `11` cases total:

- `7` `exact_success`
- `3` `exact_trap_parity`
- `1` `boundary_refusal`

The supported i32 global/table/call-indirect subset still matches `wasmi`, and
the unsupported host-import edge still refuses explicitly instead of silently
degrading.

### 5. Module-scale Wasm workloads are live on the bounded lane

`fixtures/tassadar/reports/tassadar_module_scale_workload_suite_report.json`
reproduced with:

- `memcpy_fixed_span_exact = lowered_exact`
- `checksum_fixed_span_exact = lowered_exact`
- `parsing_token_triplet_exact = lowered_exact`
- `vm_style_dispatch_exact = lowered_exact`
- `vm_style_param_refusal = lowering_refused`

So the module-scale workload suite is not only a schema or plan. Four concrete
module families execute exactly today, and the parameter-ABI edge still refuses
honestly.

### 6. Trap and refusal parity is explicit and drift-free

`fixtures/tassadar/reports/tassadar_trap_exception_report.json` reproduced
with:

- `exact_success_parity_case_count = 1`
- `exact_trap_parity_case_count = 2`
- `exact_refusal_parity_case_count = 2`
- `drift_case_count = 0`

The important point is not just that success works. Bounds faults,
indirect-call failures, malformed imports, and unsupported-profile requests are
all preserved as first-class parity surfaces.

## Status Judgment

What is working now:

- bounded WAT-to-Wasm-to-Tassadar lowering for the currently supported exact
  subset
- bounded module ingress and exact execution on the seeded multi-function and
  module-scale suites
- differential Wasm conformance on the supported subset against `wasmi`
- explicit trap parity and refusal parity on the audited non-success cases

What is not working on this machine:

- successful canonical C-source compile receipt generation, because the local
  `clang` lacks the Wasm target

What remains bounded even when the current flow is green:

- zero-parameter export restriction in the bounded lowering lane
- parameter-ABI refusal for the still-unsupported path
- unsupported host-import boundary
- no arbitrary Wasm or broad ABI closure claim

## Final Judgment

The right current statement is:

`the bounded Tassadar Wasm lane is live and reproducible, but it is still a bounded lane`

That claim is supported by the exact compile-pipeline WAT cases, exact module
ingress cases, exact module-scale cases, green conformance on the supported
subset, and explicit trap/refusal parity. The remaining local blocker is a
toolchain prerequisite for the optional C-source compile step, not evidence
that the bounded Wasm executor substrate failed.
