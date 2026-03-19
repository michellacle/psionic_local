# Tassadar Rust-Only Article CPU Reproducibility Audit

Date: March 18, 2026

## Result

The Rust-only article path now has a canonical CPU reproducibility matrix and
operator summary.

The current measured host is `host_cpu_aarch64`, and it reproduced the
Rust-only article runtime closeout with exact current-host posture and passed
throughput floors. The declared supported CPU classes are now explicit:
`host_cpu_aarch64` and `host_cpu_x86_64`. All other host CPU classes remain
explicitly unsupported.

The optional historical C-to-Wasm path remains a non-blocking boundary fact for
this Rust-only claim. On the current committed compile-pipeline matrix it still
surfaces as `compile_refused` with
`compile_refusal_kind=toolchain_unavailable`.

## Commands Run

- `cargo run -p psionic-eval --example tassadar_article_cpu_reproducibility_report`
- `cargo run -p psionic-research --example tassadar_article_cpu_reproducibility_summary`
- `cargo test -p psionic-runtime article_cpu_reproducibility -- --nocapture`
- `cargo test -p psionic-eval article_cpu_reproducibility -- --nocapture`
- `cargo test -p psionic-research article_cpu_reproducibility -- --nocapture`

## Decisive Outcomes

### 1. The current host is now measured explicitly

`fixtures/tassadar/reports/tassadar_article_cpu_reproducibility_report.json`
now records:

- `current_host_machine_class_id = host_cpu_aarch64`
- `current_host_os = macos`
- `current_host_arch = aarch64`
- `current_host_measured_green = true`

The measured row carries the runtime identity and digest rather than relying on
a doc-only statement.

### 2. Supported and unsupported CPU classes are explicit

The same report now freezes:

- supported classes:
  `host_cpu_aarch64`, `host_cpu_x86_64`
- unsupported class:
  `other_host_cpu`

The `host_cpu_x86_64` row is explicit as a declared supported class with a
required throughput floor and exactness posture, not as a silent implication
from the current `aarch64` host.

### 3. Throughput-floor posture is now recorded per machine class

The matrix rows now carry:

- exactness posture
- throughput-floor posture
- `throughput_floor_steps_per_second = 250000.0`
- `slowest_workload_horizon_id = state_machine_kernel.two_million_step`

That makes the portability claim concrete without baking volatile
machine-local measured steps-per-second values into the committed artifact.

### 4. The Rust toolchain posture is explicit and shared

The report now records the current Rust toolchain as:

- `compiler_family = rustc`
- target `wasm32-unknown-unknown`
- shared pipeline features:
  `compiler_binary:rustc`, `crate_type:cdylib`, `edition:2024`,
  `optimization:3`, `panic:abort`, `target:wasm32-unknown-unknown`

The report also records `rust_toolchain_case_count = 8`, proving the full
committed Rust article source canon is included in the portability surface.

### 5. The optional C path stays explicit and non-blocking

The report now binds the optional historical C path directly to the compile
pipeline matrix and keeps it out of the Rust-only success condition:

- `optional_c_path_case_id = c_missing_toolchain_refusal`
- `optional_c_path_status = compile_refused`
- `optional_c_path_blocks_rust_only_claim = false`

That is the correct boundary. The repo can now say the Rust-only article path
has CPU portability truth without smuggling the historical C frontend back into
the claim.

## Status Judgment

What is now closed:

- current-host CPU reproducibility for the Rust-only article path
- explicit supported vs unsupported CPU machine classes
- explicit throughput-floor posture for the declared CPU classes
- explicit shared Rust toolchain posture for the committed Rust article canon
- explicit non-blocking treatment of the optional C-path refusal

What remains outside this issue:

- acceptance-gate enforcement for requiring the new matrix
- final closeout audit for the full Rust-only article claim
- broader profile portability beyond the Rust-only article path
- non-CPU backend portability

## Final Judgment

The right current statement is:

`the Rust-only article path now has explicit CPU reproducibility truth, but only for the declared host CPU classes`

That is strong enough for operator portability truth and still narrow enough to
avoid one-host optimism or universal-portability overclaiming.
