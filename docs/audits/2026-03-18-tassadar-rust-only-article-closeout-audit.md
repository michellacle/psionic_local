# Tassadar Rust-Only Article Closeout Audit

Date: March 18, 2026

## Result

The repo now has one final machine-readable closeout audit for the current
Rust-only article claim.

That audit is subordinate to the underlying harness, gate, proof, and CPU
artifacts. It does not replace them. It binds them into one last publication
surface so the repo can say exactly what was reproduced, on which machines,
through which proof route, and with which remaining exclusions.

The committed closeout report is:

- `fixtures/tassadar/reports/tassadar_rust_only_article_closeout_audit_report.json`

The companion summary is:

- `fixtures/tassadar/reports/tassadar_rust_only_article_closeout_summary.json`

On the current committed truth both are green.

## Commands Run

- `cargo run -p psionic-eval --example tassadar_rust_only_article_closeout_audit_report`
- `cargo run -p psionic-research --example tassadar_rust_only_article_closeout_summary`
- `cargo test -p psionic-eval rust_only_article_closeout_audit -- --nocapture`
- `cargo test -p psionic-research rust_only_article_closeout_summary -- --nocapture`
- `./scripts/check-tassadar-rust-only-article-closeout-audit.sh`

## Decisive Outcomes

### 1. Final publication closure is now one binary surface

The closeout audit now turns green only when all of these are green:

- the one-command reproduction harness
- the Rust-only article acceptance gate v2
- the direct model-weight execution proof surface
- the CPU reproducibility matrix

The committed report records:

- `harness_green = true`
- `acceptance_gate_green = true`
- `direct_model_weight_proof_green = true`
- `cpu_reproducibility_green = true`
- `all_surfaces_green = true`
- `green = true`

That is the right shape. Partial success on one upstream surface no longer
looks like final closure.

### 2. The direct in-model proof route is frozen explicitly

The closeout audit now binds the final claim to:

- `direct_route_descriptor_digest = 1f1f6784ad8b8e8ea59f0b0ef5480cd47844aa8c3de650a6e640d4fbf32457fc`
- `direct_proof_case_ids = [long_loop_kernel, sudoku_v0_test_a, hungarian_matching]`

It also re-checks the route posture and execution posture underneath those
cases:

- requested decode remains `reference_linear`
- effective decode remains `reference_linear`
- selection state remains `direct`
- fallback remains unobserved
- external call count remains `0`
- external tool surface remains unobserved
- CPU result substitution remains unobserved
- route posture remains `direct_guaranteed`

That keeps the "inside the model weights" claim tied to the committed route and
receipts instead of drifting into a broader executor claim.

### 3. The machine boundary is now explicit in the same closeout artifact

The closeout audit now carries the current host class and the supported CPU
classes directly:

- `current_host_machine_class_id = host_cpu_aarch64`
- `supported_machine_class_ids = [host_cpu_aarch64, host_cpu_x86_64]`

It also re-checks the portability boundary facts from the CPU matrix:

- current host remains measured green
- `other_host_cpu` remains the only declared unsupported CPU class
- the historical optional C path remains `compile_refused`
- the optional C-path refusal remains `toolchain_unavailable`
- the optional C path remains non-blocking for the Rust-only claim

That makes the final closeout report say exactly which machine classes are in
scope rather than relying on separate portability prose.

### 4. Remaining exclusions are now preserved inside the closeout artifact

The closeout report now preserves these exclusions explicitly:

- no arbitrary Rust or arbitrary Wasm closure beyond the committed Rust-only
  article family
- no non-CPU backend portability or backend-invariant closure
- no portability claim beyond `host_cpu_aarch64` and `host_cpu_x86_64`
- no claim beyond the committed canonical article workloads and the direct
  route-bound proof surface
- no world-mount, accepted-outcome, settlement, or market-grade closure from
  this audit

That is important because the article claim is now strong enough to be
over-read if those exclusions live only in scattered historical notes.

## Status Judgment

What is now closed:

- one-command operator procedure for the Rust-only article path
- explicit prerequisite closure over source canon, profile boundary, ABI
  closure, Hungarian, Sudoku, runtime closeout, direct proof, and CPU
  portability
- direct model-weight proof on the committed article cases and committed route
- explicit CPU portability boundary over `host_cpu_aarch64` and
  `host_cpu_x86_64`
- one final machine-readable closeout audit that turns the whole publication
  claim red if any upstream surface regresses

What remains outside this issue:

- arbitrary Rust frontend closure
- arbitrary Wasm module or ABI closure
- non-CPU backend closure
- broader portability beyond the two declared host CPU classes
- broader route widening beyond the committed proof route
- world-computer, accepted-outcome, settlement, or market-grade closure

## Final Judgment

The right current statement is:

`Psionic now reproduces the full Rust-only Percepta article claim end to end on the committed canonical article workloads, with one-command operator procedure, a green prerequisite gate, direct model-weight execution proof on the declared article cases, and explicit CPU portability closure on the declared host CPU classes.`

That is now true on the committed artifact set, and the closeout audit keeps it
narrow enough to stay honest.
