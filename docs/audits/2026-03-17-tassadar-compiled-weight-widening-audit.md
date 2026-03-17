# Tassadar Compiled-Weight Widening Audit

Date: 2026-03-17

## Scope

This audit answers `PTAS-601`: has the Psionic-owned Tassadar
program-specialized compiled-weight lane widened beyond the older bounded
Sudoku-v0 and Hungarian-v0 corpora?

The answer is yes.

What is landed is still bounded and workload-shaped, not arbitrary-program
compile-to-weights closure. But the repo no longer stops at the older 4x4
proxy corpora.

## Canonical Evidence

- `fixtures/tassadar/runs/compiled_kernel_suite_v0/run_bundle.json`
- `fixtures/tassadar/runs/compiled_kernel_suite_v0/compiled_kernel_suite_exactness_report.json`
- `fixtures/tassadar/runs/compiled_kernel_suite_v0/compiled_kernel_suite_compatibility_report.json`
- `fixtures/tassadar/runs/compiled_kernel_suite_v0/compiled_kernel_suite_scaling_report.json`
- `fixtures/tassadar/runs/compiled_kernel_suite_v0/claim_boundary_report.json`
- `fixtures/tassadar/reports/tassadar_compiled_article_closure_report.json`
- `docs/ROADMAP_TASSADAR.md`

## Findings

The widened compiled-weight lane is already present in
`fixtures/tassadar/runs/compiled_kernel_suite_v0`.

That bundle extends the compiled/proof-backed deployment path across four
workload families under the article i32 profile:

- `arithmetic_kernel`
- `memory_update_kernel`
- `forward_branch_kernel`
- `backward_loop_kernel`

The widening is not just naming. Each deployment preserves:

- source `program_artifact.json`
- `compiled_weight_artifact.json`
- `runtime_contract.json`
- `compiled_weight_bundle.json`
- `compile_evidence_bundle.json`
- `model_descriptor.json`
- `runtime_execution_proof_bundle.json`
- `runtime_trace_proof.json`

The exactness and refusal posture are green on the widened set:

- exactness: `12/12` exact traces and `12/12` final-output matches
- compatibility: `48/48` matched expected refusals

The committed claim boundary is also already explicit:

- the widened lane is exact for bounded arithmetic, memory-update,
  forward-branch, and backward-loop families
- it widens compiled article evidence beyond Sudoku and Hungarian
- it is still not arbitrary-program closure by itself

That exact boundary is repeated in
`fixtures/tassadar/reports/tassadar_compiled_article_closure_report.json`,
which is already green and already treats the kernel suite as required
compiled article evidence rather than optional side data.

## Verdict

`PTAS-601` is implemented.

What remains partial is the broader Epic 6 story:

- shared direct-vs-compiled program-to-weights benchmarking
- article-workload served session surfaces
- planner-owned hybrid article workflows
- final article-parity closeout

Those are follow-on issues. The widening itself has already landed.
