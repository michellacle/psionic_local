# Psionic Tassadar TAS Sync

This file maps the public `TAS-*` GitHub issues in `OpenAgentsInc/psionic` to
the concrete repo-owned implementation and artifact surfaces that close them.
It exists so issue closure stays tied to landed code, tests, and committed
artifacts instead of comment-only summaries.

Ownership note: the canonical executor-trace schema is runtime-owned in
`psionic-runtime`, not `psionic-ir`, because it is an execution ABI and proof
surface rather than a generic graph/program IR contract.

## Implemented

| Issue | Status | Repo evidence |
| --- | --- | --- |
| `TAS-001` / `#65` | implemented | `psionic-runtime` now carries canonical typed trace-step, trace-artifact, trace-proof, and trace-diff surfaces, plus deterministic replay and JSON round-trip tests; `psionic-provider` now exposes provider-facing trace-artifact and trace-diff receipts above those runtime-owned artifacts. |
| `TAS-002` / `#66` | implemented | The CPU-reference executor harness already spans `psionic-runtime`, `psionic-eval`, and `psionic-environments`; the validation corpus now also includes an explicit bounded shortest-path fixture beside arithmetic, memory, branch, Sudoku, Hungarian, and broader Wasm-like workloads, so the public issue scope is covered by committed golden cases and parity helpers. |
