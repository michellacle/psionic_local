# TAS-176 Article Frontend/Compiler Envelope Audit

`TAS-176` closes one narrow but necessary public-claim boundary: the repo now
declares exactly which frontend/compiler route article-facing claims are
allowed to rely on.

The canonical public envelope is:

- Rust source only
- `rustc` targeting `wasm32-unknown-unknown`
- `#![no_std]` plus `#![no_main]`
- `core`-only source surface
- `cdylib`, `panic=abort`, opt-level `3`
- the already-bounded i32-oriented article ABI rows

The canonical machine-readable artifact is
`fixtures/tassadar/sources/tassadar_article_frontend_compiler_envelope_v1.json`.
The canonical machine-readable closure report is
`fixtures/tassadar/reports/tassadar_article_frontend_compiler_envelope_report.json`.

What this tranche closes:

- the repo no longer relies on implication to define the public article
  frontend/compiler boundary
- the admitted Rust article fixtures are checked against one declared envelope
- representative out-of-envelope rows now refuse explicitly for C ingress,
  std/alloc surface, host imports, UB dependence, and wider ABI shapes

What this tranche still does not close:

- corpus expansion across the broader article-shaped frontend space
- Hungarian and Sudoku demo parity through that declared envelope
- interpreter breadth, benchmark parity, single-run no-spill closure, clean
  weight causality, or final article-equivalence green status
