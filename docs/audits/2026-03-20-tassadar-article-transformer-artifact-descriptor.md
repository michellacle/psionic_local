# TAS-168 Article Transformer Artifact Descriptor Audit

`TAS-168` closes the artifact-backed descriptor and weight-bundle tranche for
the canonical owned article route.

The landed split is explicit:

- `psionic-models` now owns the committed canonical and trace-bound
  article-Transformer descriptors plus safetensors bundles under
  `fixtures/tassadar/models/`
- `psionic-models` now owns descriptor save/load, explicit tensor inventory,
  and artifact roundtrip logic in
  `crates/psionic-models/src/tassadar_article_transformer.rs`
- `psionic-runtime` now owns the runtime-visible model-artifact binding fields
  in `crates/psionic-runtime/src/tassadar_article_transformer_forward_pass.rs`
- `psionic-eval` now owns the machine-readable closure report in
  `fixtures/tassadar/reports/tassadar_article_transformer_artifact_descriptor_report.json`
- `psionic-research` now owns the disclosure-safe operator summary in
  `fixtures/tassadar/reports/tassadar_article_transformer_artifact_descriptor_summary.json`

This means the owned article route now:

- loads the canonical paper-faithful and trace-bound article models from real
  committed safetensors artifacts instead of a fixture-only descriptor path
- keeps one explicit tensor inventory for the canonical encoder-decoder weight
  surface, including embeddings, encoder layers, decoder layers, and logits
  projection tensors
- preserves stable descriptor identity across save/load roundtrip even when the
  artifact is rewritten at a different file location
- binds runtime evidence and replay receipts to real model-artifact identity
  inputs, including descriptor digest, artifact id, weight-bundle digest, and
  primary safetensors SHA-256
- keeps the older fixture-backed lane in `crates/psionic-models/src/tassadar.rs`
  explicit as a non-canonical surface instead of silently reusing it for the
  article-equivalence route

This closure is still bounded.

It does not prove:

- reference-linear exactness on the Transformer-backed route
- fast-route promotion
- benchmark parity
- single-run closure
- final article-equivalence green status

Targeted validation for this tranche:

- `cargo run -p psionic-models --example tassadar_article_transformer_artifact_bundle`
- `cargo test -p psionic-models tassadar_article_transformer -- --nocapture`
- `cargo test -p psionic-runtime tassadar_article_transformer_forward_pass -- --nocapture`
- `cargo test -p psionic-eval article_transformer_artifact_descriptor -- --nocapture`
- `cargo test -p psionic-research article_transformer_artifact_descriptor_summary -- --nocapture`
- `cargo run -p psionic-eval --example tassadar_article_transformer_artifact_descriptor_report`
- `cargo run -p psionic-research --example tassadar_article_transformer_artifact_descriptor_summary`
