# Psionic Parameter Golf Restricted-Attention Report

> Status: canonical `PGOLF-613` / `#257` restricted-attention evidence record,
> updated 2026-03-19 after landing
> `crates/psionic-research/src/parameter_golf_restricted_attention_report.rs`
> and the committed challenge-format fixture at
> `fixtures/parameter_golf/parity/parameter_golf_seq1024_validation_fixture.json`.

This document records the first bounded locality or restricted-attention report
for Parameter Golf.

It exists to close the earlier "locality might help" bucket with one explicit
machine-readable result on a real `seq_len=1024` challenge-format validation
window instead of on the toy local-reference fixture.

## What Landed

`psionic-research` now exposes:

- `ParameterGolfRestrictedAttentionReport`
- `build_parameter_golf_restricted_attention_report()`
- `write_parameter_golf_restricted_attention_report(...)`

The committed machine-readable artifacts now live at:

- `fixtures/parameter_golf/parity/parameter_golf_seq1024_validation_fixture.json`
- `fixtures/parameter_golf/reports/parameter_golf_restricted_attention_report.json`

## Frozen Input Surface

The report does not fetch live challenge data at runtime.

Instead it freezes one committed input surface built from the local cached
challenge assets:

- the full `SP1024` tokenizer metadata from
  `~/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`
- the first committed `1025` validation tokens from
  `~/code/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin`

That gives Psionic one public-safe `seq_len=1024` evaluation slice with exact
byte-accounting metadata without pretending to cover the whole challenge
validation split.

## Result

The report compares the frozen dense baseline row against one fixed
restricted-attention proxy with a `256`-token causal window.

The outcome is explicit negative evidence:

- dense baseline on the committed slice: `val_bpb = 4.558717170348182`
- restricted-attention window `256`: `val_bpb = 4.609091082071537`
- delta: `+0.050373911723355`
- analytic attention-score terms fall to `0.437317073170732` of the dense row
- artifact bytes stay unchanged because the proxy reuses the same weights and
  the same counted code surface

So the current fixed-window locality cut improves analytic attention workload
but hurts the metric on this committed challenge-format slice.

## Honest Boundary

This report is still research-only.

It does not claim:

- retrained restricted-attention closure
- full validation-split closure
- single-H100 or `8xH100` speed closure
- record-track promotion

The correct conclusion today is narrower:

- Psionic now owns a public-safe restricted-attention eval path on real
  `seq_len=1024` challenge-format windows
- the first fixed-window candidate is not a win under that evidence
- any future locality promotion must beat this negative-evidence floor rather
  than leaning on generic efficiency intuition
