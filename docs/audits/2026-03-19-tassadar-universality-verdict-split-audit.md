# Tassadar Universality Verdict-Split Audit

Date: 2026-03-19

## Purpose

This audit freezes the explicit theorem-to-product split for the terminal
universality claim. It exists to stop the repo from collapsing theoretical
universality, operator-owned resumable execution, and served/public posture
into one widened statement.

Canonical machine-readable artifact:

- `fixtures/tassadar/reports/tassadar_universality_verdict_split_report.json`

## Current verdicts

- theory: green
- operator: green
- served: suppressed

## What theory-green means

- `TCM.v1` is declared and bound to the landed runtime contract
- the universal-machine witness construction is exact on the committed proof
  targets
- the minimal universal-substrate gate is green

Theory-green means the repo can honestly say there is one declared universal
substrate with explicit witness constructions under the declared semantic
model.

It does not mean the same thing as operator or served availability.

## What operator-green means

- the claim survives only inside named resumable process envelopes
- session-process execution is bounded and deterministic
- spill/tape continuation is explicit and replayable
- process-object artifacts are first-class and refusal truth remains explicit

Operator-green means operators have one bounded universality-capable lane under
checkpoint, spill/tape, and process-object semantics.

It does not mean arbitrary Wasm, broad ambient effects, or an unconstrained
served capability.

## Why served stays suppressed

- no named served universality profile is published
- the current served internal-compute profile remains
  `tassadar.internal_compute.article_closeout.v1`
- route constraints remain tied to narrower named profiles
- `kernel-policy` still owns canonical served universality publication policy
  outside standalone `psionic`
- `nexus` still owns canonical accepted-outcome issuance and settlement-
  qualified served closure outside standalone `psionic`

## Non-implications

This audit still does not imply:

- arbitrary Wasm execution
- broad served internal compute
- public universality publication
- settlement-qualified universality closure
- final Turing-completeness closeout

## Follow-on

The next and last terminal-contract step is the final Turing-completeness
closeout audit. That audit must cite only the declared substrate, the witness
suite, the minimal gate, this verdict split, portability envelopes, and
explicit refusal boundaries.
