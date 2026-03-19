# Psionic Parameter Golf External Non-Record PR

> Status: canonical `PGOLF-607` / `#251` first real external non-record PR
> record, updated 2026-03-19 after opening
> `https://github.com/openai/parameter-golf/pull/119`.

This document records the first real upstream Parameter Golf submission loop
for Psionic's non-record lane.

It exists so the exact folder, fork branch, PR URL, verifier evidence, and
maintainer-facing accounting question do not live only in GitHub comments or
terminal history.

## What Landed

Psionic has now opened one real upstream non-record PR:

- upstream PR: `https://github.com/openai/parameter-golf/pull/119`
- upstream repo: `openai/parameter-golf`
- fork branch: `OpenAgentsInc:psionic-non-record-runtime-replay-v2`
- submitted folder:
  `records/track_non_record_16mb/2026-03-18_psionic_local_reference_runtime_replay_v2`

The machine-readable receipt now lives at:

- `fixtures/parameter_golf/reports/parameter_golf_external_non_record_pr.json`

## Preserved Evidence

The receipt freezes:

- the exact fork commit and record-folder tree hash used for the PR
- the canonical Psionic final-PR-bundle report digest
- the canonical Psionic local-clone dry-run report digest
- the staged-folder compatibility-verifier digest
- the staged-folder replay-verifier digest
- the PR URL and current PR state

That means later external submissions can start from one committed reference
instead of reconstructing the first submission path from memory.

## Maintainer-Facing Accounting Question

The first upstream submission also carries the explicit counted-runtime
question that still blocks `PGOLF-608` / `#252` from closing.

That question is now asked on the live PR:

- `https://github.com/openai/parameter-golf/pull/119#issuecomment-4092766684`

Current posture:

- the question is asked
- no maintainer answer exists yet
- Psionic still keeps record-track accounting posture blocked on explicit
  upstream response rather than assumption

## Current Honest Boundary

This closes the first real external non-record PR loop only.

It does not claim:

- maintainer acceptance
- record-track readiness
- counted-runtime rule clarity
- real `8xH100` success

The PR stays explicitly non-record until stronger evidence and upstream rule
clarity actually exist.
