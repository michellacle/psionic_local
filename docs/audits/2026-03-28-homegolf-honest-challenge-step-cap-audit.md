# HOMEGOLF Honest Challenge Step Cap Audit

Date: 2026-03-28

## Scope

This audit records the next operator and score-integrity correction after the
honest local `600s` relaunch.

The live `20260328b` run answered the open question from that relaunch:

- can the local `g64` baseline complete one honest optimizer step inside the
  declared `600s` contract?

The answer is yes.

But the same run also exposed the next quality blocker:

- the second optimizer step immediately diverges

## Retained Live Evidence

Run:

- host: `archlinux`
- run id: `homegolf-baseline-g64-roundtrip-600s-20260328b`

Observed retained live facts:

- `train_step_complete step=1 mean_microbatch_loss=8.29199219`
- `optimizer_step_ms=3803`
- `train_step_start step=2/20000`
- `micro_step=1/64 train_loss=13.90855503`
- `micro_step=2/64 train_loss=13.96454716`
- `micro_step=3/64 train_loss=13.86529732`
- `micro_step=4/64 train_loss=14.02747917`

So the current local `g64` baseline is not failing because it cannot fit a
first honest step.

It is failing because quality collapses as soon as the second optimizer step
begins.

## What Changed

Updated:

- `crates/psionic-train/src/bin/parameter_golf_homegolf_single_cuda_train.rs`
- `scripts/run-parameter-golf-homegolf-local-cuda.sh`

The local HOMEGOLF lane now supports an honest challenge-step cap:

- env:
  `PSIONIC_PARAMETER_GOLF_HOMEGOLF_MAX_CHALLENGE_STEPS=<n>`
- runner flag:
  `--challenge-max-steps <n>`

This is intentionally distinct from the older positional `--max-steps`
argument.

Why that distinction matters:

- positional `--max-steps` selects `bounded_proof_defaults`
- bounded proof defaults set `max_wallclock_seconds=None`
- that makes bounded runs useful for probes, but not honest local score
  receipts

The new challenge-step cap instead starts from the normal local HOMEGOLF
challenge defaults and only lowers `config.max_steps`.

So it preserves:

- `max_wallclock_seconds=600`
- `warmup_steps=0`
- `homegolf_local_cuda`
- the standard local HOMEGOLF accounting and closeout path

## Why This Improves The System

Compared with the prior iteration:

- the system no longer has to choose between:
  - one good local first step in a non-honest bounded probe
  - or one honest local run that continues into obviously worse second-step
    quality
- the operator now has one repo-owned way to keep the honest `600s` contract
  while stopping at the last known stable local step count

This does not yet prove that one capped honest step is the best local score.

It does make that hypothesis runnable without falling back to the bounded proof
lane.

## Honest Boundary After This Audit

What is true:

- the local HOMEGOLF loop now has one explicit honest challenge-step cap
- that cap preserves the real local `600s` score semantics
- the fix was motivated by retained live divergence evidence from `20260328b`

What is not true:

- this audit does not contain the new capped run receipt yet
- this audit does not improve the retained public PGOLF score directly
- the retained actual in-repo PGOLF score is still `6.306931747817168`
