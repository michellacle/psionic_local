# HOMEGOLF Explicit Grad-Accum Queue Correction Audit

Date: 2026-03-28

## Scope

This audit records one operator-surface correction in the repo-owned local
HOMEGOLF loop.

The detached-score queue was real, but one naming assumption inside that loop
was wrong:

- multiple queued run ids encoded `g64`
- the repo-owned runner never exposed an explicit grad-accum override

That made the queue labels stronger than the real operator contract.

## What Changed

Updated:

- `scripts/run-parameter-golf-homegolf-local-cuda.sh`
- `scripts/queue-parameter-golf-homegolf-local-cuda.sh`

New operator control:

- `--grad-accum-steps <n>`

The runner now:

- records `GRAD_ACCUM_STEPS` in `launch.env`
- exports `PSIONIC_PARAMETER_GOLF_HOMEGOLF_GRAD_ACCUM_STEPS=<n>`

That landed at commit `a5f10015`.

## Why This Improves The System

Before this change:

- the trainer supported the grad-accum override already
- the repo-owned runner did not expose it
- queued run ids like `g64` were only labels unless the trainer default
  happened to match

After this change:

- the repo-owned runner can state the intended grad-accum contract directly
- retained operator evidence now records that contract in `launch.env`
- future queued HOMEGOLF runs can truthfully distinguish explicit `g64` from
  the silent default posture

## Live Evidence

At `2026-03-28 07:36:00 CDT` on `archlinux`, the active detached-score control
run
`homegolf-baseline-g64-stepcap2-clip1-lr075-artifactonly-600s-20260328f`
showed:

- `launch.env` retained:
  - `FINAL_VALIDATION_MODE=artifact_only`
  - `GRAD_CLIP_NORM=1.0`
  - `LEARNING_RATE_SCALE=0.75`
- `launch.env` did not retain `GRAD_ACCUM_STEPS`
- the live trainer log still reported `grad_accum_steps=32`

So `20260328f` is a useful control run, but not an explicit `g64` receipt.

One more operational correction was explicit in the same pass:

- the previously documented queued `20260328g`, `20260328h`, and `20260328i`
  waiters were no longer alive on `archlinux`

The first actually live explicit-`g64` queue is now:

- `20260328j`, queue pid `786146`, waiting on active trainer pid `784939`
- `20260328k`, queue pid `786219`, waiting on `20260328j/train.pid`
- `20260328l`, queue pid `786291`, waiting on `20260328k/train.pid`
- `20260328m`, queue pid `786363`, waiting on `20260328l/train.pid`

All four new queue processes were retained live in `pgrep -af` output with
explicit `--grad-accum-steps 64`.

## Honest Boundary

This audit does not add a completed new score receipt.

Current retained truth:

- `20260328f` is still alive as the detached-score control run
- the first live explicit `g64` sweep is now `j`, `k`, `l`, `m`
- the retained actual public PGOLF score is still `6.306931747817168`
- XTRAIN did not change in this pass
