# HOMEGOLF Chained LR Sweep Queue Audit

Date: 2026-03-28

## Scope

This audit records the next operational improvement after detached artifact
score closeout.

Detached scoring removed final validation from the critical path of the next
training run, but the operator loop still only staged one future run at a
time.

That was not enough for a multi-hour autonomous iteration loop.

## What Changed

Updated:

- `scripts/queue-parameter-golf-homegolf-local-cuda.sh`

New queue behavior:

- `--wait-for-pid-file`

When enabled, the queue runner no longer requires the target run root to
already contain a `train.pid`.

It now waits until:

1. `<run-root>/train.pid` appears
2. that queued run's process exits
3. the next repo-owned HOMEGOLF run launches

This improvement landed at commit `12f60f5b`.

## Why This Improves The System

Before this change:

- one queued run could wait behind the current active run
- a second queued run behind a future run root was not possible without manual
  intervention

After this change:

- a real chained sweep can be staged ahead of time
- the local HOMEGOLF loop can continue iterating without a human requeueing
  every next candidate
- detached score closeout stays attached to each run, so GPU training remains
  decoupled from score receipt production

## Live Chain

At `2026-03-28 06:42:19 CDT` on `archlinux`, the remote chain was:

- active current run:
  `homegolf-baseline-g64-stepcap1-clip1-600s-20260328d`
- queued first follow-on:
  `homegolf-baseline-g64-stepcap2-clip1-lr075-artifactonly-600s-20260328f`
- queued second follow-on:
  `homegolf-baseline-g64-stepcap2-clip1-lr050-artifactonly-600s-20260328g`
- queued third follow-on:
  `homegolf-baseline-g64-stepcap2-clip1-lr035-artifactonly-600s-20260328h`

Retained queue pids:

- `20260328f`: `778424`
- `20260328g`: `780303`
- `20260328h`: `780408`

Retained queue waiting evidence:

- `20260328g` waiting for:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/homegolf-baseline-g64-stepcap2-clip1-lr075-artifactonly-600s-20260328f/train.pid`
- `20260328h` waiting for:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/homegolf-baseline-g64-stepcap2-clip1-lr050-artifactonly-600s-20260328g/train.pid`

## Honest Boundary

This audit does not add a new finished PGOLF score.

Current retained truth:

- `20260328d` is still in inline roundtrip validation
- the retained actual public PGOLF score is still `6.306931747817168`
- the new improvement is queue depth and loop autonomy, not a completed new
  score receipt
- XTRAIN is unchanged in this pass
