# HOMEGOLF Repo-Owned Queue Runner Audit

Date: 2026-03-28

## Scope

This audit records the next operator improvement after the LR-scale queued
follow-on was first staged.

Before this change, the queued `20260328e` run existed only behind one ad hoc
remote scratch script.

That worked, but it was not the right long-running operator shape.

## What Changed

Added:

- `scripts/queue-parameter-golf-homegolf-local-cuda.sh`

The queue runner:

- waits on one existing HOMEGOLF output root
- reads that root's `train.pid`
- polls until the trainer exits
- then launches the normal repo-owned local CUDA runner
- defaults to `--sync-main` for the queued launch

This keeps queued HOMEGOLF follow-ons inside the repository's normal operator
surface instead of leaving them in untracked scratch shell files.

## Active Use

The queued next run on `archlinux` was restaged through the committed queue
runner:

- current active run:
  `homegolf-baseline-g64-stepcap1-clip1-600s-20260328d`
- queued next run:
  `homegolf-baseline-g64-stepcap2-clip1-lr075-600s-20260328e`
- queue pid:
  `775656`
- queue log:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/queue_homegolf_20260328e_repo.log`

Retained queue log head:

- `queue_wait_root=/home/christopherdavid/scratch/psionic_homegolf_runs/homegolf-baseline-g64-stepcap1-clip1-600s-20260328d`
- `queue_poll_seconds=60`
- `queue_wait_pid=768560`
- `queue_waiting timestamp=2026-03-28T06:25:29-05:00 pid=768560`

## Why This Improves The System

Compared with the prior ad hoc queue:

1. The queued launch path is now committed and reusable.
2. The queue behavior stays aligned with the clean-main runner contract.
3. The next long-running HOMEGOLF iteration is no longer hidden in scratch-only
   shell glue.

That matters because the current local score loop has multi-hour full score
closeout times.

Once that is true, queueing is no longer incidental operator glue.

It becomes part of the actual retained execution path.

## Honest Boundary After This Audit

What is true:

- the next queued HOMEGOLF run now waits behind a repo-owned queue runner
- the current honest clipped run is still alive in full roundtrip validation
- the queued follow-on still preserves the same honest local `600s` contract

What is not true:

- this audit does not contain a newly completed PGOLF score
- the retained actual in-repo PGOLF score is still `6.306931747817168`
