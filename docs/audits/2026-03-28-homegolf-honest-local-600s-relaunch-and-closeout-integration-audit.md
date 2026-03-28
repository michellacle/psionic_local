# HOMEGOLF Honest Local 600s Relaunch And Closeout Integration Audit

Date: 2026-03-28

## Scope

This audit records the next correction after the current-main local score launch
and the first prompt-closeout watcher.

Two things needed to become explicit:

1. the prior `20260328a` HOMEGOLF run was only a bounded probe
2. the closeout watcher still depended on one hand-built env wrapper to stay
   off `/tmp`

## What Changed

Updated:

- `scripts/run-parameter-golf-homegolf-local-cuda.sh`
- `scripts/wait-parameter-golf-homegolf-prompt-closeout.sh`

The local CUDA runner now supports:

- `--attach-prompt-closeout`
- `--closeout-prompt`
- `--closeout-max-new-tokens`
- `--closeout-poll-seconds`
- `--closeout-timeout-seconds`
- `--closeout-binary-path`

The prompt closeout watcher now supports:

- `--tmpdir`
- `--cargo-target-dir`

That means the repo-owned launch path can now:

- start one clean-main HOMEGOLF run
- attach one prompt closeout watcher at launch time
- keep both trainer and prompt closeout on scratch-backed temp and cargo paths
- persist the trainer pid and prompt watcher pid under the same output root

## Honest Local Relaunch

The first current-main launch audit kept one incorrect implication alive:

- `homegolf-baseline-g64-2step-roundtrip-20260328a` looked like the live local
  score loop

But that run passed `--max-steps 2`.

That matters because the trainer then ran in the bounded posture rather than the
real local wallclock posture.

The corrected live run is:

- host: `archlinux`
- repo revision at launch: `60c97bf0`
- run id: `homegolf-baseline-g64-roundtrip-600s-20260328b`
- output root:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/homegolf-baseline-g64-roundtrip-600s-20260328b`

Observed retained log contract:

- `machine_profile=homegolf_local_cuda`
- `warmup_steps=0`
- `grad_accum_steps=64`
- `final_validation_mode=roundtrip_only`
- `validation_eval_mode=non_overlapping`
- `max_wallclock_seconds=600`

Observed live training movement from the retained log:

- `train_step_start step=1/20000`
- `micro_step=1/64 train_loss=8.35887241`
- `micro_step=2/64 train_loss=8.34251022`
- `micro_step=3/64 train_loss=8.30022907`
- `micro_step=4/64 train_loss=8.27811050`
- `micro_step=9/64 train_loss=8.33421516`

At the time of this audit:

- `training_report.json` was still absent
- `closeout_summary.json` was still absent
- the trainer and watcher were both still live

## Why This Improves The System

Compared with the previous iteration:

- the honest-score boundary is cleaner because the bounded `2`-step probe is no
  longer conflated with the real local `600s` run
- the prompt closeout path is stronger because it no longer depends on one
  ad-hoc remote env wrapper to avoid `/tmp`
- future HOMEGOLF operator launches can stay fully repo-owned from train start
  through prompt closeout

This does not improve model quality directly.

It improves the integrity of the iteration loop around quality receipts.

## Honest Boundary After This Audit

What is true:

- the current-main local HOMEGOLF score loop is now relaunched in the honest
  `600s` posture
- the repo-owned runner can attach the prompt closeout watcher directly
- both trainer and prompt closeout now have explicit scratch-backed execution
  controls

What is not true:

- this audit does not contain a new completed PGOLF score
- this audit does not show the final prompt output from `20260328b` yet
- the retained actual in-repo PGOLF score is still `6.306931747817168`
