# HOMEGOLF Learning Rate Scale And Queued Follow-On Audit

Date: 2026-03-28

## Scope

This audit records the next operator iteration after the grad-clip closeout
unlock.

The grad-clip pass fixed the immediate structural blocker:

- the honest capped local lane can now export a retained artifact and generate
  actual text

That moves the next tuning question forward:

- can the local lane keep more than one optimizer step stable if the learning
  rates are reduced while the honest `600s` contract stays intact?

## What Changed

Updated:

- `crates/psionic-train/src/bin/parameter_golf_homegolf_single_cuda_train.rs`
- `scripts/run-parameter-golf-homegolf-local-cuda.sh`

The local HOMEGOLF lane now has one uniform LR scale override:

- env:
  `PSIONIC_PARAMETER_GOLF_HOMEGOLF_LR_SCALE=<f32>`
- runner flag:
  `--learning-rate-scale <f32>`

The override scales:

- `embed_lr`
- `head_lr`
- `tied_embed_lr`
- `matrix_lr`
- `scalar_lr`

It does not alter:

- the `600s` wallclock contract
- the challenge profile selection
- the grad-accum setting
- the closeout path

## Why This Improves The System

Compared with the prior grad-clip-only surface:

1. The operator can now pursue a real quality follow-on without another code
   change.
2. The next quality experiment can stay on the honest local HOMEGOLF lane.
3. The next run can be queued immediately behind the still-active current score
   closeout instead of waiting for manual restart.

This matters because the current system state is now split cleanly:

- grad clip fixed export and promptability
- LR control is the next obvious quality knob for the second-step stability
  problem

## Queued Follow-On

The current active run is still:

- `homegolf-baseline-g64-stepcap1-clip1-600s-20260328d`

At the time of this audit it remained alive inside the full roundtrip score
closeout and had progressed to:

- `validation_batch_start ... batch=18/947`
- `elapsed_ms=424969`

The next run is already queued on `archlinux` to start automatically when that
current run exits:

- queued run id:
  `homegolf-baseline-g64-stepcap2-clip1-lr075-600s-20260328e`
- queue script:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/queue_homegolf_20260328e.sh`
- queue log:
  `/home/christopherdavid/scratch/psionic_homegolf_runs/queue_homegolf_20260328e.log`
- queue pid:
  `773908`

Queued run settings:

- `grad_accum_steps=64`
- `challenge_max_steps=2`
- `grad_clip_norm=1.0`
- `learning_rate_scale=0.75`
- `final_validation_mode=roundtrip_only`
- `validation_eval_mode=non_overlapping`

## Honest Boundary After This Audit

What is true:

- the local HOMEGOLF lane now has a repo-owned LR scale control
- the next honest quality iteration is queued automatically on the remote CUDA
  node
- the current retained clipped artifact still generates actual text

What is not true:

- this audit does not contain the queued `20260328e` result yet
- this audit does not add a new completed PGOLF score yet
- the retained actual in-repo PGOLF score is still `6.306931747817168`
