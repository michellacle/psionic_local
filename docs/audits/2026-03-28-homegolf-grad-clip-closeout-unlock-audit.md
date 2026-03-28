# HOMEGOLF Grad Clip Closeout Unlock Audit

Date: 2026-03-28

## Scope

This audit records the next local HOMEGOLF iteration after the honest
challenge-step-cap pass.

The open question from that pass was narrow:

- can the honest one-step local `600s` lane survive artifact closeout?

Before this change, the answer was no.

The capped honest run `20260328c` completed its one good optimizer step, then
panicked during artifact export with:

- `min = NaN, max = NaN`

## What Changed

Updated:

- `crates/psionic-train/src/bin/parameter_golf_homegolf_single_cuda_train.rs`
- `scripts/run-parameter-golf-homegolf-local-cuda.sh`

The local HOMEGOLF lane now exposes one repo-owned grad-clip override:

- env:
  `PSIONIC_PARAMETER_GOLF_HOMEGOLF_GRAD_CLIP_NORM=<f32>`
- runner flag:
  `--grad-clip-norm <f32>`

The first retained use of that override is:

- host: `archlinux`
- repo revision at launch: `68e46b82`
- run id: `homegolf-baseline-g64-stepcap1-clip1-600s-20260328d`
- effective grad clip:
  `1.0`

## Retained Live Evidence

The clipped honest run kept the same local score posture:

- `machine_profile=homegolf_local_cuda`
- `max_steps=1`
- `warmup_steps=0`
- `grad_accum_steps=64`
- `final_validation_mode=roundtrip_only`
- `validation_eval_mode=non_overlapping`
- `max_wallclock_seconds=600`

Retained step result:

- `train_step_complete step=1 mean_microbatch_loss=8.29197598`
- `optimizer_step_ms=3927`
- `step:1/1 train_loss:8.2920 train_time:274570ms`

Retained closeout result:

- `final_artifact_export_complete`
- `final_artifact_persist_complete`

Retained artifact path:

- `/home/christopherdavid/scratch/psionic_homegolf_runs/homegolf-baseline-g64-stepcap1-clip1-600s-20260328d/training_report.final_model.st`

Retained artifact size:

- `4132303` bytes

The run then advanced into the honest final roundtrip validation sweep:

- `validation_batch_start stage=int8_zlib_roundtrip ... batch=10/947`
- `evaluated_tokens=589824`
- `elapsed_ms=225066`

So the clipped lane is still live.

It is no longer blocked on export.

It is now blocked on the time required to finish the full roundtrip score
closeout.

## Fresh Actual Text Proof

The new retained artifact was also prompted directly through the
artifact-only HOMEGOLF prompt entrypoint.

Local prompt artifacts:

- artifact:
  `/tmp/psionic_homegolf_prompt_20260328d/training_report.final_model.st`
- tokenizer:
  `/tmp/psionic_homegolf_prompt_20260328d/fineweb_1024_bpe.model`
- vocab:
  `/tmp/psionic_homegolf_prompt_20260328d/fineweb_1024_bpe.vocab`
- prompt report:
  `/tmp/psionic_homegolf_prompt_20260328d/prompt_report.json`

Prompt:

- `the meaning of life is`

Generated text:

- `do do hased do do do do do do do do do do do do do has has hased hased hasedededededededed`

## Why This Improves The System

Compared with the prior honest capped run:

1. The local one-step lane no longer dies in quantized artifact export.
2. The honest one-step training time improved from `304562ms` to `274570ms`.
3. The same retained artifact can now already prove direct text generation,
   even before the long full-validation score closeout finishes.

That is a real system improvement because the prior blocker was structural:

- no artifact
- no prompt on the new artifact
- no path into the honest full roundtrip score sweep

Now all three surfaces are reachable.

## Honest Boundary After This Audit

What is true:

- the local honest capped HOMEGOLF lane now exports a retained artifact cleanly
- the retained clipped artifact generates actual text
- the honest full roundtrip score closeout is actively running on the same
  retained run

What is not true:

- this audit does not contain a completed new PGOLF score yet
- the retained actual in-repo PGOLF score is still `6.306931747817168`
- the local `RTX 4080` lane is still not public `8xH100` leaderboard-equivalent
