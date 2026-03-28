# HOMEGOLF Local Wallclock Projection Refusal Audit

Date: 2026-03-28

## Scope

This audit records the next follow-up after
`docs/audits/2026-03-28-homegolf-initial-validation-step-zero-fix-audit.md`.

That earlier fix removed the accidental `step=0` validation sweep.

The next live `archlinux` rerun then exposed the remaining local truth:

- the exact local competitive HOMEGOLF lane still starts real training
- but the first optimizer step itself does not fit inside the declared
  `600` second wallclock on the current `RTX 4080` posture

This change makes that boundary machine-legible and immediate.

## What Changed

`crates/psionic-train/src/parameter_golf_single_h100_training.rs` now projects
the full optimizer-step wallclock during the first live micro-steps.

If the projected full step exceeds the remaining declared training wallclock,
the trainer now:

- emits one explicit `wallclock_projection_refusal ...` log line
- writes a typed training report with disposition
  `refused_wallclock_projection`
- preserves `executed_steps=0`
- emits no training artifact or fake local score

`crates/psionic-train/src/parameter_golf_single_h100_visualization.rs` now maps
that new disposition into the existing remote-training `refused` class.

This keeps the local HOMEGOLF operator surface honest without pretending the
current single-device exact lane is still a viable 10-minute score loop.

## Live Proof

Machine:

- `archlinux`
- `NVIDIA GeForce RTX 4080`

Committed revision:

- `8dad683d` (`Refuse impossible HOMEGOLF wallclock runs`)

Command posture:

```bash
CARGO_TARGET_DIR=~/code/psionic/target \
PSIONIC_PARAMETER_GOLF_MODEL_VARIANT=competitive_homegolf_v1 \
PSIONIC_PARAMETER_GOLF_HOMEGOLF_GRAD_ACCUM_STEPS=64 \
PSIONIC_PARAMETER_GOLF_DISABLE_SCORE_FIRST_TTT=1 \
PSIONIC_PARAMETER_GOLF_FINAL_MODEL_SURFACE=ema \
cargo run -q -p psionic-train --bin parameter_golf_homegolf_single_cuda_train -- \
  ~/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
  ~/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  /tmp/psionic_homegolf_runs/homegolf_competitive_nottt_ema_projection_refusal_8dad683d_20260328.json \
  roundtrip_only non_overlapping
```

Retained log lines:

- `train_step_start step=1/20000 grad_accum_steps=64`
- `micro_step_complete step=1/20000 micro_step=1/64 ...`
- `wallclock_projection_refusal step=1/20000 observed_micro_steps=1/64 observed_step_wallclock_ms=29584 projected_full_step_wallclock_ms=1893376 remaining_training_wallclock_ms=600000`
- `wrote /tmp/psionic_homegolf_runs/homegolf_competitive_nottt_ema_projection_refusal_8dad683d_20260328.json with disposition RefusedWallclockProjection executed_steps=0 stop_reason=None machine_profile=homegolf_local_cuda`

Retained report facts:

- `disposition=refused_wallclock_projection`
- `executed_steps=0`
- `stop_reason=null`
- refusal subject:
  `homegolf_local_cuda.wallclock_projection`
- refusal detail:
  `HOMEGOLF local-CUDA single-device trainer projected optimizer step 1 to require about 1893376ms after 1/64 micro-steps, which exceeds the remaining 600000ms wallclock budget on this machine posture.`

## Why This Matters

Before this change, the local exact HOMEGOLF lane could still spend many
minutes inside a first optimizer step even when the current machine posture had
already made the `600` second contract impossible.

After this change:

- the same impossible lane now proves its failure boundary after the first
  measured micro-step
- the refusal is typed, retained, and visible in the same report surface as
  successful runs
- later operators do not need to rediscover the same local dead end through
  another half-hour rerun

This is not a score improvement.

It is the correct operational fix for the reachable hardware set.

## Validation

Local validation:

- `rustfmt crates/psionic-train/src/parameter_golf_single_h100_training.rs`
- `rustfmt crates/psionic-train/src/parameter_golf_single_h100_visualization.rs`
- `cargo check -q -p psionic-train --bin parameter_golf_homegolf_single_cuda_train`

Live validation:

- one fresh `archlinux` rerun from clean worktree
- one retained refusal report at:
  `/tmp/psionic_homegolf_runs/homegolf_competitive_nottt_ema_projection_refusal_8dad683d_20260328.json`

## Honest Boundary After This Audit

What is true:

- the local exact HOMEGOLF lane now fails fast and honestly on impossible
  wallclock postures
- the local loop still emits real artifacts and promptable text in bounded
  one-step proof mode
- the retained actual full-validation PGOLF score in-repo is still
  `6.306931747817168`

What is not true:

- this did not produce a new HOMEGOLF full-validation score
- this did not make the single-`4080` exact lane competitive
- this did not create a live dense mixed-device HOMEGOLF operator path

## Improvement Over The Previous HOMEGOLF Audit

Compared with the previous step-zero validation audit, this change improves the
local operator loop in one concrete way:

- the exact local competitive lane no longer reaches real training and then
  burns the machine for an obviously impossible first step before anyone learns
  the current hardware cannot honor the declared `10` minute contract
