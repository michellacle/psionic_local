# HOMEGOLF Initial Validation Step-Zero Fix Audit

Date: 2026-03-28

## Summary

This audit records the next HOMEGOLF integration fix after the competitive
BigramHash input bug was fixed.

The next live rerun on `archlinux` did not crash on `missing input tensor t1`.
It reached the competitive CUDA trainer and then immediately spent its runtime
budget inside a full validation sweep before any optimizer step ran.

Observed retained log tail:

- `single_device_train_start ... max_wallclock_seconds=600`
- `initial_validation_start sequences=60568 batch_sequences=64`
- `validation_batch_start stage=initial_validation ... batch=1/947 ...`
- `validation_progress stage=initial_validation ... batch=1/947 units=64 tokens=65536 elapsed_ms=58979`

That was not an honest way to spend a local `10` minute training loop.

It meant the current HOMEGOLF local runner was still blocked by a control-loop
bug, not by model quality.

## Root Cause

The main training loop treated periodic validation as:

- run when `validation_loss_every > 0`
- and `step % validation_loss_every == 0`

That includes `step == 0`.

So any local HOMEGOLF posture with periodic validation enabled launched a full
live validation pass before the first training step.

On the `RTX 4080` local full-score lane, that meant:

- `947` full validation batches
- about `58.9s` just for batch `1/947`
- zero actual training progress before the score loop had already burned a
  large part of the wallclock budget

## Code Changes

Files changed:

- `crates/psionic-train/src/parameter_golf_single_h100_training.rs`
- `docs/HOMEGOLF_TRACK.md`

Behavioral changes:

- added `should_run_live_validation_checkpoint(...)` to centralize the
  validation-checkpoint rule
- periodic validation now starts only when `step > 0` and the configured
  interval is reached
- final raw-surface validation behavior remains unchanged
- added a unit test that pins the intended control-loop contract:
  - no periodic validation at `step=0`
  - periodic validation at later configured checkpoints
  - final validation still runs for raw-surface closeout
  - final validation stays disabled for non-raw final-model surfaces

## Validation

The following validations passed locally after the fix:

```bash
cargo check -q -p psionic-train --bin parameter_golf_homegolf_single_cuda_train
cargo test -q -p psionic-train live_validation_checkpoints_skip_step_zero_but_keep_later_and_final_validation -- --nocapture
```

## Improvement Over The Earlier Bigram Fix Audit

Compared with the earlier 2026-03-28 BigramHash input fix audit, this pass
improves the local HOMEGOLF lane in the next way that matters:

1. The competitive local run is no longer structurally doomed to spend its
   budget on pre-training validation.
2. The first `600` seconds are now available for actual optimizer steps.
3. The remaining bottleneck is again honest score closeout and model quality,
   not a control-loop bug.

This still does not create a new retained PGOLF score by itself.

It does remove the next false floor from the local competitive iteration loop.
