# HOMEGOLF Competitive Bigram Input Fix Audit

Date: 2026-03-28

## Summary

This audit records the next HOMEGOLF integration fix after the local
competitive override lane landed.

The first live rerun of the new local competitive score posture on `archlinux`
did compile and start the trainer, but it failed immediately in
`initial_validation` with:

- `missing input tensor t1`

That failure was not a cluster or operator problem.

It was a real local runtime bug in the `competitive_homegolf_v1` lane:

- the competitive model variant enables the optional `BigramHash` graph input
- the device-resident CUDA validation and training sessions only bound
  `input_ids` and `target_ids`
- they never allocated or wrote the optional bigram-token input buffer

So the eval graph asked for one extra input tensor that the session never
supplied.

## Live Failure Reproduction

The failure was reproduced on:

- `archlinux`
- `NVIDIA GeForce RTX 4080`

Launch posture:

```bash
CARGO_TARGET_DIR=~/code/psionic/target \
PSIONIC_PARAMETER_GOLF_MODEL_VARIANT=competitive_homegolf_v1 \
PSIONIC_PARAMETER_GOLF_HOMEGOLF_GRAD_ACCUM_STEPS=64 \
PSIONIC_PARAMETER_GOLF_DISABLE_SCORE_FIRST_TTT=1 \
PSIONIC_PARAMETER_GOLF_FINAL_MODEL_SURFACE=ema \
cargo run -q -p psionic-train --bin parameter_golf_homegolf_single_cuda_train -- \
  ~/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
  ~/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  /tmp/psionic_homegolf_runs/homegolf_competitive_nottt_ema_fullscore_20260328.json \
  roundtrip_only \
  non_overlapping
```

Observed log tail:

- `single_device_train_start ... machine_profile=homegolf_local_cuda ...`
- `initial_validation_start sequences=60568 batch_sequences=64`
- `validation_batch_start stage=initial_validation eval_mode=non_overlapping batch=1/947 ...`
- `missing input tensor t1`

That matters because it proves the local competitive lane had crossed from
operator bring-up into the actual trainer, and the next blocker was the model
surface itself.

## Code Changes

Files changed:

- `crates/psionic-train/src/parameter_golf_single_h100_training.rs`
- `docs/HOMEGOLF_TRACK.md`

Behavioral changes:

- `ParameterGolfCudaValidationSession` now stores the model config plus an
  optional bigram-token buffer and staging buffer
- `ParameterGolfCudaTrainingSession` now does the same
- both session types now materialize the optional BigramHash input and bind it
  whenever the lowered graph declares `bigram_token_ids_tensor_id`
- resident validation now explicitly checks that training and eval graphs agree
  on the optional bigram-input posture

## Validation

The following validations passed locally after the fix:

```bash
cargo check -q -p psionic-train --bin parameter_golf_homegolf_single_cuda_train
cargo test -q -p psionic-train competitive_homegolf_validation_session_binds_bigram_input_tensor -- --nocapture
cargo test -q -p psionic-train competitive_homegolf_training_session_binds_bigram_input_tensor -- --nocapture
```

Those two new tests matter because they pin the exact competitive-lane bug:

- one test covers the device-resident eval session
- one test covers the training session

So the optional BigramHash input is now frozen as a required part of the local
competitive CUDA session contract.

## Improvement Over The Earlier Competitive Override Audit

Compared with the earlier 2026-03-28 override-lane audit, this pass improves
the system in one decisive way:

1. The competitive local HOMEGOLF lane is no longer merely configurable; it is
   also executable through the first real validation batch on the BigramHash
   model family.

That still does not mean a new retained score exists yet.

It does mean the next rerun is blocked by honest model quality/runtime limits
again, not by a missing graph input.
