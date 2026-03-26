# 2026-03-26 Psionic Parameter Golf Current-Main Single-H100 OOM Audit

This audit records the first fresh current-`main` same-node H100 rerun after
the retained-surface and CUDA scorepath slices that landed through `14fc612b`.
It is now historical context. The later submission-scoped CUDA attention
scratch-reuse fix let the exact public-shape same-node H100 train step complete
again; see
`docs/audits/2026-03-26-psionic-parameter-golf-single-h100-scratch-reuse-step-proof.md`.

## Conclusion

At the time of this rerun, current `main` was still blocked from fresh
single-H100 proof receipts on the exact public PGOLF train shape.

The release binary builds and launches on a real RunPod `NVIDIA H100 80GB HBM3`
node. It then reaches:

- `single_h100_train_start ... grad_accum_steps=8 ... local_train_sequences=64 ... local_validation_sequences=64`
- `train_step_start step=1/1 grad_accum_steps=8`

and then fails with:

- `cudaMalloc failed: out of memory`

That meant Psionic still did not have a fresh current-`main` same-node H100
receipt for the exact public train shape. The later scratch-reuse fix removed
this blocker.

## Run Identity

- pod device: `NVIDIA H100 80GB HBM3`
- repo SHA: `14fc612b`
- clean source tree on pod: `/workspace/psionic-h100-proof`
- build target dir: `/root/psionic-target`
- train-step proof root:
  `/workspace/parameter-golf-runpod/single-h100-proof-20260326T021405Z-14fc612b`
- retained rerun root that reproduced the same failure after the build cache was
  warm:
  `/workspace/parameter-golf-runpod/single-h100-proof-20260326T022034Z-14fc612b-oom`

Exact bounded proof command:

```bash
/root/psionic-target/release/parameter_golf_single_h100_train \
  /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
  /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  /workspace/parameter-golf-runpod/single-h100-proof-20260326T022034Z-14fc612b-oom/parameter_golf_single_h100_training.json \
  1 \
  live_only \
  non_overlapping
```

## Observed Runtime Output

The rerun printed:

```text
single_h100_train_start run_id=parameter-golf-single-h100-trainer device=NVIDIA H100 80GB HBM3 max_steps=1 iterations=20000 warmup_steps=0 grad_accum_steps=8 val_loss_every=0 train_log_every=1 final_validation_mode=live_only validation_eval_mode=non_overlapping validation_batch_sequences=64 local_train_sequences=64 local_validation_sequences=64 max_wallclock_seconds=0
train_step_start step=1/1 grad_accum_steps=8
cudaMalloc failed: out of memory
```

No completed `parameter_golf_single_h100_training.json` receipt was emitted for
either fresh current-`main` run root.

## Historical Impact

This exact-shape same-node rerun does not close:

- `#546`
- `#547`
- `#558`
- `#562`
- `#563`

At this point in time, those issues still required fresh same-node H100 or
`8xH100` receipts because the public-shape trainer could not finish the first
train step on an `80GB` H100.

## What This Means

The repo now had two separate same-node H100 truths:

- the older committed same-node parity report proves the comparison harness
  exists
- this fresh current-`main` rerun proves the latest scorepath branch still does
  not fit the exact public single-H100 train shape

The next honest same-node action was a memory-surface fix that let current
`main` complete one exact-shape H100 train step again. That later fix is
captured in the follow-on scratch-reuse audit linked above.
