# 2026-03-26 Psionic Parameter Golf Single-H100 Scratch-Reuse Step Proof

This audit records the first fresh exact public-shape same-node H100 reruns
after the CUDA submission-scoped attention scratch-reuse fix in
`crates/psionic-backend-cuda/src/lib.rs`.

## Conclusion

Current `main` no longer OOMs on the exact public single-H100 train shape.

Fresh real RunPod `NVIDIA H100 80GB HBM3` reruns now complete one exact-shape
train step and emit the live train-step receipt lines again:

- live-validation root:
  `/workspace/parameter-golf-runpod/single-h100-proof-20260326T030500Z-scratch-reuse`
- roundtrip-only root:
  `/workspace/parameter-golf-runpod/single-h100-proof-20260326T032500Z-scratch-reuse-roundtrip`

The exact-shape same-node H100 blocker from
`docs/audits/2026-03-26-psionic-parameter-golf-single-h100-current-main-oom-audit.md`
is resolved.

The parent scorepath issue stays open for a narrower reason: the train step is
still dominated by CUDA forward and backward time, not by host materialization,
and it is still far from competitive contest runtime.

## Run Identity

- pod device: `NVIDIA H100 80GB HBM3`
- clean source tree on pod:
  `/root/psionic-h100-proof-main`
- build target dir:
  `/root/psionic-target-proof`
- exact trainer command shape:

```bash
/root/psionic-target-proof/release/parameter_golf_single_h100_train \
  /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
  /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  <run-root>/parameter_golf_single_h100_training.json \
  1 \
  <final-validation-mode> \
  non_overlapping
```

## Observed Same-Node H100 Outputs

Warm-cache live-validation rerun:

```text
single_h100_train_start run_id=parameter-golf-single-h100-trainer device=NVIDIA H100 80GB HBM3 max_steps=1 iterations=20000 warmup_steps=0 grad_accum_steps=8 val_loss_every=0 train_log_every=1 final_validation_mode=live_only validation_eval_mode=non_overlapping validation_batch_sequences=64 local_train_sequences=64 local_validation_sequences=64 max_wallclock_seconds=0
train_step_start step=1/1 grad_accum_steps=8
micro_step_complete step=1/1 micro_step=8/8 window_id=42845a4917f7d6f04439320a0c3f0c34ca83982ff80685ae3eea363b67381c13 train_loss=8.29110241 forward_ms=358831 backward_ms=118194 host_materialization_ms=610 retained_binding_f32=36642059168 gradient_f32=136479296
train_step_complete step=1 mean_microbatch_loss=8.28840256 lr_mult=1.00000000 muon_momentum=0.85000002 host_materialization_ms=610 optimizer_step_ms=3200
train_runtime_receipt step=1 path=device_resident_cuda_training_graph_v1 graph_surface=parameter_golf_baseline_training_graph_v2 sessions=1 stable_parameter_buffers=42 stable_parameter_values=17059912 resident_parameter_upload_us=34627 parameter_refresh_us=0 input_token_write_us=1650 target_token_write_us=1203 resident_buffers_reused=true
step:1/1 train_loss:8.2884 train_time:482626ms step_avg:482626.00ms
```

Conservative roundtrip-only rerun:

```text
single_h100_train_start run_id=parameter-golf-single-h100-trainer device=NVIDIA H100 80GB HBM3 max_steps=1 iterations=20000 warmup_steps=0 grad_accum_steps=8 val_loss_every=0 train_log_every=1 final_validation_mode=roundtrip_only validation_eval_mode=non_overlapping validation_batch_sequences=64 local_train_sequences=64 local_validation_sequences=64 max_wallclock_seconds=0
train_step_start step=1/1 grad_accum_steps=8
micro_step_complete step=1/1 micro_step=8/8 window_id=42845a4917f7d6f04439320a0c3f0c34ca83982ff80685ae3eea363b67381c13 train_loss=8.29108715 forward_ms=583203 backward_ms=123654 host_materialization_ms=612 retained_binding_f32=36642059168 gradient_f32=136479296
train_step_complete step=1 mean_microbatch_loss=8.28840160 lr_mult=1.00000000 muon_momentum=0.85000002 host_materialization_ms=612 optimizer_step_ms=3999
train_runtime_receipt step=1 path=device_resident_cuda_training_graph_v1 graph_surface=parameter_golf_baseline_training_graph_v2 sessions=1 stable_parameter_buffers=42 stable_parameter_values=17059912 resident_parameter_upload_us=97923 parameter_refresh_us=0 input_token_write_us=1423 target_token_write_us=1154 resident_buffers_reused=true
step:1/1 train_loss:8.2884 train_time:713464ms step_avg:713464.00ms
final_validation_skipped mode=roundtrip_only reason=explicit_final_validation_mode
```

No completed JSON report was retained from these fresh reruns because the
follow-on validation passes were intentionally interrupted after the train-step
proof:

- the live-validation sweep was still at `batch=3/947` after about `91.9s`
- the `roundtrip_only` int8-zlib sweep was still at `batch=2/947` after about
  `72.7s`

Those long validation sweeps are real same-node truths, but they are not the
fast proof surface for the train-step issues below.

## Exact Same-Node Comparison Against The Committed Baseline

Committed exact-shape same-node H100 baseline:

- report:
  `fixtures/parameter_golf/reports/parameter_golf_runpod_single_h100_first_real_training_report.json`
- audit:
  `docs/audits/2026-03-25-psionic-parameter-golf-runpod-single-h100-first-run-audit.md`

Committed baseline receipt facts:

- `train_time_ms=1348854`
- `forward_loss_cuda_ms=625710`
- `backward_cuda_ms=700639`
- `host_gradient_materialization_ms=3321`
- `retained_binding_f32_count=59600540576`

Fresh conservative current-main proof facts:

- `train_time_ms=713464`
- `forward_ms=583203`
- `backward_ms=123654`
- `host_materialization_ms=612`
- `retained_binding_f32=36642059168`

That means the exact public-shape same-node H100 lane moved materially:

- step wallclock: `1348854 -> 713464 ms` down `47.1%`
- forward wallclock: `625710 -> 583203 ms` down `6.8%`
- backward wallclock: `700639 -> 123654 ms` down `82.4%`
- retained binding volume: `59600540576 -> 36642059168` down `38.5%`
- host materialization: `3321 -> 612 ms` down `81.6%`

## Issue Impact

This proof resolves the old same-node-H100-does-not-fit blocker on:

- `#547`
- `#562`
- `#563`

It does not close the parent hot-path issue:

- `#546`

Even on the fresh exact-shape H100 reruns, the step is still dominated by
forward and backward CUDA time rather than host materialization, and the full
train step is still hundreds of seconds instead of contest-speed milliseconds.

This proof also does not isolate a clean banked-execution-only wallclock delta
for:

- `#558`

The current same-node receipts include direct banked execution on current
`main`, but they also include the newer attention and retained-surface cuts, so
the exact step-time reduction is not attributable to banked execution alone.
