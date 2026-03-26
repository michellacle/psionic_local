# 2026-03-26 Psionic Parameter Golf Single-H100 Zero-Copy Bank-Slice PTY Proof

This audit records the corrected exact public-shape same-node H100 rerun after
`df70fcea` landed the zero-copy direct-banked CUDA bank-slice path in
`crates/psionic-backend-cuda/src/lib.rs`.

It supersedes the earlier non-PTY capture failure recorded in
`docs/audits/2026-03-26-psionic-parameter-golf-single-h100-zero-copy-bank-slice-blocker-audit.md`.

## Conclusion

The zero-copy direct-banked bank-slice slice does have a valid same-node H100
train-step receipt.

The earlier "reportless CPU-hot blocker" conclusion was a proof-capture
artifact from a non-PTY redirected-file run, not a valid train-step result. A
fresh PTY-backed exact public-shape rerun on the same H100 and same binary
reached `train_runtime_receipt` cleanly.

The runtime result is effectively flat against the retained direct-banked
baseline:

- retained baseline:
  `478.511s`
- fresh PTY-backed zero-copy rerun:
  `480.234s`

That is about `0.36%` slower. This slice does not close `#546`, but it also
does not justify the earlier blocker posture.

## Run Identity

- pod device: `NVIDIA H100 80GB HBM3`
- proof binary:
  `/root/psionic-target-proof-df70fcea/release/parameter_golf_single_h100_train`
- live run root:
  `/workspace/parameter-golf-runpod/single-h100-proof-20260326T090409Z-zero-copy-bank-slices-pty`
- runtime posture:
  `PSIONIC_PARAMETER_GOLF_MATRIX_EXECUTION_MODE=direct_banked`
- exact trainer command shape:

```bash
PSIONIC_PARAMETER_GOLF_MATRIX_EXECUTION_MODE=direct_banked \
  /root/psionic-target-proof-df70fcea/release/parameter_golf_single_h100_train \
  /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
  /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  /workspace/parameter-golf-runpod/single-h100-proof-20260326T090409Z-zero-copy-bank-slices-pty/parameter_golf_single_h100_training.json \
  1 \
  live_only \
  non_overlapping
```

The rerun was launched through `script -qefc` so the trainer wrote line-buffered
stdout to a PTY-backed transcript instead of a redirected file.

## Observed Same-Node H100 Output

Fresh PTY-backed output:

```text
single_h100_train_start run_id=parameter-golf-single-h100-trainer device=NVIDIA H100 80GB HBM3 max_steps=1 iterations=20000 warmup_steps=0 grad_accum_steps=8 val_loss_every=0 train_log_every=1 final_validation_mode=live_only validation_eval_mode=non_overlapping validation_batch_sequences=64 matrix_execution_mode=direct_banked local_train_sequences=64 local_validation_sequences=64 max_wallclock_seconds=0
train_step_start step=1/1 grad_accum_steps=8
micro_step_complete step=1/1 micro_step=8/8 window_id=42845a4917f7d6f04439320a0c3f0c34ca83982ff80685ae3eea363b67381c13 train_loss=8.29109859 forward_ms=359401 backward_ms=115218 host_materialization_ms=610 retained_binding_f32=36642059168 gradient_f32=136479296
train_step_complete step=1 mean_microbatch_loss=8.28839779 lr_mult=1.00000000 muon_momentum=0.85000002 host_materialization_ms=610 optimizer_step_ms=3237
train_runtime_receipt step=1 path=device_resident_cuda_training_graph_v1 graph_surface=parameter_golf_baseline_training_graph_v2 matrix_execution_mode=direct_banked sessions=1 stable_parameter_buffers=42 stable_parameter_values=17059912 resident_parameter_upload_us=31263 parameter_refresh_us=0 input_token_write_us=1761 target_token_write_us=1224 resident_buffers_reused=true
step:1/1 train_loss:8.2884 train_time:480234ms step_avg:480234.00ms
final_validation_start sequences=60568 batch_sequences=64
validation_batch_start stage=final_validation eval_mode=non_overlapping batch=1/947 batch_sequences=64 evaluated_tokens=0 elapsed_ms=0
```

The proof wrapper then interrupted the validation sweep after the train-step
receipt was captured.

## Comparison Against The Retained Direct-Banked Baseline

Retained same-node direct-banked baseline from
`docs/audits/2026-03-26-psionic-parameter-golf-single-h100-banked-vs-split-audit.md`:

- `train_time_seconds=478.511`

Fresh PTY-backed zero-copy rerun:

- `train_time_seconds=480.234`
- `forward_ms=359401`
- `backward_ms=115218`
- `host_materialization_ms=610`
- `retained_binding_f32=36642059168`
- `resident_parameter_upload_us=31263`

Relative delta:

- step wallclock: `478.511s -> 480.234s` up about `0.36%`

The fresh same-node numbers are within noise of the retained direct-banked
baseline. The zero-copy bank-slice slice did not produce a meaningful hot-path
improvement on the exact public H100 train-step shape.

## Capture Notes

The earlier non-PTY run in
`docs/audits/2026-03-26-psionic-parameter-golf-single-h100-zero-copy-bank-slice-blocker-audit.md`
failed to surface a usable receipt because the proof path redirected stdout to a
plain file instead of running the trainer on a PTY-backed console transcript.

That earlier artifact should not be treated as evidence of a new train-path
regression. The corrected PTY-backed rerun above is the real same-node H100
truth for this slice.

## Issue Impact

This proof does not close any open issue.

It narrows `#546`:

- the zero-copy bank-slice slice is not a same-node H100 blocker
- it is also not a real wallclock improvement
- the remaining work stays on the broader exact-shape CUDA forward and backward
  hot path
