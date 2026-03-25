# 2026-03-25 Psionic Parameter Golf RunPod Single-H100 First-Run Audit

This audit records the first real RunPod-hosted single-H100 Parameter Golf
baseline run executed by the Psionic-owned Rust trainer path.

## Conclusion

`#458` is now closeable.

The bounded single-H100 Rust trainer path is real on remote H100 hardware. It
completed one full bounded RunPod run, exported the final `int8+zlib` artifact,
and produced final contest-facing roundtrip metrics from the Rust path itself.

The lane is still nowhere near challenge-speed or leaderboard closure. This
audit proves remote execution truth. It does not prove competitiveness.

## Committed Evidence

- [parameter_golf_runpod_single_h100_first_real_audit.json](/tmp/psionic-open-issues.PzsotD/fixtures/parameter_golf/reports/parameter_golf_runpod_single_h100_first_real_audit.json)
- [parameter_golf_runpod_single_h100_first_real_training_report.json](/tmp/psionic-open-issues.PzsotD/fixtures/parameter_golf/reports/parameter_golf_runpod_single_h100_first_real_training_report.json)

The retained remote run root was:

- `/workspace/issue454_single_h100_run_20260324T215106Z`

The retained RunPod identity was:

- pod id `2sul6l37mn5dwp`
- pod host `216.243.220.215`
- SSH port `12708`
- SSH user `root`
- remote hostname `f073d6941c85`

The admitted accelerator surface was:

- `NVIDIA H100 80GB HBM3`
- `memory_capacity_bytes=85520809984`
- machine contract satisfied: `true`

## Outcome

The run used the bounded proof posture:

- `max_steps=1`
- `grad_accum_steps=8`
- `final_validation_mode=roundtrip_only`
- `validation_eval_mode=non_overlapping`

The final retained metrics were:

- executed optimizer steps: `1`
- train step mean microbatch loss: `8.28843117`
- observed training time: `1,348,854 ms`
- final roundtrip `val_loss`: `10.64899007`
- final roundtrip `val_bpb`: `6.30693175`
- final roundtrip eval time: `43,306,994 ms`
- total wallclock: `44,657,278 ms`
- compressed model bytes: `4,732,744`
- compressed artifact digest:
  `4657d793ae3e64796670b6768f433c48f788518725ba2854e913db205412b250`

The finalizer also sealed the app-facing visualization mirror under the remote
run root:

- `training_visualization/parameter_golf_single_h100_remote_training_visualization_bundle_v1.json`
- `training_visualization/remote_training_run_index_v1.json`

## What This Proves

- the Psionic Rust single-H100 trainer can execute end to end on real remote
  H100 hardware outside the local review host
- the lane can preserve machine admission, accelerator evidence, training logs,
  final artifact identity, final roundtrip metrics, and a final machine-readable
  audit bundle
- the canonical RunPod single-H100 finalizer is good enough to seal a finished
  run into one stable audit surface

## What It Does Not Prove

- challenge-speed closure
- `train_gpt.py` parity closure
- leaderboard-grade score quality
- `8xH100` distributed readiness
- record-track readiness

The dominant current gap is runtime. One bounded train step took about
`22.5 minutes`. The final roundtrip validation took about `12.0 hours`. That is
enough to prove the remote path is real. It is not enough to claim the lane is
operationally competitive.

## Implication For The Remaining PGOLF Stack

This audit retires the old “no real remote single-H100 Psionic proof exists”
blocker.

It does not retire the remaining parity and speed blockers tracked by the later
PGOLF issues:

- `#466` overall `train_gpt.py` parity closure
- `#541` scoreboard-grade validation closure
- `#543` real multi-step `8xH100` challenge loop
- `#545` persistent distributed worker mesh
- `#546` full-sequence attention kernels
- `#547` activation-retention reduction
- `#548` decoupled eval geometry
- `#549` device-resident train-path model reuse

The honest new claim boundary is simple:

Psionic now has one real bounded RunPod single-H100 Parameter Golf baseline
success. It still does not have a fast or score-competitive one.
