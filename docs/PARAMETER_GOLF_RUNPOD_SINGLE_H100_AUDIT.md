# Parameter Golf RunPod Single-H100 Audit

> Status: canonical RunPod single-H100 audit-finalizer contract for the
> Parameter Golf baseline lane, written on 2026-03-23 after landing
> `scripts/parameter-golf-runpod-finalize-single-h100.sh`.

This document records the current audit surface for the first real RunPod
single-H100 Parameter Golf baseline run.

The finalizer contract is still narrower than contest readiness, but it is no
longer only a pre-closeout shell. The first real bounded RunPod audit now
exists and is committed in this repo.

## Canonical Artifact

- single-H100 audit finalizer:
  `scripts/parameter-golf-runpod-finalize-single-h100.sh`
- first real retained audit:
  `fixtures/parameter_golf/reports/parameter_golf_runpod_single_h100_first_real_audit.json`
- first real retained trainer report:
  `fixtures/parameter_golf/reports/parameter_golf_runpod_single_h100_first_real_training_report.json`
- first real run outcome audit:
  `docs/audits/2026-03-25-psionic-parameter-golf-runpod-single-h100-first-run-audit.md`

## What The Finalizer Preserves

The finalizer writes one machine-readable audit JSON that binds:

- the exact RunPod run root
- pod identity such as provider pod id, host, SSH port, and hostname
- retained `nvidia-smi` inventory and topology captures when available
- the trainer JSON report when present
- the trainer log, its SHA-256, and the latest retained micro-step progress
- the provider-neutral visualization bundle and run index under
  `training_visualization/`
- one explicit audit outcome:
  - `succeeded`
  - `refused`
  - `in_progress`
  - `failed_training_exit_nonzero`
  - `failed_missing_training_report`
  - `failed_incomplete_training_report`

That gives `#458` one deterministic closeout surface whether the run succeeds,
refuses, is still executing, or dies before the final report lands.

The same finalizer now also materializes:

- `training_visualization/parameter_golf_single_h100_remote_training_visualization_bundle_v1.json`
- `training_visualization/remote_training_run_index_v1.json`

It does that by calling the Rust-owned visualization materializer against the
trainer report, the existing live bundle, or the retained trainer log.

That keeps the RunPod lane on the same typed app contract as the Google lane.

## Command

Run the finalizer on the RunPod host after or during the bounded trainer run:

```bash
bash scripts/parameter-golf-runpod-finalize-single-h100.sh \
  --run-root /workspace/<run-id> \
  --output /workspace/<run-id>/parameter_golf_runpod_single_h100_audit.json \
  --pod-host <runpod-host> \
  --pod-port <ssh-port> \
  --trainer-pid <trainer-pid>
```

Optional fields let the operator preserve a provider id, SSH user, non-default
report or log paths, an explicit trainer exit code, or a typed failure detail.

## Honest Boundary

This audit surface does not claim:

- challenge-speed closure
- `8xH100` readiness
- record-track readiness

It closes one narrower but important thing:

- the repo now has one deterministic RunPod single-H100 audit finalizer and one
  committed real bounded success audit proving that the Rust single-H100 lane
  can execute end to end on remote H100 hardware, preserve final metrics plus
  compressed-model bytes, and seal the result into one machine-readable audit
  bundle without inventing a provider-specific reporting path at closeout
