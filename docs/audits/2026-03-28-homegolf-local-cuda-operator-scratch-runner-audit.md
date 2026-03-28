# HOMEGOLF Local CUDA Operator Scratch Runner Audit

Date: 2026-03-28

## Scope

This audit records the next operational fix in the HOMEGOLF local-CUDA loop on
`archlinux`.

The training code was already honest about:

- impossible `600` second competitive postures
- invalid strict-input semantics
- real local one-step proof runs

The remaining blocker was operator quality:

- ad-hoc local HOMEGOLF runs still defaulted to `/tmp/...` report and log paths
- Rust temp files also defaulted to `/tmp`
- the remote `archlinux` checkout had drifted behind the local clean
  `psionic/main`

That combination made repeatable local reruns less reliable than they should
have been.

## What Changed

Added one repo-owned operator wrapper:

- `scripts/run-parameter-golf-homegolf-local-cuda.sh`

It now provides one stable launch path for local HOMEGOLF CUDA reruns with:

- clean-worktree refusal by default
- `main` branch refusal by default
- optional `git pull --ff-only` through `--sync-main`
- scratch-root output under `~/scratch/psionic_homegolf_runs/<run_id>`
- explicit `TMPDIR` override away from `/tmp`
- stable `training_report.json`, `train.log`, `train.pid`, and `launch.env`
  paths
- either `cargo run` from current source or an explicit prebuilt binary

This is not a modeling change.

It is an operator-path correction for the reachable `archlinux` HOMEGOLF lane.

## Live Trigger For This Fix

On 2026-03-28, the next local reruns exposed two concrete problems:

1. The remote repo on `archlinux` was still at `de8bb658` while the local clean
   `psionic/main` here was already at `2581bbba`.
2. Fresh rerun attempts hit storage/quota failures in the old ad-hoc posture:
   - `error: failed to write /tmp/rustc.../lib.rmeta: Disk quota exceeded`
   - `tee: /tmp/psionic_homegolf_runs/...: Disk quota exceeded`

Those failures were not new modeling truth.

They were a missing operator wrapper around a still-reachable lane.

## Why This Improves On Previous Work

Compared with the earlier HOMEGOLF local audits, this change improves the
iteration loop in three concrete ways:

1. It stops relying on manual one-off commands with `/tmp` outputs that drift
   across machines and sessions.
2. It makes clean-main posture part of the runnable contract instead of a
   remembered human step.
3. It gives the local `4080` lane one reusable launch surface that can hold
   long score-closeout runs without rediscovering the same temp-path failure.

## Honest Boundary After This Audit

What is true:

- the local HOMEGOLF CUDA runner now has one repo-owned scratch-first operator
  surface
- the live Google single-H100 path is still blocked by zero available H100
  quota in `openagentsgemini`
- the live RunPod path is still not directly launchable from this shell because
  no `RUNPOD_*` credential is present in the current environment

What is not true:

- this did not create a new PGOLF score by itself
- this did not make the exact competitive `4080` lane suddenly fit
- this did not remove the need for long full-validation closeout on local
  actual-score runs
