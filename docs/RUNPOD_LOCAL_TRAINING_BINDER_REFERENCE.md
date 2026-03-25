# RunPod And Local Training Binder Reference

> Status: canonical `XTRAIN-12` / `#528` record, updated 2026-03-25 after
> landing the RunPod plus local training binder projection in
> `crates/psionic-train/src/runpod_local_training_binder_projection.rs`.

This document records the non-Google projection layer that binds the current
RunPod `8xH100` lane and the first local trusted-LAN swarm lane to the shared
cross-provider runtime binder.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-runpod-local-training-binder-projection.sh
```

## What Landed

`psionic-train` now owns one non-Google projection set that freezes:

- the exact shared runtime binding used by the RunPod `8xH100` lane
- the exact shared runtime binding used by the first local trusted-LAN lane
- the retained RunPod and local authority artifacts, launch scripts, checker
  surfaces, and evidence surfaces that must stay green under the binder

The landed surface includes:

- `RunPodLocalTrainingBinderProjectionSet`
- `RunPodLocalTrainingBinderLaneProjection`
- the binary `runpod_local_training_binder_projection`
- the checker `scripts/check-runpod-local-training-binder-projection.sh`
- the committed fixture `fixtures/training/runpod_local_training_binder_projection_v1.json`

## Why This Matters

The cross-provider system does not close if non-Google lanes keep their own
training-facing launch semantics. After this issue, RunPod and the current
local trusted-LAN path still keep their honest operator boundaries, but the
runtime env, artifact roots, and final evidence expectations now come from the
same shared binder Google uses.

## Current Limits

This issue intentionally does not claim:

- that the RunPod `8xH100` lane already closes real dense-runtime execution
- that the local trusted-LAN lane has become a successful mixed-backend dense
  trainer
- that workstation bring-up reports by themselves are the same thing as a live
  launch lane

