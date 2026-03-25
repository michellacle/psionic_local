# MLX Dense-Rank Runtime Reference

> Status: canonical `XTRAIN-20` / `#536` record, updated 2026-03-25 after
> landing the bounded MLX dense-rank runtime contract in
> `crates/psionic-train/src/mlx_dense_rank_runtime.rs`.

This document records the first truthful MLX-backed dense-rank runtime surface
inside `psionic-train`.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-mlx-dense-rank-runtime-contract.sh
```

## What Landed

`psionic-train` now owns one bounded MLX dense-rank runtime contract with:

- one shared `DenseRankRuntimeIdentity` for `psionic.dense_rank_runtime.mlx_metal.v1`
- one MLX-specific `DenseRankRuntimeExecutionReceipt` that reuses the same
  bootstrap and train-step receipt family as the generic dense CUDA runtime
- one retained Mac MLX bring-up binding through
  `fixtures/swarm/reports/swarm_mac_mlx_bringup_v1.json`
- one retained single-rank checkpoint manifest and pointer under the canonical
  pretraining checkpoint family
- one retained local metric-event family under the shared
  `LocalTrainMetricEvent` vocabulary
- one explicit final-evidence projection into the shared
  `TrainingExecutionEvidenceBundle` family
- one explicit refusal set for BF16 mixed precision, cross-host collectives,
  sharded optimizer-state exchange, and same-job mixed-backend dense meshes
- the binary `mlx_dense_rank_runtime_contract`
- the checker `scripts/check-mlx-dense-rank-runtime-contract.sh`
- the fixture `fixtures/training/mlx_dense_rank_runtime_contract_v1.json`

## What It Changes

The local Apple source no longer stops at contributor-only truth.

It now has one bounded dense-rank runtime claim:

- one single-rank MLX Metal runtime
- one generic dense-rank bootstrap receipt
- one generic dense-rank train-step receipt
- one durable single-rank checkpoint surface
- one local metric sink surface that uses the same train, validation,
  checkpoint, and summary phase vocabulary as the rest of the train system

That is enough to stop calling the local Mac lane contributor-only.

## Current Honest Boundary

This issue does not claim:

- cross-host MLX collectives
- same-job CUDA-plus-MLX dense training
- BF16 mixed precision on the MLX dense path
- sharded optimizer exchange or mixed-backend checkpoint portability
- checkpoint-writer authority for the local Apple source

This issue closes one truthful dense rank on Metal first. The later same-job
mixed-backend issues still have to define the shared math, optimizer, and
checkpoint contracts above it.
