# Live Checkpoint Catch-Up Reference

> Status: canonical `XTRAIN-30` / `#574` record, updated 2026-03-26 after
> landing the first live join-time catch-up contract.

## What This Closes

Psionic now owns one truthful live checkpoint catch-up layer above the
distributed checkpoint, dense recovery, elastic mesh, and WAN route contracts.

The new contract lives in:

- `crates/psionic-train/src/live_checkpoint_catchup_contract.rs`
- `crates/psionic-train/src/bin/live_checkpoint_catchup_contract.rs`
- `fixtures/training/live_checkpoint_catchup_contract_v1.json`
- `scripts/check-live-checkpoint-catchup-contract.sh`

This issue closes the first explicit answer to:

- which peers are actually allowed to advertise live checkpoint service
- which resume windows are admitted for join-time recovery
- how one replacement node rejoins through a live path instead of a cold reset
- how stale or optimizer-incomplete sidecar recovery is refused

## Contract Shape

The canonical contract freezes:

- one advertisement record per admitted live checkpoint source
- one explicit resume window set with freshness thresholds
- one completed catch-up receipt
- one refused catch-up receipt

## Canonical Sources

The fixture retains three live sources:

- `advertisement.checkpoint_authority.runpod.primary`
  RunPod is the primary full-state live catch-up source.
- `advertisement.checkpoint_authority.google.mirror`
  Google mirrors the same admitted checkpoint as a second checkpoint-authority
  lane.
- `advertisement.active_peer.local_mlx.sidecar`
  Apple MLX exposes a faster sidecar lane, but it does not claim optimizer
  completeness.

## First Successful Live Catch-Up

The first completed receipt is:

- `catchup.public_miner.local_mlx.after_deathrattle`

That receipt proves:

- the Apple MLX miner is admitted after the mesh replacement revision
- RunPod serves the canonical checkpoint
- the path is the overlay route
  `route.checkpoint_authority.local_mlx_runpod.overlay`
- the canonical restore assignment is satisfied

So Psionic now has one live join path that is not just a post-barrier restart
story.

## First Honest Refusal

The first refused receipt is:

- `catchup.public_miner.local_rtx4080.partial_state_refused`

That refusal keeps two truths explicit:

- the joiner is too stale for the sidecar freshness window
- the MLX sidecar does not claim optimizer-state completeness

So Psionic does not blur active-peer sidecars into checkpoint-authority
equivalence.

## Existing Psionic Binding

The catch-up contract binds directly to:

- `ShardedDistributedCheckpointContract`
- `DenseRankRecoveryContract`
- `ElasticDeviceMeshContract`
- `WanOverlayRouteContract`

That means join-time recovery now stays tied to:

- the admitted checkpoint manifest and pointer
- retained restore assignments
- admitted mesh revisions
- admitted WAN route ids

## Pass Criteria

The contract is green only if all of the following stay true:

- checkpoint advertisements stay tied to the admitted checkpoint digests
- only active checkpoint authorities may advertise full-state recovery
- active peer sidecars stay explicitly weaker than checkpoint authorities
- at least one completed catch-up receipt remains present
- at least one stale or partial-state refusal remains present

## Current Limits

This issue intentionally does not claim:

- WAN-efficient delta exchange
- quantized outer synchronization
- public internet fault or soak closure

This issue freezes the truthful live join path first: advertisement,
freshness-window, completion, and refusal semantics.
