# Public Run Explorer Reference

> Status: canonical `XTRAIN-43` / `#587` record, updated 2026-03-26 after
> landing the first public run explorer contract.

## What This Closes

Psionic now owns one typed public explorer surface above registry, consensus,
reward accounting, and settlement publication.

The new contract lives in:

- `crates/psionic-train/src/public_run_explorer_contract.rs`
- `crates/psionic-train/src/bin/public_run_explorer_contract.rs`
- `fixtures/training/public_run_explorer_contract_v1.json`
- `scripts/check-public-run-explorer-contract.sh`

This issue closes the first truthful answer to:

- which panes the public run explorer must publish
- how one current network snapshot is represented
- how score rows reconcile against the reward ledger
- how stale-data state is surfaced instead of silently serving old information

That surface is now the explorer-foundation contract beneath the dedicated
decentralized `XTRAIN Explorer` artifact family documented in
`docs/XTRAIN_EXPLORER_REFERENCE.md`. The public run explorer contract keeps the
pane and stale-policy layer stable; the XTRAIN explorer artifacts add the
participant graph, active-window, checkpoint, and settlement state needed for
the new ML-training explorer pane.

## Contract Shape

The canonical contract freezes:

- six explorer panes
- one network snapshot
- four score rows
- six stale-data policies

## Current Limits

This issue intentionally does not claim:

- public testnet onboarding gates
- curated internet runs
- open public participation windows

This issue freezes observability truth first: panes, score rows, settlement
visibility, and stale-data posture.
