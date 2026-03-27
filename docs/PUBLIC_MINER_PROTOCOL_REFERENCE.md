# Public Miner Protocol Reference

> Status: canonical `XTRAIN-36` / `#580` record, updated 2026-03-26 after
> landing the first public miner execution protocol.

## What This Closes

Psionic now owns one typed public miner protocol above the public assignment,
dataset-authority, catch-up, outer-sync, and content-addressed artifact
exchange contracts.

The new contract lives in:

- `crates/psionic-train/src/public_miner_protocol_contract.rs`
- `crates/psionic-train/src/bin/public_miner_protocol_contract.rs`
- `fixtures/training/public_miner_protocol_contract_v1.json`
- `scripts/check-public-miner-protocol-contract.sh`

This issue closes the first truthful answer to:

- what the public miner execution class actually is
- how a miner binds assignment intake to dataset authority
- how local train-step closure binds to published delta artifacts
- how checkpoint synchronization stays explicit for rejoined miners
- how stale standby miners are refused instead of drifting into active windows

## Contract Shape

The canonical contract freezes:

- one public miner execution-class binding
- one bounded retry policy
- two active canonical miner sessions
- two local-step receipts
- two delta-upload receipts
- two checkpoint-sync receipts
- one refusal for a stale standby miner

## Current Canonical Binding

The contract binds directly to:

- the public work-assignment contract
- the public dataset authority contract
- the live checkpoint catch-up contract
- the quantized outer-sync contract
- the content-addressed artifact exchange contract

That means public miner execution now has one machine-legible path from
assignment receipt to dataset receipt to delta artifact to checkpoint sync.

## Execution-Class Boundary

The protocol freezes one explicit execution-class id:

- `psionic.execution_class.public_miner.v1`

It still binds back to the current legacy assignment execution class
`validated_contributor_window`, which keeps the current program manifest honest
while turning the public miner lane into a first-class protocol surface.

## Canonical Active Sessions

The current canonical sessions are:

- `session.public_miner.google.window1231`
- `session.public_miner.local_mlx.window1231`

Those two sessions show the expected happy path:

- deterministic assignment intake
- dataset-authority binding
- sixty-four local train steps
- quantized delta publication
- checkpoint synchronization
- window finalization

## Honest Refusal Boundary

The first explicit refusal is:

- `refusal.public_miner.local_rtx4080.checkpoint_lag`

That refusal keeps the stale RTX 4080 standby miner out of the active window
because the already-landed catch-up contract refused its checkpoint recovery.

## Pass Criteria

The contract is green only if all of the following stay true:

- active sessions still bind to miner assignments only
- session dataset receipts still match the admitted miner data receipts
- local-step receipts still consume the assigned dataset pages exactly
- delta-upload receipts still target peer-seed backends and admitted outer-sync
  exchange ids
- checkpoint-sync receipts still bind to the admitted live checkpoint artifact
- exactly one stale-standby refusal remains explicit

## Current Limits

This issue intentionally does not claim:

- validator verdict truth
- checkpoint-promotion consensus
- reward accounting or settlement

This issue freezes miner execution first: assignment intake, bounded retries,
local-step closure, delta publication, checkpoint sync, and refusal posture.
