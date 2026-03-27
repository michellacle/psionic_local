# Reward Ledger Reference

> Status: canonical `XTRAIN-40` / `#584` record, updated 2026-03-26 after
> landing the first tamper-evident public reward ledger.

## What This Closes

Psionic now owns one typed contribution-accounting and reward-ledger surface
above validator scoring, multi-validator consensus, and fraud policy.

The new contract lives in:

- `crates/psionic-train/src/reward_ledger_contract.rs`
- `crates/psionic-train/src/bin/reward_ledger_contract.rs`
- `fixtures/training/reward_ledger_contract_v1.json`
- `scripts/check-reward-ledger-contract.sh`

This issue closes the first truthful answer to:

- which public work items counted as miner, validator, or checkpoint-authority
  earnings
- how penalties are carried into the same retained ledger as positive work
- how net allocations reconcile against the ledger budget
- which participants are payout-eligible and which remain negative or withheld

## Contract Shape

The canonical contract freezes:

- one accounting period
- five contribution entries
- one penalty entry
- four final allocations

## Current Canonical Binding

The contract binds directly to:

- the decentralized network contract
- the signed node identity contract set
- the validator challenge scoring contract
- the multi-validator consensus contract
- the fraud quarantine slashing contract

That means the first public reward ledger is no longer prose. It is a typed
surface that reconciles public work, penalties, and payout weights.

## Canonical Allocations

The current net allocations are:

- `allocation.local_mlx.window1231`
- `allocation.google.window1231`
- `allocation.runpod.window1231`
- `allocation.local_rtx4080.window1231`

Those allocations keep one important honesty boundary explicit:

- the slashed RTX 4080 miner remains in the ledger as a negative allocation
- positive allocations alone retain payout weight

## Pass Criteria

The contract is green only if all of the following stay true:

- contribution entries still reconcile exactly to the reward budget
- penalty entries still reconcile exactly to the penalty pool
- positive allocations still carry the full `10_000` basis points of payout
  weight
- negative allocations cannot silently keep payout weight

## Current Limits

This issue intentionally does not claim:

- published settlement exports
- chain-backed payout publication
- public dashboards or operator-facing explorer views

This issue freezes accounting truth first: one retained ledger with positive
work, penalties, and reconciled net allocations.
