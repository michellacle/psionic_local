# Fraud Quarantine Slashing Reference

> Status: canonical `XTRAIN-39` / `#583` record, updated 2026-03-26 after
> landing the first sybil-watch, quarantine, and slashing policy contract.

## What This Closes

Psionic now owns one typed adversarial-discipline layer above signed node
identity, public dataset authority, validator scoring, and multi-validator
consensus.

The new contract lives in:

- `crates/psionic-train/src/fraud_quarantine_slashing_contract.rs`
- `crates/psionic-train/src/bin/fraud_quarantine_slashing_contract.rs`
- `fixtures/training/fraud_quarantine_slashing_contract_v1.json`
- `scripts/check-fraud-quarantine-slashing-contract.sh`

This issue closes the first truthful answer to:

- which retained evidence classes count as sybil-watch or replay-abuse signals
- how Psionic distinguishes observation-only incidents from blocked miners
- when a slashing decision is retained instead of left in operator notes
- how appeals are modeled without pretending every appeal succeeds

## Contract Shape

The canonical contract freezes:

- four fraud signals
- two quarantine decisions
- one slashing decision
- one appeal window

## Current Canonical Binding

The contract binds directly to:

- the signed node identity contract set
- the public dataset authority contract
- the validator challenge scoring contract
- the multi-validator consensus contract

That means fraud policy now has one typed path from identity and data truth to
incident evidence, quarantine, and penalty semantics.

## Canonical Signals

The current signals are:

- `signal.sybil.local_rtx4080.wallet_watch`
- `signal.replay.local_rtx4080.window1230`
- `signal.disagreement.local_mlx.window1231`
- `signal.software.runpod.release_watch`

Those examples keep the first public fraud boundary honest:

- duplicate-wallet or duplicate-capability risk is retained as a sybil-watch
  input
- duplicate work is retained as a replay-abuse input
- validator disagreement is tracked separately from hard fraud
- stale-software drift has a retained evidence shape before it becomes a ban

## Honest Refusal Boundary

The first explicit penalty path is:

- `slash.public_miner.local_rtx4080.window1232`

That penalty stays tied to one blocked miner quarantine and one denied appeal.

## Pass Criteria

The contract is green only if all of the following stay true:

- the duplicate replay signal still binds to the canonical duplicate
  anti-replay receipt
- slashing decisions still attach to a real quarantine decision
- nonpositive or empty penalty decisions stay invalid
- appeal windows still point at a real challenged action

## Current Limits

This issue intentionally does not claim:

- automatic fraud-proof generation
- fully permissionless public admission
- on-chain slashing execution

This issue freezes policy truth first: retained signals, retained incident
classes, retained slashing semantics, and one explicit appeal path.
