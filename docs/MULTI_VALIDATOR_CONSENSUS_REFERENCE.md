# Multi-Validator Consensus Reference

> Status: canonical `XTRAIN-38` / `#581` record, updated 2026-03-26 after
> landing the first public multi-validator consensus contract.

## What This Closes

Psionic now owns one typed multi-validator checkpoint-promotion surface above
the validator challenge and scoring contract.

The new contract lives in:

- `crates/psionic-train/src/multi_validator_consensus_contract.rs`
- `crates/psionic-train/src/bin/multi_validator_consensus_contract.rs`
- `fixtures/training/multi_validator_consensus_contract_v1.json`
- `scripts/check-multi-validator-consensus-contract.sh`

This issue closes the first truthful answer to:

- how many validators must participate before checkpoint authority can decide
- how validator votes are weighted
- when disagreement or replay-required votes hold promotion
- how checkpoint authority records a held decision instead of silently picking a
  side

## Contract Shape

The canonical contract freezes:

- one consensus policy
- two validator votes
- one checkpoint-promotion decision
- one disagreement receipt

## Current Canonical Binding

The contract binds directly to:

- the decentralized network contract
- the validator challenge and scoring contract
- the shared validator promotion contract
- the content-addressed artifact exchange contract

That means checkpoint authority now has one machine-legible bridge from
validator score receipts to held or promoted checkpoint state.

## Canonical Decision

The first canonical decision is:

- `decision.checkpoint.step2048.round2056`

That decision stays `held_no_promotion` because the current quorum reached the
minimum validator count but not unanimous acceptance:

- Google votes `accepted`
- Apple MLX votes `replay_required`

## Honest Disagreement Boundary

The first explicit disagreement receipt is:

- `disagreement.checkpoint.step2048.round2056`

That receipt prevents checkpoint promotion from collapsing into informal human
judgment whenever validators disagree.

## Pass Criteria

The contract is green only if all of the following stay true:

- the quorum still matches the decentralized network governance contract
- vote weights still total ten thousand basis points
- replay-required votes still block promotion
- disagreement still holds the checkpoint candidate instead of promoting it

## Current Limits

This issue intentionally does not claim:

- fraud penalties or slashing
- reward accounting
- settlement publication

This issue freezes multi-validator authority first: quorum, weighted votes,
held-no-promotion decisions, and disagreement receipts.
