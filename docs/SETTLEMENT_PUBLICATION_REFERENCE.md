# Settlement Publication Reference

> Status: canonical `XTRAIN-41` / `#585` record, updated 2026-03-26 after
> landing the first weight-publication and payout-ready settlement contract.

## What This Closes

Psionic now owns one typed settlement-publication surface above the reward
ledger.

The new contract lives in:

- `crates/psionic-train/src/settlement_publication_contract.rs`
- `crates/psionic-train/src/bin/settlement_publication_contract.rs`
- `fixtures/training/settlement_publication_contract_v1.json`
- `scripts/check-settlement-publication-contract.sh`

This issue closes the first truthful answer to:

- how validator weights are published for a closed accounting window
- how one signed-ledger settlement record exposes payout-ready results
- which wallet destinations receive the positive exports
- how Psionic refuses optional chain publication instead of implying it exists

## Contract Shape

The canonical contract freezes:

- two validator weight publications
- one signed-ledger settlement record
- three payout exports
- one chain-adapter refusal

## Current Canonical Binding

The contract binds directly to:

- the decentralized network contract
- the signed node identity contract set
- the reward ledger contract
- the validator challenge scoring contract

That means settlement publication now has one typed path from scoring and
ledger truth to exportable payout artifacts.

## Canonical Publication Boundary

The first retained settlement record is:

- `settlement.window1231.signed`

The first explicit refusal is:

- `refusal.settlement.chain.window1231`

That keeps the backend claim boundary honest:

- signed-ledger export exists now
- chain publication remains explicitly disabled rather than silently absent

## Pass Criteria

The contract is green only if all of the following stay true:

- validator weight publications still total `10_000` basis points
- settlement publication still excludes negative allocations
- payout exports still target canonical wallet bindings from node identity
- the chain refusal still stays explicit

## Current Limits

This issue intentionally does not claim:

- public explorer packaging
- operator bootstrap kits
- open public-run onboarding

This issue freezes publication truth first: weight publication, signed-ledger
settlement, payout-ready exports, and an explicit refusal posture for disabled
backends.
