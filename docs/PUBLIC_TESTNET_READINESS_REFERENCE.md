# Public Testnet Readiness Reference

> Status: canonical `XTRAIN-44` / `#588` record, updated 2026-03-26 after
> landing the first staged public-testnet readiness contract.

## What This Closes

Psionic now owns one typed staged-onboarding surface above fraud policy,
operator packages, and the explorer.

The new contract lives in:

- `crates/psionic-train/src/public_testnet_readiness_contract.rs`
- `crates/psionic-train/src/bin/public_testnet_readiness_contract.rs`
- `fixtures/training/public_testnet_readiness_contract_v1.json`
- `scripts/check-public-testnet-readiness-contract.sh`

This issue closes the first truthful answer to:

- which candidates are reward-eligible, canary-only, or blocked
- which package and compliance receipts each candidate had to pass
- how fraud policy blocks a candidate instead of relying on maintainer memory

## Contract Shape

The canonical contract freezes:

- five testnet candidates
- eight compliance receipts
- five graduation decisions

## Current Limits

This issue intentionally does not claim:

- curated internet-scale run evidence
- outside-operator public run closure
- incentivized settlement in a live public run

This issue freezes staged readiness first: candidates, compliance checks,
canary posture, reward-eligible posture, and explicit blocked admission.
