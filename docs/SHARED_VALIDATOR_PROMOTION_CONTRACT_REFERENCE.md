# Shared Validator And Promotion Contract Reference

> Status: canonical `XTRAIN-15` / `#531` record, updated 2026-03-25 after
> landing the shared validator and promotion contract in
> `crates/psionic-train/src/validator_promotion_contract.rs`.

This document records the shared vocabulary for acceptance, quarantine,
rejection, replay-required posture, and promotion outcomes across the current
training execution classes.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-shared-validator-promotion-contract.sh
```

## What Landed

`psionic-train` now owns one shared validator and promotion contract that
freezes:

- the admitted validator dispositions: `accepted`, `quarantined`, `rejected`,
  and `replay_required`
- the admitted promotion outcomes: `promoted_revision`, `held_no_promotion`,
  and `refused_promotion`
- execution-class policies that keep quarantine and replay-required as
  promotion-blocking outcomes
- the binding from the provider-neutral evidence bundle back to this shared
  contract

The landed surface includes:

- `SharedValidatorPromotionContract`
- `SharedValidatorExecutionClassPolicy`
- the binary `shared_validator_promotion_contract`
- the checker `scripts/check-shared-validator-promotion-contract.sh`
- the committed fixture `fixtures/training/shared_validator_promotion_contract_v1.json`

## Why This Matters

The repo already had validator and promotion language in bounded subsystems,
but not one shared contract above them. After this issue, replay and quarantine
no longer get to drift by provider or execution class.

## Current Limits

This issue intentionally does not claim:

- app-review workflow closure
- weakening of replay or provenance refusal posture
- automatic promotion when the shared contract still blocks it

