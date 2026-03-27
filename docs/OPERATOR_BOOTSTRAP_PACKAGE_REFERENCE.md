# Operator Bootstrap Package Reference

> Status: canonical `XTRAIN-42` / `#586` record, updated 2026-03-26 after
> landing the first public miner and validator bootstrap packages.

## What This Closes

Psionic now owns one typed operator package surface for decentralized miners
and validators.

The new contract lives in:

- `crates/psionic-train/src/operator_bootstrap_package_contract.rs`
- `crates/psionic-train/src/bin/operator_bootstrap_package_contract.rs`
- `fixtures/training/operator_bootstrap_package_contract_v1.json`
- `scripts/check-operator-bootstrap-package-contract.sh`

This issue closes the first truthful answer to:

- which container images, env manifests, and commands define the miner and
  validator bootstrap kits
- which preflight checks are required before a node can even attempt public
  registration
- which canonical miner session and validator replay receipt operators can use
  for dry runs

## Contract Shape

The canonical contract freezes:

- two operator packages
- four preflight checks
- two bootstrap kits

## Current Limits

This issue intentionally does not claim:

- public run explorer visibility
- public testnet graduation
- curated or open public training runs

This issue freezes the package boundary first: reproducible images, commands,
and preflight gates for the first public roles.
