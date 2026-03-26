# Signed Node Identity Contract Reference

> Status: canonical `XTRAIN-26` / `#572` record, updated 2026-03-26 after
> landing the first signed node identity contract set in
> `crates/psionic-train/src/signed_node_identity_contract.rs`.

This document records the first signed public-node identity layer for the
decentralized Psionic training network.

## Canonical Runner

Run the contract checker from the repo root:

```bash
scripts/check-signed-node-identity-contract-set.sh
```

## What Landed

`psionic-train` now owns one signed identity contract set above the retained
compute-source contracts and the decentralized network contract.

The landed surface includes:

- `SignedNodeIdentityContractSet`
- `SignedNodeIdentityRecord`
- `SignedNodeIdentityWalletBinding`
- `SignedNodeSoftwareAttestation`
- `SignedNodeCapabilityProjection`
- `SignedNodeBenchmarkEvidence`
- `SignedNodeCapabilitySignature`
- `SignedNodeCapabilityRefusalExample`
- `SignedNodeIdentityRevocationAuthority`
- `write_signed_node_identity_contract_set(...)`
- the canonical fixture
  `fixtures/training/signed_node_identity_contract_set_v1.json`
- the checker
  `scripts/check-signed-node-identity-contract-set.sh`

## What The Contract Makes Explicit

The first signed identity layer freezes these seams in one machine-legible
surface:

- one node identity record per current canonical compute source
- one wallet namespace per admitted node
- one deterministic software build digest per admitted node
- one capability projection digest over accelerator, backend, network, and
  storage posture
- one retained benchmark-evidence bundle per admitted node
- one admitted public-role set per node
- one admitted execution-class set per node
- one deterministic ed25519 signature over the signed payload
- one revocation feed and grace-window posture
- one explicit refusal surface for unsupported role or execution-class claims

## Current Canonical Node Set

The first canonical set covers the current four retained compute sources:

- `google_l4_validator_node`
- `runpod_8xh100_dense_node`
- `local_rtx4080_workstation`
- `local_mlx_mac_workstation`

The current public-role admissions are intentionally uneven:

- Google is the strongest multi-role node and currently carries
  `public_miner`, `public_validator`, `checkpoint_authority`, `aggregator`,
  and `relay`
- RunPod carries `checkpoint_authority` and `aggregator`, but explicitly does
  not claim `public_miner` because the current network still binds that role to
  `validated_contributor_window` instead of `dense_full_model_rank`
- the RTX 4080 workstation carries `public_miner` and `aggregator`
- the Apple MLX workstation carries `public_miner`, `public_validator`, and
  `aggregator`

## Existing Psionic Binding

The contract set does not invent a second discovery or admission authority.

Each identity record binds directly to:

- `CrossProviderComputeSourceContract`
- `DecentralizedNetworkContract`
- the root cross-provider training-program manifest environment key
- retained authority artifacts already cited by the source contracts

That means wallet, build digest, capability projection, benchmark evidence, and
public role claims are all anchored back to canonical Psionic-owned source
truth instead of mutable operator prose.

## Pass Criteria

The contract set is green only if all of the following stay true:

- the committed fixture matches the generator output exactly
- the identity set stays aligned with the current four canonical compute
  sources
- every identity stays bound to the decentralized network contract digest
- every identity keeps a deterministic ed25519 signature over its signed
  payload
- relay admission remains explicit instead of hidden in endpoint prose
- unsupported role or execution-class claims stay fail-closed through typed
  refusals

## Current Limits

This issue intentionally does not claim:

- hardware-backed or TPM-backed remote attestation
- on-chain wallet verification
- permissionless registration
- automatic slashing or payout execution
- public dense-rank role binding for `dense_full_model_rank`
- live network discovery or matchmaking

This issue freezes signed node identity, wallet, capability, benchmark, and
revocation truth first.
