# Public Network Registry Reference

> Status: canonical `XTRAIN-27` / `#570` record, updated 2026-03-26 after
> landing the first public registry, discovery, and matchmaking contract in
> `crates/psionic-train/src/public_network_registry_contract.rs`.

This document records the first shared public-network registry layer for the
decentralized Psionic training network.

## Canonical Runner

Run the contract checker from the repo root:

```bash
scripts/check-public-network-registry-contract.sh
```

## What Landed

`psionic-train` now owns one registry and discovery contract above the signed
node identity layer.

The landed surface includes:

- `PublicNetworkRegistryContract`
- `PublicNetworkRegistryRecord`
- `PublicNetworkCompatibilityPolicy`
- `PublicNetworkRegistryCompatibility`
- `PublicNetworkEndpoint`
- `PublicNetworkDiscoveryFilter`
- `PublicNetworkDiscoveryExample`
- `PublicNetworkDiscoveryRefusal`
- `PublicNetworkMatchmakingOffer`
- `write_public_network_registry_contract(...)`
- the canonical fixture
  `fixtures/training/public_network_registry_contract_v1.json`
- the checker
  `scripts/check-public-network-registry-contract.sh`

## What The Contract Makes Explicit

The first registry layer freezes these seams in one machine-legible object:

- one registry record per current signed node identity
- one current epoch id for discovery and matchmaking
- one compatibility policy over signed-node release id, environment key,
  manifest digest, and revocation posture
- one endpoint set per registry record
- one relay-posture field instead of hidden relay assumptions
- one typed discovery filter and refusal surface
- one typed matchmaking-offer surface for miners, validators, and checkpoint
  authorities

## Current Canonical Registry

The first canonical registry still covers the same four retained nodes:

- `google_l4_validator_node.registry`
- `runpod_8xh100_dense_node.registry`
- `local_rtx4080_workstation.registry`
- `local_mlx_mac_workstation.registry`

The first discovery layer proves these current outcomes:

- `discover_public_miner_nodes` resolves Google plus the two local
  contributor-window nodes
- `discover_public_validator_nodes` resolves Google plus the Apple MLX node
- `discover_checkpoint_authority_nodes` resolves Google plus RunPod
- `discover_relay_nodes` resolves Google only

That keeps the current public-network shape honest: one relay-capable node, two
validators, two checkpoint authorities, and three contributor-window miners.

## Existing Psionic Binding

The registry contract does not invent a second admission authority.

Each registry record binds directly to:

- `SignedNodeIdentityContractSet`
- `DecentralizedNetworkContract`
- the canonical cross-provider training-program manifest
- the canonical compute-source trust tier already retained in the source
  contracts

That means discovery and matchmaking stay anchored to signed identity, current
role admission, current execution-class admission, current trust tier, and
current compatibility policy all at once.

## Matchmaking Surfaces

The first canonical offers are intentionally concrete:

- `public_miner_window_offer_v1`
- `validator_quorum_offer_v1`
- `checkpoint_promotion_offer_v1`

They prove that the current network can pick active miners, a validator quorum,
relay support, and checkpoint authorities from one shared registry surface
instead of private host lists or operator memory.

## Pass Criteria

The contract is green only if all of the following stay true:

- the committed fixture matches the generator output exactly
- the registry record set stays aligned with the signed node identity set
- discovery queries cover the whole registry and produce typed refusals for
  non-matches
- relay posture remains explicit and only the relay-capable node matches relay
  discovery
- the validator quorum offer still resolves to two admitted validators

## Current Limits

This issue intentionally does not claim:

- permissionless registry gossip
- public internet admission
- automated stake or reward market selection
- live elastic runtime membership
- NAT traversal or overlay path selection

This issue freezes public registry, discovery, compatibility, and matchmaking
truth first.
