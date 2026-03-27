# 2026-03-26 XTRAIN System State And Operations Audit

## Why This Audit Exists

`XTRAIN-24` through `XTRAIN-48` are now closed, but the closure spans several
distinct kinds of artifacts:

- typed Rust contract generators in `crates/psionic-train/src`
- canonical JSON fixtures in `fixtures/training`
- cargo-backed checker scripts in `scripts`
- focused reference docs in `docs`
- retained proof-run and acceptance audits in `docs/audits`

That means "the XTRAIN system is implemented" needs a more exact reading than a
single acceptance note.

This audit answers four concrete questions:

- what the decentralized XTRAIN stack currently contains
- what has been tested and re-checked as of this audit
- how an engineer should operate the current stack
- what remains honestly blocked or unproven

## Bottom Line

Psionic now has a full decentralized-training contract and evidence stack on
top of the earlier provider-neutral cross-provider training substrate from
`XTRAIN-0` through `XTRAIN-23`.

The current system is strongest as a typed, cross-linked, retained authority
surface. It now covers:

- decentralized network authority and signed participant identity
- public registry and discovery
- WAN-tolerant elastic runtime and catch-up surfaces
- deterministic public work assignment and dataset authority
- content-addressed artifact exchange
- public miner, validator, consensus, fraud, reward, and settlement surfaces
- operator bootstrap kits, public explorer visibility, and testnet readiness
- one curated run, one open public run, one incentivized run, and one final
  acceptance audit

The system is not yet fully cargo-green in this checkout. The shared blocker is
the pre-existing compile failure in
`crates/psionic-backend-cuda/src/lib.rs:876`, where
`PlatformSubmission::allocate` is missing. That blocker currently prevents the
repo's cargo-backed contract checkers from completing end to end.

So the honest current state is:

- the XTRAIN system is implemented as a coherent contract stack
- its fixtures, references, and retained evidence are present and internally
  cross-linked
- its cargo-backed regeneration path is still blocked by unrelated workspace
  breakage

## Scope Covered By This Audit

This audit covers the decentralized continuation beginning with `XTRAIN-25` and
ending with `XTRAIN-48`, plus the `XTRAIN-24` umbrella closeout:

- `XTRAIN-25` network epoch, role, and governance contract
- `XTRAIN-26` signed node identity contract set
- `XTRAIN-27` public network registry and discovery
- `XTRAIN-28` elastic internet device mesh
- `XTRAIN-29` WAN overlay and route policy
- `XTRAIN-30` live checkpoint catch-up
- `XTRAIN-31` quantized outer sync
- `XTRAIN-32` internet fault harness
- `XTRAIN-33` deterministic public work assignment
- `XTRAIN-34` public dataset authority
- `XTRAIN-35` content-addressed artifact exchange
- `XTRAIN-36` public miner protocol
- `XTRAIN-37` validator challenge scoring
- `XTRAIN-38` multi-validator consensus
- `XTRAIN-39` fraud, quarantine, and slashing
- `XTRAIN-40` reward ledger
- `XTRAIN-41` settlement publication
- `XTRAIN-42` operator bootstrap packages
- `XTRAIN-43` public run explorer
- `XTRAIN-44` public testnet readiness
- `XTRAIN-45` curated decentralized run
- `XTRAIN-46` open public decentralized run
- `XTRAIN-47` incentivized decentralized run
- `XTRAIN-48` final decentralized training acceptance audit

## Current System Shape

### 1. Authority And Identity

The stack starts from one canonical network contract, one signed identity set,
and one registry surface:

- `fixtures/training/decentralized_network_contract_v1.json`
- `fixtures/training/signed_node_identity_contract_set_v1.json`
- `fixtures/training/public_network_registry_contract_v1.json`

Current fixture facts:

- the network contract freezes 5 role bindings
- the identity set contains 4 signed identities
- the registry contains 4 registry records, 4 discovery examples, and 3
  matchmaking offers

Operationally, this is the authority layer that answers:

- what network is this
- what roles exist
- who may participate
- what capabilities and compatibility posture each node claims

### 2. Internet Runtime Mechanics

The runtime layer now has explicit contract surfaces for membership churn,
transport, catch-up, low-bandwidth synchronization, and realism testing:

- `fixtures/training/elastic_device_mesh_contract_v1.json`
- `fixtures/training/wan_overlay_route_contract_v1.json`
- `fixtures/training/live_checkpoint_catchup_contract_v1.json`
- `fixtures/training/quantized_outer_sync_contract_v1.json`
- `fixtures/training/internet_fault_harness_contract_v1.json`

Current fixture facts:

- elastic mesh: 5 role lease policies, 8 member leases, 8 heartbeat samples,
  1 deathrattle, 5 revision receipts
- WAN overlay: 4 NAT postures, 4 route-quality samples, 4 route records,
  1 failover receipt
- catch-up: 3 advertisements, 2 resume windows, 2 catch-up receipts
- outer sync: 3 delta policies, 3 exchange receipts, 1 aggregation receipt,
  2 correctness receipts
- fault harness: 4 fault profiles, 3 throughput baselines, 2 soak suites,
  7 run receipts

This is the part of XTRAIN that most directly closes the gap to
`prime-diloco`: Psionic now has typed surfaces for join, leave, route
selection, low-bandwidth exchange, and repeated fault evidence.

### 3. Public Work, Data, And Artifact Truth

The next layer defines what work exists, what dataset pages are legal, and how
training artifacts move:

- `fixtures/training/public_work_assignment_contract_v1.json`
- `fixtures/training/public_dataset_authority_contract_v1.json`
- `fixtures/training/content_addressed_artifact_exchange_contract_v1.json`

Current fixture facts:

- public work assignment: 2 windows, 8 assignments, 8 assignment receipts,
  1 late-window refusal
- dataset authority: 8 dataset pages, 8 page proofs, 5 anti-replay receipts
- artifact exchange: 5 backends, 5 published artifacts, 5 fetch receipts

This layer is what keeps decentralized participation from collapsing into
mutable operator prose. It makes the public-data and public-artifact claims
machine-legible.

### 4. Miner, Validator, Consensus, And Fraud

The public execution and governance surfaces now include:

- `fixtures/training/public_miner_protocol_contract_v1.json`
- `fixtures/training/validator_challenge_scoring_contract_v1.json`
- `fixtures/training/multi_validator_consensus_contract_v1.json`
- `fixtures/training/fraud_quarantine_slashing_contract_v1.json`

Current fixture facts:

- miner protocol: 2 sessions, 2 local-step receipts, 2 delta-upload receipts,
  2 checkpoint-sync receipts, 1 refusal
- validator scoring: 2 replay rules, 2 score receipts, 1 refusal
- consensus: 2 votes, 1 promotion decision, 1 disagreement receipt
- fraud policy: 4 fraud signals, 2 quarantine decisions, 1 slashing decision,
  1 appeal window

This is the part of the stack that most directly closes the gap to `templar`:
Psionic now has Psionic-native equivalents for public miner and validator
surfaces, challenge scoring, checkpoint authority, and explicit fraud posture.

### 5. Reward, Settlement, And Public Operations

The economic and operator-facing surfaces now include:

- `fixtures/training/reward_ledger_contract_v1.json`
- `fixtures/training/settlement_publication_contract_v1.json`
- `fixtures/training/operator_bootstrap_package_contract_v1.json`
- `fixtures/training/public_run_explorer_contract_v1.json`
- `fixtures/training/public_testnet_readiness_contract_v1.json`

Current fixture facts:

- reward ledger: 5 contribution entries, 1 penalty entry, 4 final allocations
- settlement publication: 2 validator weight publications, 1 settlement
  record, 3 payout exports, 1 refusal
- operator packages: 2 packages, 4 preflight checks, 2 bootstrap kits
- run explorer: 6 panes, 4 score rows, 6 stale-data policies
- testnet readiness: 5 candidates, 8 compliance receipts, 5 graduation
  decisions

This layer makes the stack operable by humans rather than just complete as a
contract graph.

### 6. Retained Run Evidence

The end of the XTRAIN lane is not just more contracts; it is proof-run closure:

- `fixtures/training/curated_decentralized_run_contract_v1.json`
- `fixtures/training/open_public_decentralized_run_contract_v1.json`
- `fixtures/training/incentivized_decentralized_run_contract_v1.json`
- `docs/audits/2026-03-26-curated-decentralized-run-after-action-audit.md`
- `docs/audits/2026-03-26-open-public-miner-validator-run-audit.md`
- `docs/audits/2026-03-26-incentivized-decentralized-run-audit.md`
- `docs/audits/2026-03-26-full-decentralized-training-acceptance-audit.md`

Current fixture facts:

- curated run: 4 participants
- open public run: 4 participants and 4 events
- incentivized run: 3 rewarded participants

These are the retained evidence surfaces that close `XTRAIN-45` through
`XTRAIN-48`.

## What Was Re-Checked During This Audit

The following checks were run again while preparing this document.

### Static Inventory And Linkage Checks

A repo-local inventory pass over the XTRAIN decentralized lane found:

- 24 relevant training fixtures in `fixtures/training`
- zero missing linked reference docs
- zero missing checker scripts
- zero missing linked audit docs where audit docs were expected
- zero missing `docs/TRAIN_SYSTEM.md` or self-referenced fixture paths

I also re-checked the digest linkage across the final decentralized chain. The
current fixtures still bind in the intended order:

- network digest `37205ba6736d97b64f7156dde917b02d4bd4eeaa5bcfcfd96544c3bcdf338bda`
- identity set digest `e2423cfcb3e960bbf5790da0f0723f87bacc58003e1ad173d73b9ec63c41ffcc`
- registry digest `ebb5ca08893fe16f1a72c67f2fd99aefe44a95af7670bb31e875b2d77b88b7f4`
- mesh digest `c28470e11457d28464a6e5ae42543996b7048ec13e228b64e6394e233d1965fc`
- overlay digest `3f8ac57a330f37acfed56c582353643cd09072f6b1792472917165db979922e4`
- catch-up digest `3589f9e87675511227b73ec95e226ab99354f09cba35cc9c0d72c1608d9a10d8`
- artifact exchange digest `1d2153ccb088b9bac59de0e9a0f7534251b42799590bd97e7a5604c3b22f45ef`
- miner protocol digest `cd5d4fe71a5cd7ceaa43389127266ea38847a33631f8b7071eb44e77650fad65`
- validator scoring digest `99ad085b9a12258380dc4fb948c51c0cd86644a91afd6f5137d8a76a13830b25`
- consensus digest `ce2a463d0703a5cf5dc3d30efc9b9f8981c17e60bfa8a0d9b2f716912af0470f`
- fraud digest `a700e0a8d16213fb91848ffa3eabff857c88fafdf73098420d2ba26a530f1dda`
- reward ledger digest `42d7dc87c9dc416023ee2a96f838946533fb366780ec7449a1f9feed8002f103`
- settlement digest `ca13e653115f6001022f0573d7d8f09da1b8f778fe811301c9c712f26072ed65`
- operator package digest `5d0f0ecf67d414dc39cf5324a70684fa43e5291639e5c2536fff75ae376de5ce`
- explorer digest `6ef8e0ebf8a3475fb5ff8514dc44ca158419cb9db6c563b0b28163745c656828`
- testnet readiness digest `bf39625ba7482a9698c8c873be7c41c39c763d6a85d4c9956e421fa83d41c041`
- curated run digest `17310438fe3baacd455d34490558b33440ebf53927602fa02d7341e81d9af9f2`
- open public run digest `6aa7955fa12c5ee2865dadbf48c31940bd7ccbaa946b96fc7b39a11955be9775`
- incentivized run digest `1b4379c607674ef2198424370c3ad8a2940d1a9d2aea752b54219a3fff5e4e90`

### Cargo-Backed Validation Attempt

I attempted to rerun the shipped checker path with:

```bash
./scripts/check-decentralized-network-contract.sh
```

That checker still fails before reaching its fixture comparison phase because
the workspace compile currently stops at:

```text
error[E0599]: no method named `allocate` found for struct `PlatformSubmission`
--> crates/psionic-backend-cuda/src/lib.rs:876:37
```

The follow-on error is:

```text
error: could not compile `psionic-backend-cuda` (lib) due to 1 previous error
```

Because the decentralized checker path and the rest of the XTRAIN checkers all
depend on the same cargo workspace build, this blocker currently prevents
full cargo-backed revalidation of the XTRAIN generators in this checkout.

## How To Operate The XTRAIN System Today

### First Principle

Treat the current XTRAIN system as a contract-first and evidence-first
authority stack.

Do not treat it as already equivalent to a one-command production public
network daemon. The repo currently proves the typed surfaces, the retained
fixtures, the cross-linked references, and the retained run evidence. It does
not yet prove a cargo-clean, continuously executable public network in this
checkout.

### Read Order

An engineer picking this up should start here:

1. `docs/TRAIN_SYSTEM.md`
2. `docs/audits/2026-03-26-full-decentralized-training-acceptance-audit.md`
3. this audit
4. the focused reference docs for the specific surface being touched

### Operational Sequence

When reasoning about or extending the system, use this order:

1. Freeze network authority:
   `decentralized_network_contract`,
   `signed_node_identity_contract_set`,
   `public_network_registry_contract`
2. Freeze runtime behavior:
   `elastic_device_mesh_contract`,
   `wan_overlay_route_contract`,
   `live_checkpoint_catchup_contract`,
   `quantized_outer_sync_contract`,
   `internet_fault_harness_contract`
3. Freeze public work and data truth:
   `public_work_assignment_contract`,
   `public_dataset_authority_contract`,
   `content_addressed_artifact_exchange_contract`
4. Freeze execution and governance:
   `public_miner_protocol_contract`,
   `validator_challenge_scoring_contract`,
   `multi_validator_consensus_contract`,
   `fraud_quarantine_slashing_contract`
5. Freeze economics and operator surfaces:
   `reward_ledger_contract`,
   `settlement_publication_contract`,
   `operator_bootstrap_package_contract`,
   `public_run_explorer_contract`,
   `public_testnet_readiness_contract`
6. Freeze and audit run evidence:
   `curated_decentralized_run_contract`,
   `open_public_decentralized_run_contract`,
   `incentivized_decentralized_run_contract`

If a change breaks an earlier layer, treat everything above it as potentially
invalid until regenerated and rechecked.

### Canonical Generation Pattern

Each XTRAIN contract surface is currently operated through a matching binary in
`crates/psionic-train/src/bin` and, where provided, a checker script in
`scripts`.

The canonical generation pattern is:

```bash
cargo run -q -p psionic-train --bin decentralized_network_contract -- /tmp/decentralized_network_contract_v1.json
```

The canonical checker pattern is:

```bash
./scripts/check-decentralized-network-contract.sh
```

Equivalent binaries and checkers now exist across the decentralized lane,
including:

- `decentralized_network_contract`
- `signed_node_identity_contract_set`
- `public_network_registry_contract`
- `elastic_device_mesh_contract`
- `wan_overlay_route_contract`
- `live_checkpoint_catchup_contract`
- `quantized_outer_sync_contract`
- `public_work_assignment_contract`
- `public_dataset_authority_contract`
- `content_addressed_artifact_exchange_contract`
- `public_miner_protocol_contract`
- `validator_challenge_scoring_contract`
- `multi_validator_consensus_contract`
- `fraud_quarantine_slashing_contract`
- `reward_ledger_contract`
- `settlement_publication_contract`
- `operator_bootstrap_package_contract`
- `public_run_explorer_contract`
- `public_testnet_readiness_contract`
- `curated_decentralized_run_contract`
- `open_public_decentralized_run_contract`
- `incentivized_decentralized_run_contract`

### What To Do Right Now If You Need To Work On XTRAIN

Until the CUDA compile blocker is fixed:

- use the committed fixtures in `fixtures/training` as the source of truth
- use the focused reference docs in `docs` as the explanation layer
- use the retained audits in `docs/audits` as the proof-run evidence layer
- do not claim that the generator/checker path has been freshly revalidated by
  cargo on this checkout

After the CUDA blocker is fixed, the immediate first action should be:

```bash
./scripts/check-decentralized-network-contract.sh
./scripts/check-signed-node-identity-contract-set.sh
./scripts/check-public-network-registry-contract.sh
./scripts/check-elastic-device-mesh-contract.sh
./scripts/check-wan-overlay-route-contract.sh
./scripts/check-live-checkpoint-catchup-contract.sh
./scripts/check-public-dataset-authority-contract.sh
./scripts/check-content-addressed-artifact-exchange-contract.sh
./scripts/check-public-miner-protocol-contract.sh
./scripts/check-validator-challenge-scoring-contract.sh
./scripts/check-multi-validator-consensus-contract.sh
./scripts/check-fraud-quarantine-slashing-contract.sh
./scripts/check-reward-ledger-contract.sh
./scripts/check-settlement-publication-contract.sh
./scripts/check-public-testnet-readiness-contract.sh
./scripts/check-curated-decentralized-run-contract.sh
./scripts/check-open-public-decentralized-run-contract.sh
./scripts/check-incentivized-decentralized-run-contract.sh
```

That will convert the current "implemented and statically coherent" claim into
an "implemented and cargo-revalidated again on current main" claim.

## What The System Still Does Not Honestly Prove

Even with the roadmap closed, there are still important boundaries:

- this checkout does not currently prove a cargo-clean XTRAIN generator path
- the current implementation is still much more contract-heavy than daemonized
  public-network-heavy
- the incentivized run still reflects the current admitted reward-eligible set
  rather than outside canary payouts
- optional chain publication remains refused rather than implemented
- the retained run contracts are truthful evidence surfaces, but they are not
  a substitute for repeating the live network run on demand

## Recommended Next Actions

In order:

1. fix the unrelated `psionic-backend-cuda` compile regression at
   `crates/psionic-backend-cuda/src/lib.rs:876`
2. rerun the full checker sweep across the decentralized lane
3. record one follow-on audit confirming cargo-backed regeneration parity on
   current `main`
4. only after that, claim the XTRAIN system is both implemented and freshly
   executable from this checkout

## Verdict

The XTRAIN continuation is real and substantial. Psionic now owns a coherent
decentralized training system as a typed contract stack with retained evidence,
not just as a roadmap.

The main current weakness is not missing XTRAIN surface area. The main current
weakness is operational validation closure: the repo's shared cargo build is
broken in an unrelated CUDA path, so the decentralized generators cannot yet be
re-proven end to end from this checkout.
