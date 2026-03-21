# Tassadar Plugin-System And Turing-Completeness Audit

Date: 2026-03-20

## Purpose

This audit reviews the draft plugin-system spec at
`~/code/alpha/tassadar/plugin-system.md` against the current public Psionic
repo state and the current post-`TAS-186` Turing-completeness plan.

It answers three concrete questions:

1. What does the plugin-system spec actually require?
2. Which parts of that system already exist in Psionic/Tassadar, and which do
   not?
3. If the long-term goal is for the Turing-complete Tassadar path to implement
   that plugin system, does that change the shape of the post-`TAS-186`
   Turing-completeness push?

Companion audit:

- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`

## Source Set

Primary external design inputs:

- `~/code/alpha/tassadar/plugin-system.md`
- `~/code/alpha/tassadar/vm.md`
- `~/code/alpha/tassadar/tassadar-gap-analysis.md`

Primary repo surfaces reviewed:

- `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md`
- `docs/audits/2026-03-18-tassadar-post-tas-102-final-audit.md`
- `docs/audits/2026-03-19-tassadar-general-internal-compute-red-team-audit.md`
- `docs/audits/2026-03-19-tassadar-pre-closeout-universality-audit.md`
- `docs/audits/2026-03-19-tassadar-turing-completeness-closeout-audit.md`
- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`

Primary machine-readable repo surfaces reviewed:

- `fixtures/tassadar/reports/tassadar_tcm_v1_model.json`
- `fixtures/tassadar/reports/tassadar_tcm_v1_runtime_contract_report.json`
- `fixtures/tassadar/reports/tassadar_turing_completeness_closeout_audit_report.json`
- `fixtures/tassadar/reports/tassadar_world_mount_compatibility_report.json`
- `fixtures/tassadar/reports/tassadar_import_policy_matrix_report.json`
- `fixtures/tassadar/reports/tassadar_module_trust_isolation_report.json`
- `fixtures/tassadar/reports/tassadar_module_promotion_state_report.json`
- `fixtures/tassadar/reports/tassadar_internal_compute_package_manager_report.json`
- `fixtures/tassadar/reports/tassadar_internal_compute_package_route_policy_report.json`
- `fixtures/tassadar/reports/tassadar_internal_component_abi_report.json`
- `fixtures/tassadar/reports/tassadar_session_process_profile_report.json`
- `fixtures/tassadar/reports/tassadar_spill_tape_store_report.json`
- `fixtures/tassadar/reports/tassadar_virtual_fs_mount_profile_report.json`
- `fixtures/tassadar/reports/tassadar_simulator_effect_profile_report.json`
- `fixtures/tassadar/reports/tassadar_async_lifecycle_profile_report.json`
- `fixtures/tassadar/reports/tassadar_effectful_replay_audit_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_profile_publication_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_route_policy_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_portability_report.json`

Relevant public issue waves reviewed:

- `TAS-058` through `TAS-068`
- `TAS-101` through `TAS-111`
- `TAS-126`
- `TAS-129` through `TAS-136`
- `TAS-141`
- `TAS-150` through `TAS-156`
- `TAS-157` through `TAS-186`

## Executive Verdict

Current audit verdict:

- software-module, mount, effect, replay, and publication substrate below the
  plugin system: `implemented`
- plugin-specific Wasm manifest, packet ABI, runtime API, receipt family, and
  hot-swap contract: `partial`
- weighted controller that selects and sequences plugins while preserving
  owned-route control logic: `planned`
- plugin-system-complete route built on the canonical post-`TAS-156A` owned
  Transformer path: `planned`
- broad served/public plugin platform: `partial_outside_psionic`

The short answer to the user's question is:

- no, the plugin system should **not** become a prerequisite for the core
  post-`TAS-186` Turing-completeness rebase
- yes, it **does** change the shape of that push if the goal is not just
  bounded universality, but a productizable weighted software platform

The right move is:

1. finish `TAS-182` through `TAS-186`
2. land the post-article Turing-completeness rebase from the companion audit
3. make that rebase plugin-aware at the boundary level
4. then land a separate plugin-system tranche on top of it

That sequencing keeps the Turing-completeness claim honest while still steering
the system toward the plugin model.

## What The Plugin-System Spec Actually Requires

The draft plugin-system spec is not asking for "arbitrary tools" in a vague
sense. It is asking for one very specific architecture:

- weighted control logic in the model
- Wasm capability components as hot-swappable software artifacts
- host-owned bounded execution with explicit policy and receipts
- explicit separation between reasoning ownership and capability ownership

The key design law in the spec is:

- model owns decision logic
- runtime owns execution contract
- plugin owns black-box capability implementation

That means the plugin system is not merely:

- more broad internal compute
- more module packaging
- more effect profiles
- or more Turing-completeness evidence

It is a software-platform layer above those substrates, with one hard extra
requirement:

the host must not quietly become the owner of orchestration logic.

That requirement is exactly why the plugin system intersects with the
post-`TAS-186` ownership and route-minimality work.

## What Already Exists In Psionic

Large parts of the plugin-system substrate are already present in bounded form.

### 1. Module And Package Substrate: `implemented`

These old TAS lanes already give Psionic most of the software-component
substrate a plugin system would need:

- `TAS-058`:
  deterministic module linker and dependency-graph semantics
- `TAS-059`:
  installed-module evidence bundles and decompilation receipts
- `TAS-060`:
  module catalog and reusable primitive library surfaces
- `TAS-061`:
  overlap resolution and mount-aware import selection
- `TAS-134`:
  bounded internal-compute package manager and routeable named packages
- `TAS-135`:
  cross-profile link compatibility and downgrade/refusal behavior
- `TAS-136`:
  installed-process lifecycle with migration and rollback receipts

This means the repo already understands:

- digest-bound component identity
- dependency graphs
- package resolution
- compatibility and downgrade planning
- lifecycle, rollback, and evidence posture

Those are real plugin-adjacent building blocks.

### 2. Capability And Policy Substrate: `implemented`

The plugin spec's world-mount and capability model also maps strongly onto
already-landed Psionic surfaces:

- `TAS-062`:
  typed host-call and import policy matrix
- `TAS-063`:
  trust-tier isolation and privilege-escalation refusal
- `TAS-064`:
  promotion lifecycle, quarantine, and revocation posture
- `TAS-066`:
  self-installation policy and rollback gate
- `TAS-068`:
  world-mount compatibility and mount-time negotiation
- `TAS-101` and `TAS-102`:
  portability envelopes, publication posture, and route policy
- `TAS-110` and `TAS-111`:
  subset promotion and backend/toolchain portability matrices

This means the repo already has real machinery for:

- explicit mounted capability posture
- trust and publication boundaries
- route policy
- portability gating
- profile-specific suppression
- quarantine and revocation

That is much closer to the plugin spec than a greenfield system would be.

### 3. Effect, Session, And Replay Substrate: `implemented`

The plugin spec requires:

- bounded external interaction
- deterministic or declared replay classes
- explicit cancellation and timeout posture
- durable state only in host-backed stores

Psionic already has direct analogs:

- `TAS-126`:
  bounded session-process profile with explicit refusal on open-ended external
  streams
- `TAS-129`:
  deterministic virtual filesystem and artifact-mount effect profile
- `TAS-130`:
  deterministic clock, randomness, and simulator-backed network profiles
- `TAS-131`:
  async call, interrupt, retry, and cancellation semantics
- `TAS-132`:
  effectful replay, challenge, and audit receipts
- `TAS-127`, `TAS-136`, and the `TCM.v1` closeout:
  process objects, spill/tape continuation, and persistent lifecycle receipts

This means the repo already understands:

- mounted artifact state
- seeded external effect simulation
- bounded async lifecycle
- replay classes and challenge receipts
- durable host-backed continuation state

Again, those are real plugin-system prerequisites.

### 4. Component ABI Substrate: `implemented`

`TAS-133` already lands a bounded internal component-model ABI lane.

That matters because the plugin spec wants:

- versioned Wasm modules or linked Wasm packages
- packet-style entrypoints
- declared exports
- explicit refusal on unsupported interface shapes

The current component ABI is not yet the plugin ABI, but it proves the repo
already has one bounded interface-contract lane for modular software-like
computation.

## What Does Not Yet Exist

Even with all that substrate, the actual plugin system described in
`plugin-system.md` is not yet implemented.

The missing pieces are specific.

### 1. Canonical Plugin Manifest Contract: `planned`

The repo does not yet have one dedicated plugin manifest that freezes:

- `plugin_id`
- `plugin_version`
- `artifact_digest`
- declared export list
- packet ABI version
- input/output schema ids
- per-plugin limits
- capability requirements
- replay class
- trust tier
- evidence settings
- hot-swap compatibility posture

Current module and package reports are close, but they are not yet this exact
plugin contract.

### 2. Packet ABI And Rust-First Plugin PDK: `planned`

The spec explicitly wants:

- one narrow packet ABI
- one input packet
- one output packet or typed refusal
- one explicit error channel
- one Rust-first guest authoring path

The repo has component and module ABI surfaces, but not one dedicated
plugin-packet ABI plus one Rust-first guest PDK for bounded Wasm plugins.

### 3. Host-Owned Plugin Runtime API: `planned`

The spec wants one plugin runtime with:

- engine abstraction
- artifact loading
- export invocation
- capability mounting
- packet marshalling
- timeout and memory ceilings
- per-plugin pools
- cancellation handles
- usage and evidence capture

Psionic has Wasm runtimes, sandboxes, and effect profiles, but not one unified
plugin runtime API that exposes plugins as named hot-swappable Wasm capability
components.

### 4. Plugin-Specific Receipt Family: `planned`

The plugin spec requires every invocation receipt to carry:

- plugin id
- plugin version
- artifact digest
- export name
- packet ABI version
- input/output digests
- mount-envelope identity
- capability-envelope identity
- backend id
- refusal/failure class

Current receipt families cover execution truth, effects, replay, modules, and
processes. They do not yet freeze this exact plugin invocation identity.

### 5. Weighted Plugin Control Trace: `planned`

This is the biggest missing piece.

The spec requires the model to own:

- plugin selection
- export selection
- argument generation
- multi-step sequencing
- retries
- refusal
- stop conditions

The repo does **not** yet have one canonical weighted plugin control trace on
the post-`TAS-156A` owned Transformer route.

That is a deeper requirement than ordinary route policy or module resolution.
It is a weights-ownership requirement.

### 6. Refusal-Aware Return Path Into The Model Loop: `planned`

The plugin system is incomplete without a canonical path from:

- model emits plugin-use intent
- runtime checks admissibility and runs bounded plugin
- runtime emits receipt
- result packet returns to the model loop
- weights decide continue/retry/stop

The repo has route and process profiles, but not this exact weighted
continuation loop for plugins.

### 7. Hot-Swap Compatibility Contract: `planned`

The plugin spec explicitly wants hot-swappable Wasm components with stable
identity and compatibility checks.

Current package, module, and lifecycle systems provide parts of this, but the
repo does not yet have one plugin-specific hot-swap contract over:

- stable plugin id
- version compatibility
- ABI compatibility
- schema compatibility
- trust posture continuity

### 8. Plugin-Specific Benchmarks And Conformance Harness: `planned`

The plugin spec requires:

- packet roundtrip tests
- malformed-packet refusal tests
- capability denial tests
- timeout tests
- memory-limit tests
- digest mismatch refusal tests
- replay posture checks
- receipt integrity checks
- mount-envelope compatibility checks
- cold/warm/pool/cancel overhead benchmarks

That conformance surface does not yet exist as one plugin-system bar.

## The Most Important Boundary Question

The central architectural question is not "can Tassadar host Wasm modules?"

It already can, in bounded and policy-mediated ways.

The real question is:

Can the canonical owned Transformer route become the owner of plugin
orchestration logic without the host quietly becoming the real planner?

That is why the plugin system is not just another module or package issue.
It intersects directly with:

- `TAS-184`: clean-room weight causality and interpreter ownership
- `TAS-184A`: cache and activation-state discipline
- `TAS-185A`: route minimality and publication verdict
- `TAS-186`: final article-equivalence checker and audit

Until those are closed, any weighted plugin-controller claim would be weak,
because the host runtime could still be doing too much orchestration work.

## Does The Plugin System Change The Shape Of The Turing-Completeness Push?

Yes, but only in a constrained way.

### What Should Not Change

The plugin system should **not** change the minimal post-`TAS-186`
Turing-completeness objective from the companion audit:

- rebase the old `TCM.v1` closeout onto the canonical owned route
- prove continuation ownership on that route
- rebind universal-machine proofs and witness suites to that route
- issue a new canonical-route universal-substrate gate
- refresh portability and verdict split on that route

That is still the right minimal push.

If the repo skips that and jumps straight to plugins, it will blur:

- pure compute universality
- bounded operator continuation
- weighted control ownership
- capability-mediated software execution

That would weaken the claim discipline.

### What Should Change

If the long-term target includes the plugin system, the post-`TAS-186`
Turing-completeness push should become plugin-aware in four concrete ways.

#### 1. Do Not Freeze A Pure-Compute-Only Terminal Shape

The rebased canonical-route universality contract should not be written as if
the only future consumer is:

- witness programs
- pure interpreters
- and resumable kernels

It should leave room for:

- packet-mediated capability calls
- mount-envelope identities
- plugin-specific receipts
- explicit software-like capability components

That does **not** mean making plugins a prerequisite for Turing completeness.
It means not hard-coding a terminal contract that would need to be rewritten as
soon as plugins arrive.

#### 2. The Bridge Contract Should Reserve A Plugin Boundary

The post-`TAS-186` bridge tranche should explicitly classify plugin execution
as:

- outside the model's pure compute core
- inside the runtime's bounded execution contract
- admissible only under explicit capability envelopes

That lets later plugin work attach cleanly without mutating `TCM.v1` into a
vague "everything" substrate.

The right pattern is:

- keep `TCM.v1` as the bounded compute substrate
- add a separate named plugin execution contract above it
- bind the two through policy and receipts, not through hand-waving

#### 3. Continuation Ownership Must Become State-Class-Aware

The plugin spec makes a strong distinction between:

- packet-local state
- instance-local ephemeral state
- host-backed durable state

The canonical-route continuation-ownership audit from the companion
post-`TAS-186` plan should be expanded to preserve exactly that split.

Otherwise the repo will later struggle to say:

- what canonical workflow truth may live in host-backed stores
- what may only be cache-like ephemeral plugin state
- what the weighted controller is allowed to rely on across resumes

That state split is useful even before plugins exist.

#### 4. The Verdict Split Should Keep Plugin Capability Separate

The final rebased theory/operator/served split should not silently imply:

- operator universality means weighted plugin capability
- plugin capability means served plugin platform
- or served plugin platform means broad public universality

The plugin system adds one more policy-bearing capability layer. It should be
kept separate in the same way the repo already keeps:

- theory vs operator vs served
- profile promotion vs publication
- runtime success vs accepted outcomes

## What Should Be In The Immediate Post-`TAS-186` Push

The immediate post-`TAS-186` push should contain only the plugin-aware changes
that protect the architecture from rework.

Those are:

- preserve a clean separation between pure compute substrate and capability
  execution layer
- make the bridge contract reserve plugin-boundary identity fields
- make the continuation-ownership audit explicit about packet-local,
  ephemeral-instance, and durable-host state
- keep the rebased verdict split from over-reading future plugin capability

That is enough for now.

It is **not** the right moment to make the immediate push also close:

- full plugin manifest schema
- full packet ABI
- plugin PDK
- full plugin runtime API
- pool/cancel plugin runtime benchmarks
- weighted plugin control trace
- plugin-specific publication and promotion gates

Those belong in the next tranche.

## What The Plugin Tranche Should Look Like

After the post-`TAS-186` rebase lands, the plugin work should be its own issue
wave.

Recommended tranche shape:

### 1. Plugin Charter And Claim Boundary

Freeze:

- internal-only posture
- plugin non-goals
- policy and publication boundary
- explicit relation to `TCM.v1` and post-article owned-route truth

### 2. Plugin Manifest And Identity Contract

Land:

- plugin manifest schema
- canonical invocation identity
- export declaration rules
- replay class
- trust tier
- evidence configuration

### 3. Packet ABI And Rust-First PDK

Land:

- packet ABI
- schema ids
- guest refusal type
- narrow host import binding surface
- conformance harness

### 4. Host-Owned Plugin Runtime

Land:

- runtime engine abstraction
- artifact load and digest verification
- invoke API
- pool and queue semantics
- timeout and memory enforcement
- cancellation handles

### 5. Plugin Receipts And Replay

Land:

- invocation receipt schema
- plugin replay classes
- route-integrated evidence bundle
- replay and refusal checks

### 6. World-Mount Envelope Compiler For Plugins

Land:

- plugin admissibility check
- capability namespace mount compilation
- route-to-envelope binding
- explicit denial behavior

### 7. Weighted Controller Integration

Land:

- structured weighted plugin control trace
- packet encoding from model outputs
- result path back into the model loop
- refusal-aware continuation

This step should only start after the owned-route mechanistic bar is strong
enough that the repo can honestly say the host is not the real planner.

### 8. Plugin Promotion And Publication

Land:

- plugin trust tiers
- plugin benchmark bars
- plugin publication gates
- validator and accepted-outcome policy hooks
- explicit operator-only vs served posture

## Recommended Dependency Order

The dependency order should be:

1. finish `TAS-182` through `TAS-186`
2. land the post-article Turing-completeness bridge tranche from the companion
   audit
3. land plugin charter, manifest, packet ABI, runtime API, and receipts
4. land world-mount envelope compiler and policy integration for plugins
5. land weighted plugin-controller integration
6. land plugin promotion/publication policy
7. only then consider any stronger "weighted software platform" closeout

That order matters because it keeps the repo from using a not-yet-audited
weighted controller as evidence for a plugin system whose core control-ownership
question is still unresolved.

## What The Repo Can Honestly Say Today

The strongest honest statement today is:

Psionic/Tassadar already has most of the bounded software-platform substrate a
plugin system would need:

- module and package identity
- module evidence and promotion lifecycle
- import and capability policy
- world-mount negotiation
- bounded effect profiles
- replay and challenge receipts
- session-process and persistent process objects
- profile publication and route policy

But the repo does **not** yet have:

- a canonical plugin manifest
- a packet ABI and Rust plugin PDK
- a host-owned plugin runtime API
- plugin-specific invocation receipts
- or a weighted controller that owns plugin selection and sequencing on the
  canonical owned Transformer route

So the plugin system is not blocked on basic substrate. It is blocked on
unifying that substrate under one plugin contract and then proving weighted
control ownership.

## Final Judgment

The plugin-system spec does change the shape of the Turing-completeness push,
but not by replacing it.

It changes the shape by forcing one architectural discipline:

the post-`TAS-186` universality rebase must be written so the canonical owned
Transformer route can later become the weighted controller for a bounded
software-like Wasm capability platform, without turning `TCM.v1` into a vague
everything-machine and without letting the host quietly steal orchestration.

So the right answer is:

- finish the core post-`TAS-186` Turing-completeness rebase first
- make that rebase plugin-aware at the boundary level
- then build the plugin system as the next explicit tranche on top of it

That is the cleanest way to get both:

- one honest bounded Turing-completeness story
- and one honest weighted plugin-system story

without collapsing them into one over-broad claim.
