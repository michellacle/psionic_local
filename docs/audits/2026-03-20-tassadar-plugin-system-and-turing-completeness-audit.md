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

## Two Non-Negotiable Ownership Laws

The plugin tranche should freeze two blunt laws early, because they are the
easiest place for the repo to drift into host-orchestrated tool behavior while
still using weighted-plugin language.

### 1. State Ownership Law

Plugin-related state should stay in explicit classes:

- packet-local state:
  request and response state carried in the invocation itself
- instance-local ephemeral state:
  warmed tables, memoized helpers, and caches that may improve performance but
  may never become canonical workflow truth
- host-backed durable state:
  explicit mounted stores, artifact stores, checkpoints, queues, and worklists
  under receipts and replay posture
- weights-owned control state:
  the model's own plugin-selection, branching, retry, and stop logic

The hard refusal is:

no durable workflow truth may be smuggled into plugin instance memory, hidden
runtime state, or cache-like host layers and still be described as weighted
software behavior.

### 2. Control Ownership Law

The plugin system also needs one blunt rule:

host may execute capability, but host may not decide workflow.

That means the host may:

- resolve manifests and verify digests
- mount capabilities
- enforce bounds
- run Wasm exports
- emit receipts and typed failure/refusal outcomes

But the host may not:

- choose which plugin to call
- choose which export to call
- decide retries
- decide stop conditions
- or branch multi-step workflow logic on behalf of the model

Without this law, the repo would drift into a tool runtime that only sounds
weighted.

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
Turing-completeness push should become plugin-aware in five concrete ways.

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

#### 4. Freeze The Workflow-Ownership Rule

The post-`TAS-186` bridge should carry one blunt rule forward:

host may execute capability, but host may not decide workflow.

That rule belongs in the bridge before the plugin tranche starts, because later
plugin work will otherwise be forced to infer it from indirect article or
continuation artifacts.

#### 5. The Verdict Split Should Keep Plugin Capability Separate

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
- freeze the workflow-ownership rule before any plugin-controller claim is made
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

## Proposed GitHub Issue Roadmap

Suggested numbering below assumes the post-`TAS-186` Turing-completeness
bridge consumes `TAS-187` through `TAS-196`. If the tracker advances first,
preserve the dependency order and titles, not the exact numerals.

### Suggested `TAS-197`: Freeze Plugin Charter, Claim Boundary, And Ownership Laws

Suggested GitHub title:

`Tassadar: freeze weighted plugin charter and ownership boundary`

Summary:

Freeze the plugin system as a bounded software-capability layer above the
rebased compute substrate, with explicit state ownership and control ownership
rules.

Description:

- define plugin non-goals and internal-only/publication posture
- bind the plugin system explicitly to the post-`TAS-186` owned-route truth
  without mutating `TCM.v1`
- freeze the state-class split across packet-local, instance-local ephemeral,
  host-backed durable, and weights-owned control state
- freeze the rule that host may execute capability but may not decide workflow

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `fixtures/tassadar/reports/tassadar_world_mount_compatibility_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_profile_publication_report.json`

### Suggested `TAS-198`: Canonical Plugin Manifest, Identity, And Hot-Swap Contract

Suggested GitHub title:

`Tassadar: add canonical plugin manifest and hot-swap identity contract`

Summary:

Land the canonical plugin contract so plugins become real named software
artifacts rather than loosely packaged modules.

Description:

- define `plugin_id`, `plugin_version`, `artifact_digest`, declared exports,
  packet ABI version, schema ids, limits, trust tier, replay class, and
  evidence settings
- define canonical invocation identity fields
- define compatibility and hot-swap rules across versions, ABI shapes, and
  trust posture
- keep multi-module packaging explicit when one plugin is a linked bundle

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `fixtures/tassadar/reports/tassadar_module_trust_isolation_report.json`
- `fixtures/tassadar/reports/tassadar_module_promotion_state_report.json`
- `fixtures/tassadar/reports/tassadar_internal_compute_package_manager_report.json`
- `fixtures/tassadar/reports/tassadar_internal_compute_package_route_policy_report.json`

### Suggested `TAS-199`: Packet ABI And Rust-First Plugin PDK

Suggested GitHub title:

`Tassadar: add packet ABI and Rust-first plugin PDK`

Summary:

Define the narrow invocation ABI and guest authoring surface that every bounded
plugin must use.

Description:

- define one input packet, one output packet or typed refusal, and one explicit
  error channel
- define schema ids, codec ids, payload bytes, and metadata envelope shape
- define the Rust-first guest refusal type and narrow host import surface
- keep the ABI narrow enough that conformance is easy to test and audit

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `fixtures/tassadar/reports/tassadar_internal_component_abi_report.json`
- `docs/audits/2026-03-18-tassadar-post-tas-102-final-audit.md`

### Suggested `TAS-200`: Host-Owned Plugin Runtime API And Engine Abstraction

Suggested GitHub title:

`Tassadar: add host-owned plugin runtime API and engine abstraction`

Summary:

Build the bounded runtime substrate that loads, mounts, executes, and cancels
plugin invocations without leaking backend-specific behavior into the top-level
contract.

Description:

- define artifact loading and digest verification
- define the engine abstraction for instantiate, invoke, mount, cancel, and
  usage collection
- define pool, queue, timeout, memory, concurrency, and optional fuel bounds
- keep backend-specific details below the host-owned runtime API

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `fixtures/tassadar/reports/tassadar_import_policy_matrix_report.json`
- `fixtures/tassadar/reports/tassadar_async_lifecycle_profile_report.json`
- `fixtures/tassadar/reports/tassadar_simulator_effect_profile_report.json`

### Suggested `TAS-201`: Plugin Invocation Receipts And Replay Classes

Suggested GitHub title:

`Tassadar: add plugin invocation receipts and replay classes`

Summary:

Freeze the receipt identity and replay posture needed to make plugin execution
auditable and challengeable.

Description:

- define invocation id, plugin id, version, artifact digest, export name,
  packet ABI version, digests, envelope ids, backend id, refusal/failure
  class, and resource summary
- define deterministic and snapshot-based replay classes
- connect plugin receipts to route-integrated evidence and challenge receipts

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `fixtures/tassadar/reports/tassadar_effectful_replay_audit_report.json`
- `fixtures/tassadar/reports/tassadar_installed_module_evidence_report.json`
- `fixtures/tassadar/reports/tassadar_module_promotion_state_report.json`

### Suggested `TAS-202`: World-Mount Envelope Compiler And Plugin Admissibility Contract

Suggested GitHub title:

`Tassadar: compile world-mount envelopes for plugins`

Summary:

Translate route and mount policy into explicit runtime-admissible plugin
envelopes so capability mediation is no longer ad hoc.

Description:

- define plugin admissibility checks
- compile capability namespace grants, network rules, and mount posture into
  runtime envelopes
- bind route policy to plugin version constraints and trust posture
- keep explicit denial behavior for unsupported or disallowed invocations

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `fixtures/tassadar/reports/tassadar_world_mount_compatibility_report.json`
- `fixtures/tassadar/reports/tassadar_import_policy_matrix_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_route_policy_report.json`

### Suggested `TAS-203`: Plugin Conformance Sandbox And Benchmark Harness

Suggested GitHub title:

`Tassadar: add plugin conformance sandbox and benchmark harness`

Summary:

Prove that plugins are real software components with clean identity, bounded
execution, and refusal behavior before the repo claims model-owned plugin
planning.

Description:

- run static conformance harnesses and host-scripted traces only, not
  model-owned sequencing
- cover roundtrip success, malformed packet refusal, capability denial,
  timeout, memory-limit, packet-size, digest-mismatch, replay, and hot-swap
  compatibility rows
- benchmark cold, warm, pooled, queued, and cancelled execution paths
- keep receipt integrity and envelope compatibility explicit

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `fixtures/tassadar/reports/tassadar_async_lifecycle_profile_report.json`
- `fixtures/tassadar/reports/tassadar_effectful_replay_audit_report.json`
- `fixtures/tassadar/reports/tassadar_module_trust_isolation_report.json`
- `fixtures/tassadar/reports/tassadar_world_mount_compatibility_report.json`

### Suggested `TAS-204`: Weighted Plugin Controller Trace And Refusal-Aware Model Loop

Suggested GitHub title:

`Tassadar: add weighted plugin controller trace and refusal-aware model loop`

Summary:

Only after the substrate is proven should the repo claim that the model owns
plugin selection, sequencing, retries, and stop conditions.

Description:

- define the structured weighted plugin control trace
- encode packet arguments from model outputs under the canonical packet ABI
- return result packets and typed refusals back into the model loop
- prove that the host validates and executes but does not become the planner
- add negative rows for hidden host-side sequencing or retry logic

Supporting material:

- `~/code/alpha/tassadar/plugin-system.md`
- `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md`
- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json`

### Suggested `TAS-205`: Plugin Promotion, Publication, And Trust-Tier Gate

Suggested GitHub title:

`Tassadar: add plugin promotion and publication gate`

Summary:

Add the policy and trust machinery needed to promote bounded plugins without
accidentally widening them into a public arbitrary-software claim.

Description:

- define plugin benchmark bars and trust-tier gates
- define operator-only versus served/public posture
- bind validator and accepted-outcome hooks where required
- keep promotion, quarantine, revocation, and publication refusal explicit

Supporting material:

- `fixtures/tassadar/reports/tassadar_module_promotion_state_report.json`
- `fixtures/tassadar/reports/tassadar_module_trust_isolation_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_profile_publication_report.json`
- `fixtures/tassadar/reports/tassadar_broad_internal_compute_route_policy_report.json`

### Suggested `TAS-206`: Publish The Bounded Weighted Plugin Platform Closeout Audit

Suggested GitHub title:

`Tassadar: publish bounded weighted plugin platform closeout audit`

Summary:

Publish the first honest plugin-platform closeout only after manifest, ABI,
runtime, receipts, mount compiler, conformance sandbox, weighted control, and
promotion/publication bars are all green.

Description:

- state clearly what the plugin system does and does not claim
- preserve the separation from the bounded Turing-completeness closeout
- keep operator and served/plugin publication posture explicit
- refuse any implication of arbitrary public Wasm or arbitrary public tool use

Supporting material:

- `docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md`
- `docs/audits/2026-03-20-tassadar-post-article-turing-completeness-audit.md`
- `~/code/alpha/tassadar/plugin-system.md`

## Recommended Dependency Order

The dependency order should be:

1. finish `TAS-182` through `TAS-186`
2. land the post-article Turing-completeness bridge tranche from the companion
   audit
3. land plugin charter, manifest, packet ABI, runtime API, receipts, and
   replay classes
4. land the world-mount envelope compiler and plugin admissibility contract
5. land the plugin conformance sandbox with static harnesses and scripted
   traces, not model-owned sequencing
6. only then land weighted plugin-controller integration
7. land plugin promotion/publication policy
8. only then consider any stronger "weighted software platform" closeout

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
