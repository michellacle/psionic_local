# Psionic Burn Port Audit

Date: 2026-03-25

## Scope

This audit answers one question:

- what additional Burn ideas are worth porting into Psionic now

It does not assume Burn is a better base framework than Psionic. Psionic
already owns larger public surfaces for routing, receipts, capability
publication, model interchange, distributed execution, and execution-truth
contracts.

The job here is narrower:

- find Burn subsystems that fill a real Psionic gap
- reject subsystems that duplicate or weaken existing Psionic architecture
- define the benchmark and refusal posture before calling any port worthwhile

Reviewed Burn surfaces:

- `README.md`
- `crates/burn-store`
- `crates/burn-backend-tests`
- `crates/burn-tensor-testgen`
- `crates/burn-train`
- `crates/burn-fusion`
- `crates/burn-router`
- `crates/burn-remote`
- `crates/burn-dataset`
- `crates/burn-core`

Reviewed Psionic comparison surfaces:

- `docs/ARCHITECTURE.md`
- `docs/FRAMEWORK_CORE_ACCEPTANCE_MATRIX.md`
- `docs/TRAIN_SYSTEM.md`
- `crates/psionic-train/src/model_io.rs`
- `crates/psionic-nn/src/lib.rs`
- `crates/psionic-router`
- `crates/psionic-ir`

## Summary

Psionic should not port Burn wholesale.

Psionic should port or adapt four narrow ideas:

1. partial checkpoint load and key-remap tooling
2. a reusable backend conformance harness
3. asynchronous checkpoint writeback for long-running train loops
4. a smaller train-loop metric sink split for local telemetry

Psionic should treat Burn fusion heuristics and module-record ergonomics as
reference material, not as direct subsystem imports.

Psionic should not port Burn autodiff, router, remote, or dataset subsystems as
first-class architecture. Psionic already owns stronger public surfaces there.

## Pull In Now

### 1. Partial checkpoint load, key remap, and lazy tensor materialization

Burn evidence:

- `crates/burn-store/src/filter.rs`
- `crates/burn-store/src/keyremapper.rs`
- `crates/burn-store/src/pytorch/lazy_data.rs`

Problem:

Psionic already has serious model IO in `psionic-train`, but large checkpoint
migration still needs better selective import, key translation, and deferred
materialization tools.

Hypothesis:

Psionic should add a bounded checkpoint migration layer that can:

- load only named tensor subsets
- rewrite parameter keys through explicit maps
- defer tensor byte materialization until the consumer actually needs the data

Surface:

- `psionic-train`
- `psionic-array-io`
- `psionic-models`

Claim class:

Model-interchange and migration ergonomics.

This is not a new truth plane. This is not a new checkpoint semantics model.

Benchmark / tests:

- unit tests for include and exclude filters
- unit tests for one-to-one and many-to-one key remap refusal
- import parity tests against full eager load
- peak-memory benchmark during large checkpoint translation
- refusal-path tests for unsupported tensor encodings, opaque pickle-only
  payloads, and incompatible shape rewrites

Exit criteria:

Psionic can import a large external checkpoint into a Psionic model surface
without eagerly materializing every tensor and without requiring ad hoc rename
scripts.

Why this matters:

This closes a real workflow gap. Burn’s store layer is the best additional code
source in the repo for that exact problem.

### 2. Backend conformance harness across CPU, CUDA, Metal, and MLX-backed lanes

Burn evidence:

- `crates/burn-backend-tests/README.md`
- `crates/burn-backend-tests/tests/common/backend.rs`
- `crates/burn-tensor-testgen/src/lib.rs`

Problem:

Psionic has strong backend-specific tests and strong acceptance docs, but it
does not yet appear to own a single reusable backend conformance harness that
can execute the same operator and dtype suites across every backend lane with
consistent failure semantics.

Hypothesis:

Psionic should add a shared backend conformance package so backend parity stops
depending on hand-copied tests and backend-local coverage drift.

Surface:

- `psionic-array`
- `psionic-runtime`
- `psionic-backend-cpu`
- `psionic-backend-cuda`
- `psionic-backend-metal`
- `psionic-mlx-compat`
- `psionic-mlx-serve`

Claim class:

Validation infrastructure.

This is not a user-facing feature. It is a backend truth-discipline upgrade.

Benchmark / tests:

- shared operator correctness suites across all supported dtypes
- deterministic seed and multi-thread behavior tests
- backend-specific refusal suites for unsupported ops or layouts
- backend capability-matrix publication tests
- optional quantization and fused-kernel suites gated behind declared support

Exit criteria:

Every backend lane can run the same canonical conformance family and emit a
clear pass, fail, or unsupported result without custom per-backend test
scaffolding.

Why this matters:

This is the cleanest Burn pattern to port. It improves Psionic quality without
dragging in Burn’s higher-level architecture.

### 3. Async checkpoint writeback for train loops

Burn evidence:

- `crates/burn-train/src/checkpoint/async_checkpoint.rs`

Problem:

Checkpoint serialization should not stall a train loop longer than necessary,
especially once checkpoints include larger optimizer state, richer receipts, or
sharded payloads.

Hypothesis:

Psionic should support bounded asynchronous checkpoint emission where the train
step hands an immutable checkpoint payload to a writer worker and continues once
the handoff is sealed.

Surface:

- `psionic-train`
- `psionic-runtime`

Claim class:

Training systems throughput.

This is not a weaker replacement for receipt-bound checkpoint lineage. It is a
write-path execution improvement.

Benchmark / tests:

- replay-equivalence tests between sync and async checkpoint restore
- interrupted-write refusal tests
- checksum and atomic-finalization tests
- train-step stall benchmark with sync versus async writeback
- bounded queue pressure tests under repeated checkpoint cadence

Exit criteria:

Psionic can emit checkpoints asynchronously without weakening restore
correctness, digest stability, or lineage validation.

Why this matters:

Burn’s implementation pattern is narrow and useful. Psionic already has richer
checkpoint meaning than Burn. The value here is operational writeback behavior.

### 4. Local metric-sink split for train-loop telemetry

Burn evidence:

- `crates/burn-train/src/learner/base.rs`
- `crates/burn-train/src/logger/metric.rs`

Problem:

Psionic has machine-legible receipts and benchmark artifacts, but train-loop
local telemetry can still become bespoke and uneven when each package writes
its own narrow progress reporting.

Hypothesis:

Psionic should add a small metric-sink layer for train loops that can fan out
to:

- structured logs
- local progress output
- JSONL metric files
- receipt pre-aggregation inputs

Surface:

- `psionic-train`
- `psionic-observe`

Claim class:

Developer ergonomics and instrumentation consistency.

Benchmark / tests:

- unit tests for sink fanout and flush ordering
- deterministic JSONL emission tests
- no-duplication tests between local metric sinks and final receipts
- refusal tests for invalid metric schema or non-monotonic step bindings

Exit criteria:

Training packages can emit consistent local metrics without inventing ad hoc
logging structures, and those metrics do not blur the line between telemetry
and final benchmark truth.

Why this matters:

This is lower priority than the first three items, but it is still a useful
Burn pattern to port narrowly.

## Use As Reference Only

### 5. Fusion search heuristics

Burn evidence:

- `crates/burn-fusion/src/search/merging.rs`

Problem:

Burn has practical local merge heuristics for fused execution regions.

Recommendation:

Do not port Burn fusion as a subsystem.

Psionic already owns the correct architectural surfaces for graph lowering,
schedule formation, alias-aware planning, and backend realization in:

- `psionic-ir`
- `psionic-compiler`

The useful part is narrower:

- inspect Burn’s bounded region-merging heuristics as implementation reference
  when current Psionic compiler benchmarks show a real fusion gap

Claim class:

Compiler implementation reference.

Benchmark / tests:

- compile-plan identity tests
- alias-safety tests
- numerics equivalence tests
- benchmarked kernel-count and wallclock changes on representative graphs

Exit criteria:

Only port a specific heuristic if it improves Psionic compiler output on a
measured benchmark without changing semantics.

### 6. Module and record ergonomics

Burn evidence:

- `crates/burn-core/src/module`
- `crates/burn-core/src/record`

Problem:

Burn has ergonomic module-record APIs and some clean state serialization
patterns.

Recommendation:

Do not port the Burn module system.

Psionic already owns explicit module, parameter, buffer, and state-tree
surfaces in `psionic-nn`. The only worthwhile imports here are narrow helpers:

- derive ergonomics if they reduce boilerplate
- state-dict remap helpers if they strengthen checkpoint migration

Claim class:

Ergonomic refinement only.

Benchmark / tests:

- compile-time derive tests
- state-tree roundtrip tests
- migration compatibility tests

Exit criteria:

Any adopted helper must reduce boilerplate without changing the existing
Psionic state semantics.

## Do Not Port Wholesale

### 7. Burn autodiff backend decorator

Reason:

Psionic already owns a larger transform and autodiff substrate in `psionic-ir`
and its adjacent compiler surfaces.

Porting Burn autodiff would create overlap, not closure.

### 8. Burn router backend

Reason:

Burn routes tensor operations across backends. Psionic router work is about
model, provider, capability, and route selection contracts.

Those are different abstractions. A direct Burn router port would create a
second routing vocabulary inside Psionic.

### 9. Burn remote backend

Reason:

Psionic already has explicit public surfaces for remote and distributed
execution:

- `psionic-cluster`
- `psionic-distributed`
- `psionic-net`
- `psionic-serve`
- `psionic-provider`

Remote tensor forwarding is not the missing architectural layer.

### 10. Burn dataset subsystem

Reason:

Psionic already treats data and evaluation as explicit, versioned, benchmark-
bound surfaces. Burn’s dataset APIs may contain useful utilities, but a direct
dataset subsystem port would not solve a current Psionic gap.

Borrow transform utilities if they prove useful. Do not import the subsystem.

## Recommended Order

1. Port the checkpoint filter, remap, and lazy materialization ideas into
   `psionic-train` and `psionic-array-io`.
2. Build a shared backend conformance harness and make backend promotion depend
   on it.
3. Add async checkpoint writeback once restore and receipt invariants are
   encoded.
4. Add the local metric-sink split only after the first three items land.

## Bottom Line

Burn is most useful to Psionic as a source of narrow implementation patterns.

The best additional pull-ins are:

- checkpoint migration tooling
- backend conformance structure
- async checkpoint writeback
- small training telemetry helpers

The wrong move is trying to import Burn’s framework shape into Psionic. Psionic
already has stronger public architecture for execution truth, routing,
capability publication, and distributed systems.
