# 2026-03-25 Cross-Provider Pretraining System Readiness Audit

This audit answers one concrete question:

- how the new Google two-node swarm lane compares to the rest of the `psionic`
  training stack
- how close `psionic` is to one training system that works across any compute
  source
- what still has to land before Google, RunPod, local NVIDIA, local Apple, and
  later providers can all contribute truthfully to one real pretraining program

## Executive Verdict

`psionic` is now strong on training control-plane truth.

It is not yet strong enough to claim one provider-neutral pretraining system
where arbitrary compute from Google and elsewhere can all participate in one
real full-model pretraining run under the same train-step semantics.

The current state is:

- cluster identity, topology, configured-peer manifests, evidence bundles, and
  failure drills are `implemented_early`
- train run graph, orchestrator state, worker protocol, replay truth, and
  checkpoint lineage are `implemented_early`
- single-node accelerated training is real on Google
- bounded multi-node swarm contribution is real on Google
- bounded mixed-hardware swarm contribution is partially real on Mac plus RTX
  4080 hardware, but the master issue still stays open because the final
  accepted live result bar is not yet met
- the strongest distributed full-model lane, Parameter Golf `8xH100`, still
  stops at a real Rust-native bootstrap boundary and does not yet own the full
  distributed train-step, validation, export, and evidence closure
- mixed-backend dense training across Apple and CUDA is still `planned`

The fastest honest path to "many compute sources contribute to one real
pretraining program" is not one immediate mixed-backend synchronous trainer.

The fastest honest path is:

1. finish one real homogeneous distributed CUDA training runtime and reuse it
   across providers
2. unify provider launch, artifact, and telemetry contracts around the same run
   graph and runtime surfaces
3. let heterogeneous hardware participate under the same pretraining program
   through explicit contribution classes that match its actual capability
4. attempt mixed-backend dense training only after the homogeneous distributed
   trainer, model IO, checkpoint, and conformance layers are already stable

## Sources Consulted

Canonical docs:

- `docs/TRAIN_SYSTEM.md`
- `docs/PSION_GOOGLE_TWO_NODE_SWARM_RUNBOOK.md`
- `docs/PSION_GOOGLE_SINGLE_GPU_RUNBOOK.md`
- `docs/PSION_TRUSTED_CLUSTER_RUN.md`
- `docs/PSION_RENTED_CLUSTER_RUNBOOK.md`
- `docs/PSION_PILOT_PRETRAINING_RUN.md`
- `docs/PSION_PRETRAIN_STAGE.md`
- `docs/PARAMETER_GOLF_RUNPOD_8XH100_RUNBOOK.md`
- `docs/PARAMETER_GOLF_DISTRIBUTED_8XH100.md`
- `docs/CLUSTER_VALIDATION_RUNBOOK.md`
- `docs/ROADMAP_CLUSTER.md`
- `docs/TRAIN_RUN_GRAPH_REFERENCE.md`
- `docs/TRAIN_ORCHESTRATOR_REFERENCE.md`
- `docs/DISTRIBUTED_OPTIMIZER_REFERENCE.md`
- `docs/DISTRIBUTED_DATA_FEED_SEMANTICS.md`
- `docs/TRAIN_CHECKPOINT_RECOVERY_REFERENCE.md`
- `docs/TRAIN_ARTIFACT_STORAGE_REFERENCE.md`
- `docs/MODEL_IO_REFERENCE.md`
- `docs/REMOTE_TRAINING_VISUALIZATION.md`
- `docs/PSION_DECENTRALIZED_CONTRIBUTION.md`

Relevant audits:

- `docs/audits/2026-03-22-psion-training-system-full-state-audit.md`
- `docs/audits/2026-03-24-clustered-training-mac-nvidia-readiness-audit.md`
- `docs/audits/2026-03-24-google-two-node-swarm-compute-audit.md`
- `docs/audits/2026-03-24-psionic-parameter-golf-validation-runtime-audit.md`
- `docs/audits/2026-03-25-google-two-node-swarm-first-real-run-audit.md`
- `docs/audits/2026-03-25-psionic-burn-port-audit.md`

Relevant code:

- `crates/psionic-train/src/run_graph.rs`
- `crates/psionic-train/src/orchestrator.rs`
- `crates/psionic-train/src/adapter_cluster.rs`
- `crates/psionic-train/src/open_adapter.rs`
- `crates/psionic-train/src/swarm_open_adapter.rs`
- `crates/psionic-train/src/psion_google_two_node_swarm_runtime.rs`
- `crates/psionic-train/src/psion_trusted_cluster_run.rs`
- `crates/psionic-train/src/distributed_optimizer.rs`
- `crates/psionic-train/src/model_io.rs`
- `crates/psionic-train/src/parameter_golf_distributed_8xh100_runtime_bootstrap.rs`
- `crates/psionic-train/src/parameter_golf_submission_runtime.rs`
- `crates/psionic-train/src/remote_training_visualization.rs`

Live GitHub issue state checked with `gh` on 2026-03-25:

- `#473` is open for real Rust-native distributed Parameter Golf execution
- `#466` is open for the remaining `train_gpt.py` parity gaps
- `#479` is open because the remote-visualization substrate is still not fully
  live on the RunPod distributed lane
- `#484` is open because the mixed Mac-plus-RTX 4080 swarm master bar still
  requires one truthful accepted live result
- `#514`, `#515`, and `#516` are open for backend conformance, async
  checkpoint writeback, and shared local metric sinks
- the Google two-node swarm issue stack `#501` through `#508` is closed

## Comparative State

| Lane or Subsystem | Status | What Is Real | Current Limit |
| --- | --- | --- | --- |
| Psion single-node accelerated Google run | `implemented_early` | one real accelerator-backed single-node training lane with retained Google evidence | single-node only |
| Google two-node swarm | `implemented_early` | one real two-node configured-peer Google adapter-delta lane with real `bounded_success` runs and retained evidence | not full-model training, not elastic, not public swarm |
| Local Mac plus RTX 4080 swarm | `partial` | one truthful mixed-hardware decentralized adapter stack with comparable receipts, bring-up, rehearsal, evidence, and closeout surfaces | master issue still open because the lane has not yet earned the final accepted live result bar |
| Trusted-cluster training | `implemented_early` | one bounded homogeneous CUDA trusted-cluster training claim with topology and distributed-group receipts | narrow homogeneous cluster claim only |
| Parameter Golf single-H100 | `implemented_early` | one real remote CUDA training lane with provider-neutral visualization support | single host |
| Parameter Golf RunPod `8xH100` | `partial` | one real Rust-native distributed bootstrap and explicit operator/finalizer lane | full distributed train-step, validation, export, and evidence closure still open |
| Train run graph and orchestrator | `implemented_early` | typed run identity, contributor revisions, window state, assignment posture, and trainer-batch control | not yet the universal runtime entrypoint for all training lanes |
| Distributed optimizer contract | `partial` | typed optimizer, memory, and precision contracts with bounded public helper surfaces | no broad real transport-backed distributed trainer runtime yet |
| Model IO portability | `implemented_early` | typed portable import/export, selective import, remap, and deferred materialization | not yet full distributed checkpoint interchange and shard recovery closure |
| Remote training visualization | `implemented_early` | one provider-neutral app-facing bundle and run-index family | distributed RunPod lane still lacks the live writer required to close `#479` |

## What The Google Two-Node Swarm Lane Actually Adds

The new Google two-node swarm lane is important because it proves that
`psionic` can already do all of the following on real cloud hardware:

- own the network, identity, quota, launch, startup, impairment, finalizer, and
  evidence surfaces for a multi-node lane
- keep the lane provider-managed without moving training truth into provider
  logs or ad hoc operator notes
- reuse the existing generic adapter-cluster, worker-protocol, validation, and
  aggregation substrate instead of inventing a second Google-specific training
  control plane
- survive real cloud timing skew and mild network impairment with explicit
  receipts

That closes one important infrastructure question.

It does not close the main math question.

The Google swarm lane is one bounded adapter-delta contribution program. It
proves that multi-node cross-host contribution control is real on Google. It
does not prove that `psionic` can already run one full-model multi-rank
pretraining job across those nodes.

## How It Relates To The Other Training Lanes

### Google single-node Psion

This is the strongest proof that `psionic` can run one real accelerator-backed
full-model training lane under a provider-managed operator surface.

It gives the system:

- real single-node training math
- real checkpoint and evidence retention
- real Google operator scripts

It does not give the system:

- multi-node train-step semantics
- multi-rank validation
- provider-neutral multi-host launch semantics

### Google two-node swarm

This is the strongest proof that `psionic` can run one real cloud multi-node
training control path with launch, runtime, impairment, and finalization truth.

It gives the system:

- real configured-peer cloud cluster bring-up
- real multi-node runtime coordination
- real provider-owned evidence bundle retention

It does not give the system:

- full-model distributed optimizer execution
- full-model distributed validation
- a generic provider-neutral launcher shared with RunPod or local lanes

### Local mixed-hardware swarm

This is the strongest proof that `psionic` already understands that different
compute sources need different admitted execution roles.

It gives the system:

- one shared decentralized adapter contract across MLX/Metal and CUDA
- comparable backend-tagged contributor receipts
- a real mixed-hardware bring-up surface

It does not give the system:

- accepted same-job dense training across MLX and CUDA
- one truthful mixed-backend all-reduce path

### Trusted-cluster training

This is the strongest proof that the repo already has a truthful homogeneous
cluster-training claim with explicit distributed-group and checkpoint-restart
receipts.

It gives the system:

- a bounded homogeneous distributed-training proof
- a clear contract for trusted topology and replay-safe recovery

It does not give the system:

- cross-provider launch portability
- heterogeneous hardware participation
- a broad elastic training mesh

### Parameter Golf

Parameter Golf is the strongest forcing function for the actual missing runtime.

It gives the system:

- the strongest current remote CUDA operator surfaces
- the sharpest parity bar against a public distributed baseline
- a real Rust-native distributed bootstrap on `8xH100`

It does not yet give the system:

- the real distributed train-step closure
- distributed validation sharding and aggregation closure
- final export and evidence closure from real distributed execution

This matters because the universal training system should reuse the same
distributed runtime that Parameter Golf is already forcing into existence.

## What Is Already Universal Across Compute Sources

Several important layers are already shaped correctly for a compute-source-
agnostic system.

### Control-plane truth is ahead of runtime math

`run_graph.rs`, `orchestrator.rs`, `adapter_cluster.rs`, and the worker
protocol surfaces already make these things explicit:

- stable run identity
- stable participant identity
- contributor selection and window planning
- assignment posture
- heartbeats, stale-worker handling, and departure state
- validator and aggregation transitions

That is the right shared control-plane shape for local, Google, RunPod, and
later providers.

### Machine-legible evidence is already a first-class requirement

The Google single-node, Google swarm, trusted-cluster, Parameter Golf, and
mixed-hardware swarm lanes all keep receipts, manifests, evidence bundles,
finalizers, and refusal posture explicit.

That matters because a cross-provider training system fails if each provider
needs its own narrative-only success story.

### Model-state portability is finally getting real

`model_io.rs` and `docs/MODEL_IO_REFERENCE.md` now give `psionic`:

- portable state dict ownership
- adapter merge and unmerge
- selective import
- key remap
- deferred materialization

That is necessary for a system that moves state between providers, runtimes,
and machines with different memory budgets.

### Remote visualization already has the right authority split

`remote_training_visualization.rs` keeps Psionic responsible for machine-facing
truth while app surfaces consume one provider-neutral bundle family.

That same pattern should apply to the broader provider-neutral training system:
Psionic should emit one training truth family; providers should only satisfy
resource and transport contracts.

## What Is Still Missing For One Real Cross-Provider Pretraining System

### 1. One universal dense distributed training runtime

This is the largest missing piece.

`docs/DISTRIBUTED_OPTIMIZER_REFERENCE.md` is explicit that the distributed
optimizer layer is still a typed contract, not a completed broad multi-device
runtime. Parameter Golf `8xH100` now has a real Rust-native bootstrap path, but
`#473` is still open because the real distributed train-step still is not done.

Until this lands, `psionic` does not have one reusable full-model distributed
trainer that Google, RunPod, and later providers can all share.

### 2. One provider-neutral compute-source contract

The current provider lanes are still too script-shaped and provider-specific.

Google single-node, Google swarm, and RunPod `8xH100` each have good launch and
finalizer surfaces. They do not yet fold into one shared provider-neutral
contract that answers:

- what a compute source must report before admission
- how a compute source publishes network, storage, accelerator, and cost truth
- how a training program binds artifact roots, launch receipts, and teardown
- how the same run graph maps to local, Google, RunPod, or later providers

The cluster layer already has much of the topology and identity substrate.
What is still missing is the training-facing compute-source contract on top of
that substrate.

### 3. One explicit role taxonomy for heterogeneous contributors

If the goal is "any compute source can contribute," the system needs to stop
pretending every node will be a dense synchronous rank.

The current repo already hints at the correct split:

- homogeneous CUDA nodes can eventually be dense full-model ranks
- Apple MLX and weaker NVIDIA nodes can already participate honestly in bounded
  adapter windows
- validators, checkpoint writers, data builders, and eval workers can also be
  first-class run roles

That role split is not yet frozen as one universal pretraining-program
contract.

Without that contract, the system keeps oscillating between two bad options:

- overclaiming mixed-backend dense training that does not exist
- underusing heterogeneous compute that could still contribute honestly

### 4. Distributed data semantics that survive real multi-provider runs

`docs/DISTRIBUTED_DATA_FEED_SEMANTICS.md` is still bounded to fixed-world-size
seeded distributed feeds. That is not enough for:

- cross-provider contributor churn
- mixed dense-rank plus contributor-window programs
- long-running pretraining programs that reassign work across machine classes

The data plane needs a stronger contract for:

- shard ownership across provider boundaries
- deterministic re-assignment after node loss
- dense-rank versus contributor-window sampling semantics
- stable data receipts that remain comparable across execution classes

### 5. Distributed checkpoint and recovery closure

`docs/TRAIN_CHECKPOINT_RECOVERY_REFERENCE.md` still does not claim distributed
optimizer recovery or parameter-shard semantics. `#515` is still open for async
checkpoint writeback. `docs/TRAIN_ARTIFACT_STORAGE_REFERENCE.md` also still does
not claim a real external blob-store client or remote placement strategy.

That blocks a serious cross-provider pretraining program because those runs need
all of the following:

- frequent checkpoint flush without stalling the train loop
- resumable sharded checkpoint uploads
- explicit durability and restore semantics across providers
- the ability to recover a dense rank mesh after preemption or provider loss

### 6. Shared backend conformance and portability discipline

`#514` is still open. That matters directly.

The training system cannot "work the same way across any compute source" unless
the backend family beneath it can prove:

- what is supported
- what is unsupported
- what is correct
- what only works on one backend today

The Mac-plus-CUDA swarm lane already uses comparable receipts. That is not yet
the same thing as one shared correctness harness across CPU, CUDA, Metal, and
MLX-backed lanes.

### 7. Shared train-loop metric and live telemetry plumbing

`#516` is still open. `#479` is still open.

The app-facing remote visualization contract is good. The remaining gap is that
not every runtime writes a consistent local metric stream and not every remote
lane can yet keep the live bundle fresh every second.

That becomes more important, not less, when one pretraining program spans many
providers and many role classes.

## Estimated Distance To The Goal

These percentages are audit estimates, not machine-derived metrics.

| Goal | Estimated Readiness | Why |
| --- | --- | --- |
| Shared control plane, receipts, and evidence model across providers | `65%` | run graph, orchestrator, cluster manifests, evidence bundles, model IO, and visualization contracts are real; provider launch and dense-runtime entrypoints are still fragmented |
| Real homogeneous CUDA full-model training across multiple providers | `35%` | single-node Google and distributed bootstrap work are real, but the actual distributed train-step and validation closure remain open in `#473`, `#510`, `#511`, and `#512` |
| One pretraining program with multiple admitted contribution classes across mixed hardware | `45%` | decentralized adapter windows and mixed-hardware receipts are real, but the role taxonomy and shared program contract are not yet frozen end to end |
| True mixed-backend dense training across Apple and CUDA | `10%` | the repo has strong bring-up, comparable receipts, and bounded mixed-hardware rehearsals, but no truthful mixed dense train-step runtime exists |
| One system that "works the same way across any compute source" | `25%` | the foundations are good, but the universal runtime, role model, distributed data, and checkpoint semantics are still missing |

## What The Full Implementation Should Look Like

The full implementation should not be "some scripts for Google, some scripts
for RunPod, and a separate mixed-hardware experiment."

It should be one training system with these stable layers.

### Layer 1: Provider-neutral training program contract

The root object should bind:

- run identity
- stage identity
- checkpoint family
- dataset family
- admitted execution classes
- artifact roots
- cost and time budgets
- finalizer and evidence policy

This should sit above provider launch details and below app UX.

### Layer 2: Provider-neutral compute-source contract

Every compute source should publish the same training-facing identity:

- provider and region or locality
- accelerator family and count
- backend family
- precision support
- network posture
- storage posture
- cost ceiling and billing unit
- checkpoint and artifact bandwidth expectations

This should let the same planner admit:

- a Google `g2` plus `L4` node
- a RunPod `8xH100` pod
- a local RTX 4080 desktop
- a local MLX-capable Mac

without inventing a separate training truth surface for each one.

### Layer 3: Explicit execution classes

The training program should admit several role kinds and keep them separate:

- `dense_full_model_rank`
- `validated_contributor_window`
- `validator`
- `checkpoint_writer`
- `eval_worker`
- `data_builder`

That is how "many compute sources contribute" becomes truthful.

The dense full-model ranks should run the same distributed runtime. The weaker
or incompatible nodes should still contribute to the same pretraining program
through admitted validated work units, not fake dense-rank membership.

### Layer 4: One real distributed CUDA runtime reused everywhere

This is the central runtime target:

- same train-step semantics
- same validation semantics
- same checkpoint semantics
- same metric semantics
- same evidence semantics

The provider should change. The runtime contract should not.

This is why Parameter Golf matters so much. The work in `#473` and its
dependencies should become the canonical dense distributed runtime for the
broader train system, not a contest-only side lane.

### Layer 5: Shared checkpoint and model-state mobility

Checkpoint and model IO need to support:

- sharded distributed checkpoints
- resumable remote writeback
- restore across provider boundaries
- partial and remapped import for heterogeneous follow-on roles
- explicit optimizer-state portability boundaries

The new model-IO layer is a real start. It is not the whole answer.

### Layer 6: Shared telemetry, finalization, and app surfaces

Every lane should emit:

- one local metric stream
- one remote visualization bundle family
- one final evidence bundle family
- one refusal classification vocabulary

The app should not care whether the run came from Google, RunPod, or local
cluster hardware.

## Recommended Expansion Sequence

### Phase 1: Finish the real dense distributed runtime

Do this first.

Primary existing issue stack:

- `#473`
- `#466`
- `#510`
- `#511`
- `#512`

Reason:

- without this, the system still lacks the main reusable multi-rank train-step
  engine

### Phase 2: Promote that runtime from PGOLF lane to train-system substrate

After the distributed CUDA runtime is real, bind it into the broader pretrain
system instead of leaving it isolated behind Parameter Golf wrappers.

This phase should freeze:

- one provider-neutral dense-rank runtime entrypoint
- one shared finalizer and evidence family
- one shared step, validation, and checkpoint metric vocabulary

This work is only partially covered by the current issue stack. It likely needs
new issues.

### Phase 3: Freeze the compute-source and execution-class contract

Land one canonical training-facing compute-source contract and one canonical
role taxonomy.

This should cover:

- local versus rented versus cloud-managed sources
- dense ranks versus contributor windows
- validator and auxiliary roles

This also likely needs new issues.

### Phase 4: Unify Google, RunPod, and local launch paths around that contract

Keep provider-specific resource creation where it belongs. Move training truth
into shared launch manifests, startup contracts, runtime env contracts, and
finalizer contracts.

The Google swarm work already shows the right pattern. The RunPod and
single-node Google paths should converge toward it.

### Phase 5: Close checkpoint, storage, and metric gaps

Primary existing issues:

- `#515`
- `#516`
- `#479`
- `#514`

Additional work still needs to be defined for:

- distributed checkpoint shard manifests
- provider-neutral remote artifact placement and restore strategy
- dense-rank recovery after provider or node loss

### Phase 6: Expand heterogeneous contribution under the same program

Keep this separate from dense mixed-backend math.

Primary existing issue:

- `#484`

Broader follow-on should:

- generalize the mixed-hardware swarm role model
- bind the contributor-window path to a named pretraining-program contract
- keep accepted contribution, replay, and promotion rules explicit

### Phase 7: Attempt mixed-backend dense training only if still needed

Do this last.

The repo does not need mixed Apple-plus-CUDA dense all-reduce to make many
compute sources contribute to one pretraining program. It only needs that if
product or research goals require the exact same dense train-step graph to span
those backends.

## What Not To Do

- Do not build separate Google, RunPod, and local training control planes.
- Do not describe adapter-delta swarm lanes as if they already solve full-model
  pretraining.
- Do not try to close the universal system by mixing MLX and CUDA into one
  dense synchronous trainer before the homogeneous CUDA distributed runtime is
  real.
- Do not let provider-specific scripts become the long-term training API.

## Conclusion

`psionic` is meaningfully closer to a real cross-provider training system than
it was even a few days ago.

The Google two-node swarm lane proves that real cloud multi-node training
control is no longer theoretical. The model-IO work proves that state
portability is starting to become explicit. The cluster layer, run graph,
orchestrator, and evidence discipline are already strong.

The missing center is still the reusable distributed training runtime.

Until `psionic` finishes one real provider-neutral dense distributed runtime and
binds it to one compute-source contract plus one heterogeneous role taxonomy,
the truthful claim is:

- `psionic` has the right training control-plane direction
- `psionic` has several real bounded training lanes
- `psionic` does not yet have one training system that works the same way
  across any compute source for one real broad pretraining run

The shortest honest implementation path is:

1. finish homogeneous distributed CUDA execution
2. make that runtime the shared train-system substrate
3. standardize compute-source and role contracts
4. expand heterogeneous contributors under the same pretraining program
5. leave true mixed-backend dense training for later
