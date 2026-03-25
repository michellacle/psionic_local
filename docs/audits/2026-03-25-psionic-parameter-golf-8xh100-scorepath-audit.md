# Parameter Golf 8xH100 Score-Path Audit

Date: 2026-03-25

## Contract

The upstream Parameter Golf contract is explicit:

- record-track training must complete in under `600` seconds on `8xH100`
- record-track evaluation has a separate `600` second budget
- the public score is FineWeb validation `bits_per_byte`

The current public scoreboard in [`~/code/parameter-golf/records/track_10min_16mb/`](../../../parameter-golf/records/track_10min_16mb) is already below `1.13 val_bpb`, with the best local record surface at the time of this audit showing `1.119400` for `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`.

That public top record also reports:

- about `83.4 ms` per training step
- about `7,185` steps inside the `600` second training budget
- about `410` seconds for the legal score-first TTT overlay inside the separate eval budget

Psionic is not close to that score path yet. The current gap is not only model quality. The current `8xH100` execution topology is too slow to produce a valid scoreboard-grade run.

## Current Evidence

### 1. Operator and input binding are now real

Retained RunPod proof root:

- `/workspace/parameter-golf-runpod/parameter-golf-runpod-scoreproof-20260325T091348Z`

What this run proves:

- `pre_training` completed successfully
- immutable PGOLF inputs were bound correctly from the retained materialization report
- the exported-folder `distributed_8xh100_train` mode crossed the old missing-env/operator failures
- the shipped runtime entered the real distributed benchmark root at:
  - `.../parameter-golf-distributed-8xh100-run/benchmark/`

This closes the old operator/input gap. That work now belongs in the closed issue `#544`.

### 2. The current distributed proof topology is now multi-step, but it is still catastrophically slow

Earlier retained proof root:

- `/workspace/parameter-golf-runpod/parameter-golf-runpod-scoreproof-20260325T091348Z`

That older root retained one-step rank-local wallclock in the `217s..229s` range.

Fresh current-bundle proof root:

- `/tmp/parameter-golf-runpod/parameter-golf-runpod-scoreproof-20260325T121903Z`

Fresh retained rank-0 train-step receipts from that root:

- `step_00001`: `observed_wallclock_ms=92839`
- `step_00002`: `observed_wallclock_ms=89203`
- `step_00003`: `observed_wallclock_ms=81437`

Fresh retained rank-0 phase timings:

- `step_00001`: `forward_loss_cuda_ms=58931`, `backward_cuda_ms=33053`
- `step_00002`: `forward_loss_cuda_ms=58426`, `backward_cuda_ms=29797`
- `step_00003`: `forward_loss_cuda_ms=50082`, `backward_cuda_ms=30082`

The retained aggregated gradient artifact is:

- `.../benchmark/runtime_train_step_gradients/aggregated_step_1.safetensors`
- size: about `66 MB`

This is real improvement relative to the earlier one-step proof root, and it proves the repeated loop is live through `step_00003`.

It is still catastrophically slow relative to the public score path. The public top record reports `83.4 ms` per step. Psionic is still at `81.4 s` on the freshest retained rank-0 step. That is roughly three orders of magnitude away from the competitive runtime posture.

### 3. The current proof topology still uses expensive orchestration

The current `8xH100` runtime path still does all of the following in the hot path:

- parent process spawns `8` runtime child processes for the train step
- parent writes per-rank window JSON before the step
- each child exports one per-rank gradient `safetensors` artifact
- parent waits for all children, then reopens every gradient artifact on disk
- parent aggregates gradients on host
- parent applies the optimizer step on host
- parent later spawns `8` more runtime child processes for validation

This is the wrong topology for scoreboard-grade throughput.

It is useful as a retained proof lane. It is not a viable steady-state record lane.

### 4. Prior retained validation evidence was already too slow

The earlier real distributed validation proof documented on `#510` had only reached roughly `30..36 / 119` validation batches per rank after about `27..28` minutes.

That earlier evidence matters because it shows the validation path is also not just “missing a few receipts.” The old distributed validation path was materially too slow for the `600` second scoreboard budget.

## Structural Gaps

### Gap A: spawn-per-step worker lifecycle

The current distributed runtime launches fresh child processes per step and again per validation pass.

That forces:

- process startup overhead
- fresh CUDA runtime setup
- fresh model reconstruction in every child
- repeated benchmark-root file IO in the hot path

The scoreboard lane needs persistent rank workers, not repeated process fanout.

### Gap B: file-artifact gradient synchronization

The current runtime still synchronizes one step through per-rank gradient files plus one parent-side aggregation pass.

That means the hot path still depends on:

- writing eight gradient artifacts
- reopening eight gradient artifacts
- host-side aggregation
- only then moving to the next step

This is not comparable to the upstream `torch.distributed` all-reduce posture.

### Gap C: repeated training is now real, but still not close to score-path throughput

The current `8xH100` runtime now executes repeated steps under the wallclock-bounded source loop.

That closes the old one-step-only boundary. It does not close the score-path throughput boundary.

The upstream loop in `train_gpt.py` still:

- keeps model and optimizer state resident
- advances steps until the wallclock cap is reached
- then runs final roundtrip validation

Without competitive step throughput, Psionic still does not own a valid scoreboard-grade training run.

### Gap D: final artifact semantics are still wrong for real training

The exported-folder completion path still binds the static shipped `final_model.int8.ptz` from the package manifest.

That is not sufficient for a real training run. A real score path must:

- export the trained final artifact from the live runtime
- bind the completion receipt to that exact trained artifact digest and size
- evaluate the trained roundtrip artifact, not the stale packaged fixture artifact

### Gap E: validation still needs the same persistent-runtime treatment

Even after sliding-window scoring landed, the distributed validation path still uses spawned children and full per-child model reconstruction.

That is not viable for the `600` second eval budget if it remains the final design.

## What This Means

Two conclusions are now defensible:

1. The current operator lane is no longer the main blocker.
2. A simple `while elapsed < 600s` wrapper around the current one-step proof will not produce a solid score.

The next score-path work must change the runtime topology, not just the receipt surface.

## Required Path Forward

### Priority 1: persistent distributed worker mesh

Replace the spawn-per-step proof lane with one persistent `WORLD_SIZE=8` runtime mesh that:

- launches the rank workers once
- keeps model and optimizer state resident across steps
- keeps step-local graph state warm
- exposes one coordinator path for stop conditions and finalization

### Priority 2: in-memory gradient synchronization

Replace per-rank gradient artifact export in the hot path with a real distributed synchronization path:

- collective or equivalent in-memory reduction
- no per-step gradient `safetensors` roundtrip in the score lane
- no parent-side file reopen and host reduction in the score lane

### Priority 3: real wallclock-capped repeated training loop with competitive step throughput

Once the persistent mesh exists, implement the actual public challenge loop:

- repeated steps under `MAX_WALLCLOCK_SECONDS=600`
- learning-rate / warmdown behavior tied to elapsed wallclock
- final stop when the cap is reached
- measured repeated-step receipt surface instead of one-step-only proof receipts

### Priority 4: trained final artifact closure

The distributed runtime must export the real trained final artifact from the live run and bind:

- final artifact path
- final artifact digest
- final artifact size
- final distributed validation metrics

to one honest completion receipt.

### Priority 5: scoreboard-grade validation and later TTT

Only after the runtime topology is fast enough should Psionic spend serious effort on:

- sliding-window scoreboard tuning
- legal score-first TTT
- model-architecture and schedule improvements aimed at matching the public top entries

Those improvements matter for score quality. They do not remove the current execution bottleneck.

## Issue Mapping

The current open issue stack only partially captures this.

Existing relevant issues:

- `#510`: real distributed validation and aggregation
- `#512`: exported-folder completion and receipt closure
- `#541`: scoreboard-grade sliding-window evaluation
- `#543`: real `600` second multi-step `8xH100` loop
- `#545`: persistent distributed worker mesh
- `#546`: scoreboard-grade full-sequence attention kernels
- `#547`: retained activation and primal volume
- `#549`: resident train-path model surface
- `#550`: distributed legal score-first TTT
- `#551`: parameter-banked PGOLF model surface
- `#552`: Parallel Muon collectives

The open issue stack is now explicit about the two additional competitive-path gaps that were not tracked when this audit was first written:

- Parameter Banking in the model surface
- Parallel Muon collectives in the distributed optimizer path

## Bottom Line

Psionic now has a real RunPod `8xH100` operator lane and a real repeated-step distributed proof path.

Psionic does not yet have a scoreboard-grade `8xH100` execution topology.

The next serious work is not another operator fix. It is replacing the proof topology with a persistent distributed trainer that can actually use the `600` second training budget efficiently.
