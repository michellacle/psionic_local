# 2026-03-24 Autopilot WGPUI Google and RunPod Training Visualization Audit

This audit answers one concrete question:

- what has to exist so the OpenAgents Autopilot app can render a truthful WGPUI
  visualization of the Psionic training runs now happening, or actively being
  prepared, on Google Cloud and RunPod
- what has to change so that visualization stays visibly alive at least once
  per second during an active run instead of waiting for phase boundaries,
  sparse loss flushes, or post-run finalizers

I inspected the current `psionic` issue state with `gh`, the current Google and
RunPod training docs and scripts in this repo, and the current Autopilot app
code in `~/code/openagents/apps/autopilot-desktop/`.

## Executive Verdict

Autopilot can get to a strong remote training viewer without inventing a new UI
stack.

The existing pieces are already real:

- Autopilot already has a WGPUI loss-curve implementation in the AttnRes Lab
  pane.
- Autopilot already has an app-owned training projection path in
  `desktop_control` and a live training operator pane for the Apple adapter
  lane.
- Psionic already retains real Google final manifests, GPU samples, stage
  receipts, observability receipts, accelerator-validation receipts, and
  bounded cost receipts for the current Google single-node lanes.
- The Parameter Golf single-H100 trainer already emits a machine-readable
  `step_metrics` series that is good enough to drive a real loss curve.

The missing substrate is still not WGPUI rendering.

The missing substrate is live telemetry shape.

The missing substrate is:

- one live heartbeat contract that updates the app at least once per second
  while a run is active
- one provider-neutral remote-training visualization bundle
- one provider-neutral model and optimizer telemetry contract so the app can
  show more than occasional loss points
- one app-owned fetch and projection path from Google and RunPod artifacts into
  Autopilot state
- one explicit truth contract for which lanes have a real per-step loss series
  and which lanes only have summaries, logs, or final metrics

Most current Google and RunPod surfaces are finalize-time or summary-time
artifacts.

That is good enough for audits and postmortems.

That is not good enough for a viewer that must look alive every second during a
run.

That means the right architecture is:

- keep the visualization app-owned in Autopilot
- keep the execution truth machine-owned in Psionic artifacts
- add one narrow normalization layer between them

## Sources Consulted

`psionic`:

- open issues via `gh issue list` and `gh api repos/OpenAgentsInc/psionic/issues?state=open&per_page=100`
- `docs/TRAIN_SYSTEM.md`
- `docs/ARCHITECTURE.md`
- `docs/PSION_RUN_OBSERVABILITY.md`
- `docs/PARAMETER_GOLF_GOOGLE_SINGLE_H100_RUNBOOK.md`
- `docs/PARAMETER_GOLF_RUNPOD_SINGLE_H100_AUDIT.md`
- `docs/PARAMETER_GOLF_RUNPOD_8XH100_RUNBOOK.md`
- `docs/PARAMETER_GOLF_EXPORTED_SUBMISSION_EVIDENCE.md`
- `docs/audits/2026-03-22-openagentsgemini-first-google-single-gpu-pilot-run-audit.md`
- `docs/audits/2026-03-23-openagentsgemini-first-google-accelerator-backed-single-node-psion-training-audit.md`
- `docs/audits/2026-03-23-openagentsgemini-first-google-accelerator-backed-host-native-plugin-conditioned-run-audit.md`
- `docs/audits/2026-03-23-openagentsgemini-query-backed-google-single-node-cost-receipt-audit.md`
- `crates/psionic-train/src/parameter_golf_single_h100_training.rs`
- `crates/psionic-train/src/bin/parameter_golf_single_h100_train.rs`
- `scripts/psion-google-single-node-startup.sh`
- `scripts/psion-google-finalize-run.sh`
- `scripts/parameter-golf-runpod-finalize-single-h100.sh`
- `scripts/parameter-golf-runpod-finalize-8xh100.sh`

`openagents`:

- `apps/autopilot-desktop/src/app_state_domains.rs`
- `apps/autopilot-desktop/src/attnres_lab_control.rs`
- `apps/autopilot-desktop/src/panes/attnres_lab.rs`
- `apps/autopilot-desktop/src/desktop_control.rs`
- `apps/autopilot-desktop/src/panes/apple_adapter_training.rs`
- `docs/headless-compute.md`

## Current Training Work As Of March 24, 2026

The active open training backlog in `psionic` is the Parameter Golf remote and
parity stack.

Open issues:

- `#454` `PGOLF_GOOGLE-1`: restore the Rust-only single-H100 baseline trainer
- `#458` `PGOLF_GOOGLE-5`: run and audit the first real RunPod single-H100
  baseline
- `#461` `PGOLF_GOOGLE-8`: capture the first real RunPod `8xH100`
  exported-folder evidence bundle
- `#463` `PGOLF_GOOGLE-10`: freeze one H100-backed record candidate and run a
  repeated evidence campaign
- `#464` `PGOLF_GOOGLE-11`: produce the final submission dry run and readiness
  audit
- `#466` `PGOLF_PARITY-0`: close the remaining `train_gpt.py` parity gaps
- `#469` `PGOLF_PARITY-3`: move the baseline to BF16-first mixed precision
- `#470` `PGOLF_PARITY-4`: retire the remaining CUDA host-fallback ops
- `#472` `PGOLF_PARITY-6`: add a same-node H100 parity harness
- `#473` `PGOLF_PARITY-7`: replace the analytic `8xH100` lane with real
  Rust-native distributed execution

The issue state matters for the UI plan because the app has to render more than
successful completed runs.

The app must truthfully represent:

- real bounded success runs
- real bounded failure runs
- manifest-only rehearsals
- operator lanes that exist but do not yet have a real audited run
- live runs that may only have partial telemetry so far

## Non-Negotiable Live Requirement

The requirement for this audit is stronger than "show a loss curve when one is
available."

The app must visibly update at least once per second during every active run.

That rules out a design that only refreshes when:

- a training phase ends
- a checkpoint lands
- a validation pass finishes
- a sparse loss point is flushed
- a post-run finalizer uploads artifacts

The live viewer has to stay visually active during:

- provisioning
- dataset staging
- compilation or graph warmup
- microbatch execution
- optimizer steps
- distributed synchronization
- evaluation
- checkpoint I/O
- artifact upload
- teardown
- failure handling

If no optimizer step completes within one second, the runtime still has to emit
a one-second heartbeat with truthful current-state telemetry.

Loss curves alone do not satisfy this requirement.

The app needs a full live surface for:

- lifecycle and current phase
- model and optimizer math
- device utilization and memory
- data, network, and checkpoint I/O
- distributed stalls and skew when the run is multi-GPU
- event flow, warnings, and failure posture

## What Autopilot Already Has

Autopilot already has two reusable building blocks.

### 1. A real WGPUI loss-curve and telemetry pane

The AttnRes Lab is the existing WGPUI reference implementation.

The important files are:

- `apps/autopilot-desktop/src/app_state_domains.rs`
- `apps/autopilot-desktop/src/attnres_lab_control.rs`
- `apps/autopilot-desktop/src/panes/attnres_lab.rs`

What is already there:

- `AttnResLabMetricPoint` with `global_step`, `training_loss`, `ema_loss`, and
  `selectivity`
- `AttnResLabSnapshot.metrics` as a full time series
- a controller that rebuilds that history from live runtime state
- `paint_loss_curve_panel(...)` for the large chart
- `paint_loss_stream(...)` for compact rails
- event-feed rendering
- throughput, ETA, and runtime telemetry cards

This is not hypothetical. The charting code already exists and is app-owned.

### 2. A real training projection and run-detail path

Autopilot already projects training state through `desktop_control` and the
headless `autopilotctl training status` surface.

The important files are:

- `apps/autopilot-desktop/src/desktop_control.rs`
- `apps/autopilot-desktop/src/panes/apple_adapter_training.rs`
- `docs/headless-compute.md`

What is already there:

- a projected training operator summary
- projected training runs and current progress
- recent typed training events
- loss, ETA, checkpoint, and artifact detail in the Apple operator lane
- a WGPUI run-detail pane that already knows how to present live operator data

That means the app-side question is not "can Autopilot display training data at
all?"

The question is:

- how should Psionic remote-run artifacts be normalized so Autopilot can render
  them without scraping provider-specific logs directly in the pane code

## Current Remote Artifact Truth By Lane

| Lane | Current status | What is retained today | Can Autopilot drive a truthful always-live viewer today? | What is still missing |
| --- | --- | --- | --- | --- |
| Google generic accelerated single-node `Psion` | Real bounded-success run exists on March 23, 2026 | final manifest, timeline, GPU samples CSV, GPU summary, stage receipt, observability receipt, accelerator validation receipt, cost receipt, uploaded output-dir artifacts | No. The retained artifacts are summary-oriented and finalize-oriented. | one-second heartbeat, typed live runtime series, typed math series, and one visualization bundle |
| Google host-native accelerated plugin-conditioned `Psion` | Real bounded-success run exists on March 23, 2026 | final manifest, timeline, GPU samples CSV, GPU summary, stage receipt, observability receipt, evaluation receipt, run summary, uploaded output-dir artifacts | No. The retained artifacts are summary-oriented and finalize-oriented. | one-second heartbeat, typed live runtime series, typed math series, and one visualization bundle |
| Google Parameter Golf single-H100 | Operator lane exists; no real audited live run yet | launch profile, immutable input package, local rehearsal, trainer writes `parameter_golf_single_h100_training.json` with `step_metrics` when it runs | Partial. The trainer report is chartable after the fact, but it is not yet a one-second live-view contract. | live heartbeat emission, incremental trainer metric flush, and projection into a provider-neutral bundle |
| RunPod Parameter Golf single-H100 | Finalizer contract exists; real run still open in `#458` | trainer JSON if present, trainer log, pod identity, `nvidia-smi` captures, latest parsed micro-step in the audit JSON | No. The audit JSON is not enough and the finalizer runs too late for live viewing. | live heartbeat emission, full-series retention, and a stable bundle mirror |
| RunPod Parameter Golf `8xH100` exported-folder lane | Runbook and finalizer exist; real run still open in `#461` | launch manifest, provider posture, `nvidia-smi` inventory and topology, exported-folder digests, submission run evidence, raw `train.log` | No. There is no canonical live distributed metric stream. | structured distributed live series, one-second coordinator heartbeat, and a visualization bundle |

## What This Means In Practice

There is not one uniform answer across Google and RunPod.

There is also not one single chart that satisfies the requirement.

The requirement is continuous visual truth across all meaningful subsystems, not
just an occasional loss update.

`Everything` should mean every meaningful subsystem signal that can be exported
truthfully at bounded cost.

That does not mean full tensor dumps or per-parameter weights every second.

That would distort the run and swamp the UI.

It means bounded live telemetry in five bands:

- lifecycle telemetry: phase, subphase, elapsed time, progress markers,
  warnings, stale-heartbeat state
- optimizer and model math: train loss, EMA loss, validation loss, learning
  rate, gradient norm, parameter norm, update norm, clip events or clip
  fraction, mixed-precision loss scale, overflow or non-finite counts
- device telemetry: per-GPU utilization, memory used, memory headroom,
  temperature, power, and link or collective pressure when available
- runtime telemetry: dataloader wait time, forward time, backward time,
  optimizer time, checkpoint time, evaluation time, tokens per second, and
  samples per second
- distributed telemetry: rank skew, slowest-rank time, all-reduce or collective
  time, stalled-rank detection, coordinator heartbeat

Model-specific math telemetry should be included when the runtime exposes it at
bounded overhead.

Examples:

- activation norm
- logit entropy
- attention entropy
- dead-activation counters
- layerwise norm summaries

### Lanes that are already close

The closest lane is the Parameter Golf single-H100 trainer.

`crates/psionic-train/src/parameter_golf_single_h100_training.rs` already emits
`ParameterGolfSingleH100TrainingReport.step_metrics`, and each step row already
contains:

- `global_step`
- `mean_microbatch_loss`
- `observed_wallclock_ms`
- detailed phase timings

That is enough to drive:

- a real loss curve
- a runtime rail view
- a phase-breakdown panel
- live or post-run throughput summaries

That is not yet enough for the user's stated requirement.

The report has to be flushed incrementally during the run, or mirrored into an
append-only live artifact, so the app can repaint once per second even when a
long step is still in flight.

Autopilot does not need new rendering invention for that lane. It needs a live
projection path.

### Lanes that are only summary-complete

The current real Google `Psion` runs are good enough for:

- run list
- provider and profile badges
- launch and teardown timeline
- GPU utilization chart from `psion_google_gpu_samples.csv`
- throughput and cost cards
- checkpoint and receipt provenance

They are not yet good enough for a guaranteed always-live viewer from one
canonical artifact, because the retained docs and example outputs are
summary-oriented and finalize-oriented.

That is a telemetry-shape problem, not a WGPUI problem.

### Lanes that are still only operator-contract complete

The RunPod `8xH100` exported-folder lane is not chart-ready yet.

Today it has:

- operator posture
- finalizer posture
- exported-folder evidence posture
- raw `train.log`

That is enough for:

- status
- digests
- final metrics if the log or evidence report includes them

That is not enough for a truthful always-live viewer without one of these two
changes:

- a canonical structured step-series file emitted by the runtime, or
- a finalizer that parses `train.log` into a canonical series and preserves the
  parse result as an artifact

For the user's requirement, even that is incomplete unless the runtime emits a
live one-second coordinator heartbeat during the run.

## The Lowest-Regret Architecture

The lowest-regret path is one new provider-neutral artifact family plus one new
app projection.

### 1. Add one provider-neutral live visualization bundle in `psionic`

Psionic should emit one machine-readable bundle for app consumption.

Recommended shape:

- `schema_version`
- `provider`
- `profile_id`
- `lane_id`
- `run_id`
- `repo_revision`
- `result_classification`
- `refresh_contract`
  - `target_ui_update_interval_ms`
  - `emission_mode`
    - `stream`
    - `append_only_snapshots`
    - `post_run_only`
  - `last_heartbeat_at`
  - `heartbeat_seq`
- `series_status`
  - `available`
  - `partial`
  - `unavailable`
- `series_unavailable_reason`
- `timeline`
- `heartbeat_series`
  - timestamp
  - phase
  - subphase
  - step_in_progress
  - microbatch_in_progress
  - active_subsystems
  - stale_after_ms
- `summary`
  - step counts
  - current or final loss
  - final validation metrics when present
  - throughput
  - cost
  - checkpoint refs
- `loss_series`
  - step or epoch index
  - elapsed ms
  - train loss
  - EMA loss when available
  - validation loss when available
- `math_series`
  - learning rate
  - gradient norm
  - parameter norm
  - update norm
  - clip fraction or clipping events
  - loss scale
  - non-finite count
  - model-specific diagnostics when available
- `runtime_series`
  - data wait ms
  - forward ms
  - backward ms
  - optimizer ms
  - checkpoint ms
  - evaluation ms
  - tokens per second
  - samples per second
- `gpu_series`
  - timestamp
  - utilization
  - memory used
- `distributed_series`
  - timestamp
  - rank skew
  - slowest-rank ms
  - collective ms
  - stalled_rank_count
- `event_series`
  - typed phase and detail rows
- `source_artifacts`
  - URIs
  - digests
  - source receipt ids

This bundle must stay explicit about missing data.

If a lane only has summaries, the bundle must say:

- `series_status = unavailable`

and it must name the reason rather than backfilling invented points.

The same contract should allow one-second live snapshots during active runs and
a sealed final summary after completion.

### 2. Make the runtimes emit it live and the finalizers seal it

Per lane:

- Google generic accelerated `Psion`:
  add a runtime-owned live bundle writer under the output directory; finalizer
  uploads the sealed bundle and any append-only snapshots
- Google plugin-conditioned accelerated `Psion`:
  same rule
- Google PGOLF single-H100:
  map `step_metrics` directly into `loss_series`, add one-second heartbeat
  snapshots between completed steps, and flush math and runtime telemetry
- RunPod PGOLF single-H100:
  runtime should mirror live snapshots locally during the run; finalizer should
  load the trainer JSON and preserve the full series inside the visualization
  bundle
- RunPod PGOLF `8xH100`:
  extend the execution path so one coordinator-owned live distributed series
  exists, then seal it in the finalizer

Finalizers remain necessary for evidence retention.

They are not sufficient for a viewer that must look alive during the run.

### 3. Keep provider-specific fetch outside WGPUI pane code

The app should not read arbitrary GCS objects or SSH into RunPod pods from the
renderer.

The fetch path should be:

- provider sync or local cache layer
- `desktop_control` projection
- WGPUI pane state

That matches the existing Autopilot design.

The projection path should update on a one-second cadence while a run is active.

If provider sync cannot guarantee that cadence directly, the live source has to
be mirrored into a local cache first and the UI should read from that cache.

## Recommended Autopilot Landing Zone

Do not overload the AttnRes pane with remote-provider semantics.

Do not overload the Apple adapter training pane with Google and RunPod
infrastructure semantics either.

The clean landing zone is a new app-owned pane with shared components extracted
from AttnRes.

Recommended pane:

- `pane.psionic_remote_training` or similarly explicit naming

Recommended structure:

- top row: provider, lane, run status, run id, revision, cost ceiling, actual
  bounded cost, last heartbeat timestamp, stale badge when needed
- main panel: stacked live strip charts for loss, learning rate, gradient norm,
  tokens per second, and step-time or phase-time breakdown
- right rail: current phase, current subphase, checkpoint, validation, and
  device summary
- lower left: GPU utilization, memory, power, and temperature from retained
  samples
- lower middle: runtime pipeline view for dataloader, forward, backward,
  optimizer, collective, checkpoint, and eval time
- lower right: typed event feed, warnings, refusal posture, and
  source-artifact provenance

The pane should repaint every second while active even if the most recent panel
only changes in heartbeat and current-phase state.

Shared UI code that should be factored out instead of duplicated:

- the loss-curve painter from `apps/autopilot-desktop/src/panes/attnres_lab.rs`
- compact rails and ribbon helpers from the same file
- event-feed presentation patterns from both AttnRes and the Apple operator
  pane

## Recommended Projection Shape In `openagents`

Autopilot already has the right control-plane pattern in `desktop_control.rs`.

The new projection should look more like the AttnRes projection than the Apple
operator projection.

Why:

- AttnRes already projects a full metrics series into app state
- the Apple operator lane mostly projects terminal and recent-event detail, not
  a chart-ready time series

Recommended new app state:

- remote training run summary
- selected remote training run detail
- current heartbeat snapshot
- normalized metric points
- normalized math points
- normalized runtime points
- GPU sample points
- distributed sample points
- typed recent events
- provenance rows
- stale-state marker with last successful refresh time

Recommended new `desktop_control` surface:

- one summary endpoint for run lists
- one detail endpoint for a selected run
- one refresh action

That would also give `autopilotctl` a truthful text-mode twin of the same data.

## Required Psionic Work Before The App Can Be Honest

The UI work is not the long pole.

The long pole is artifact normalization and telemetry closure.

### Required item 1: define the visualization artifact contract

Without one canonical artifact, the app will accumulate one-off parsing rules
for:

- Google final manifests
- Google output-dir JSON files
- RunPod single-H100 audit JSON
- RunPod `8xH100` finalizer JSON
- raw `train.log`

That is the wrong boundary.

### Required item 2: add live one-second heartbeat emission

The current Google and RunPod retention paths are mostly finalize-time.

That fails the user's requirement directly.

Every active run needs a runtime-owned heartbeat that updates at least once per
second and records:

- current phase and subphase
- current progress marker
- current device and runtime activity
- latest math snapshot available so far
- last-known warning or stall reason when present

If a lane cannot support a one-second heartbeat, the app cannot honestly look
alive every second for that lane.

### Required item 3: close the per-step loss gap for non-PGOLF Google lanes

The current real Google `Psion` runs are strong operator proofs, but the audit
surface I found is still mostly summary-based.

If the goal is a real curve rather than a summary card, those lanes need one
typed per-step or per-checkpoint metric series.

### Required item 4: upgrade the RunPod single-H100 live writer and finalizer

The current finalizer preserves:

- the trainer JSON report when present
- the trainer log
- only the latest parsed micro-step in the audit JSON

That is good enough for postmortem status.

It is not the right contract for the app.

The live writer and finalizer should emit:

- one-second live heartbeat snapshots during the run
- the full series extracted from the trainer JSON when present
- a parsed log-derived fallback series only when the JSON is absent
- an explicit source label so the UI can distinguish authoritative metrics from
  fallback log parsing

### Required item 5: add a structured distributed series for RunPod `8xH100`

The current exported-folder lane preserves final run evidence, not a chart-ready
training trace.

That lane needs:

- a one-second coordinator heartbeat with distributed status
- a structured per-step or per-validation series emitted by the runtime, or
- a finalizer-side parser that turns the retained train log into a typed series

Until that lands, the app can show:

- provider
- topology
- digests
- final challenge metrics
- raw log tail

It cannot honestly show a full loss curve.

It also cannot honestly look alive every second during a live run until the
coordinator emits active distributed telemetry.

### Required item 6: add one run index surface

The app needs a cheap way to discover runs.

That can be:

- a local cache index generated by a sync tool, or
- a bucket-side manifest index for Google plus a mirrored local index for
  RunPod

The renderer should not be responsible for discovering runs by walking provider
storage roots.

## Suggested Implementation Tranche

### Tranche 1: ship a truthful first viewer for the lanes that are already close

Scope:

- define `remote_training_visualization_bundle.v1`
- add one-second heartbeat snapshots to the contract
- add a converter for PGOLF single-H100 trainer reports
- add a converter for Google final manifests plus GPU samples
- add a new Autopilot remote-training pane
- reuse the AttnRes loss-curve renderer

Result:

- Google runs render timeline, GPU, cost, heartbeat, and summary
- PGOLF single-H100 renders a real curve once a live run exists and the live
  writer flushes metrics during execution
- the UI refuses to invent curves or fake math panels for lanes that do not yet
  have them

### Tranche 2: close the non-PGOLF Google curve gap

Scope:

- extend the accelerated Google `Psion` outputs with one-second live snapshots
- extend the accelerated Google `Psion` outputs with a typed metric series
- retain it in the uploaded output directory
- add it to the visualization bundle

Result:

- the current real Google `Psion` lanes gain real loss curves and always-live
  status panels

### Tranche 3: close the RunPod single-H100 curve gap

Scope:

- add one-second live snapshots under the run output directory
- upgrade `scripts/parameter-golf-runpod-finalize-single-h100.sh`
- preserve the full trainer series
- add a local mirror step so Autopilot has one stable artifact location

Result:

- the first real RunPod single-H100 run can reuse the same pane immediately and
  stay visually active during execution

### Tranche 4: close the RunPod `8xH100` distributed curve gap

Scope:

- add one-second coordinator heartbeats
- add a structured distributed training series
- surface it through the `8xH100` finalizer

Result:

- the exported-folder distributed lane becomes always-live and chartable instead
  of only auditable

## What The App Must Not Do

The UI should not:

- wait for phase boundaries before repainting
- present a static pane for long stretches while a run is active
- infer missing curves from final metrics
- interpolate invented smooth curves across missing telemetry windows
- claim math diagnostics that the runtime did not actually export
- parse arbitrary provider logs directly in the WGPUI renderer
- rely on post-run finalizers as the only source for live viewing
- treat rehearsals and manifest-only launches as completed training runs
- collapse summary-only lanes and full-series lanes into one identical chart
- reuse `AttnResLabSnapshot` as if remote PGOLF or Google plugin-conditioned
  runs shared the same semantics
- dump raw tensors or per-parameter state every second when bounded summaries
  would carry the truth without distorting the run

The UI should instead:

- surface heartbeat freshness and explicit stale state
- surface explicit `series unavailable` states
- show provenance and source digests
- separate `summary truth` from `curve truth`
- read from a normalized live cache or live bundle rather than provider-shaped
  logs

## Bottom Line

The Autopilot app already has the WGPUI capability to do this.

The real work is to make remote training artifacts live, chartable, and
provider-neutral in one stable way.

The shortest honest path is:

1. add one visualization bundle contract in `psionic` with a one-second
   heartbeat
2. emit incremental live snapshots during Google and RunPod runs and seal them
   in finalizers
3. project a normalized local mirror through `desktop_control`
4. render it in one new Autopilot pane using the existing AttnRes charting
   components

If that happens, the first truly good remote-training visualization can land
without reworking WGPUI and it can stay visibly alive throughout a run.

If that does not happen, the app will still be able to show run lists, GPU
graphs, costs, and log tails, but it will not have one honest cross-provider
always-live training story.
