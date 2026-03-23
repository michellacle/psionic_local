# 2026-03-22 Psion Full System State And Operations Audit

This audit records the current end-to-end state of the `Psion` system in
`OpenAgentsInc/psionic` after the full `PSION-*` learned-model program, the
Google single-node infrastructure tranche, and the full `PSION_PLUGIN-*`
plugin-conditioned convergence tranche were implemented and closed.

It is written as a current-state audit, not as a new speculative plan.

## Executive Verdict

The system is now real in four important ways:

- the bounded `Psion` reference training lane is real
- the bounded Google single-node operator lane is real
- the bounded starter-plugin substrate is real
- the bounded plugin-conditioned learned lane is real

But the system is still intentionally narrow in the places that matter most:

- the current reference and plugin-conditioned runs are still proof-sized lanes
- the committed Google runs were real cloud executions, but they remained
  CPU-bound on L4 hosts
- cluster-scale plugin-conditioned training is correctly frozen at
  `not_warranted_yet`
- cost truth is still bounded by launch ceilings and retained run evidence
  rather than query-backed invoice truth for every run
- publication and capability posture remain deliberately operator-internal and
  non-universal

So the honest current statement is:

- `Psion` is no longer just a training design or schema program
- `Psion` now has real bounded training, plugin, and Google operator lanes
- `Psion` is not yet a broad accelerator-proved pretraining system

## Scope And Sources

This audit compares the current repo state against both the canonical current
docs and the earlier alpha planning inputs that shaped the work.

Primary current-state sources:

- `docs/TRAIN_SYSTEM.md`
- `docs/PSION_PROGRAM_MAP.md`
- `docs/PSION_PLUGIN_PROGRAM_MAP.md`
- `docs/PSION_GOOGLE_SINGLE_GPU_RUNBOOK.md`
- `docs/PSION_PLUGIN_CLUSTER_SCALE_DECISION.md`
- `docs/PSION_PLUGIN_ROUTE_REFUSAL_HARDENING.md`
- `docs/audits/2026-03-22-psion-training-system-full-state-audit.md`
- `docs/audits/2026-03-22-tassadar-full-plugin-system-state-audit.md`
- `docs/audits/2026-03-22-openagentsgemini-first-google-single-gpu-pilot-run-audit.md`
- `docs/audits/2026-03-22-openagentsgemini-first-google-host-native-plugin-conditioned-run-audit.md`
- `docs/audits/2026-03-22-openagentsgemini-first-google-mixed-plugin-conditioned-run-audit.md`

Historical comparison inputs:

- `~/code/alpha/training/initial-psionic-model-training-spec.md`
- `~/code/alpha/training/model-training-chat.md`
- `~/code/alpha/tassadar/plugin-system.md`
- `~/code/alpha/tassadar/TASSION.md`

Repository truth consulted directly:

- `crates/psionic-data/`
- `crates/psionic-train/`
- `crates/psionic-runtime/`
- `scripts/psion-google-*.sh`
- `fixtures/psion/`
- `fixtures/tassadar/`

GitHub state checked during this audit:

- `gh issue list --repo OpenAgentsInc/psionic --state open --limit 20`

Observed result:

- no open GitHub issues were returned

## What Was Implemented

The repo now carries two completed issue programs instead of one.

Completed learned-model and operator program:

- `PSION-1` through `PSION-45`

Completed plugin-conditioned convergence program:

- `PSION_PLUGIN-1` through `PSION_PLUGIN-33`

That work landed the following durable system surfaces.

### 1. Core Psion training substrate

The repo now owns real bounded versions of:

- corpus admission
- source lifecycle and removal lineage
- benchmark isolation and contamination controls
- raw-source ingestion
- tokenizer training manifests
- tokenized corpus manifests
- sampling-policy and code-token controls
- compact-decoder reference descriptors
- pretrain-stage receipts
- checkpoint lineage and recovery receipts
- route-class evaluation
- refusal calibration
- served evidence and claim posture

### 2. Reference-lane execution

The repo now has real executable bounded reference paths for:

- raw sources to tokenized reference corpus
- tokenized corpus to reference pilot bundle
- reference pilot to checkpoint archive
- archive to cold restore

The first real Google single-node run retained:

- run id: `psion-g2-l4-google-pilot-20260322t184426z`
- result class: `bounded_success`
- machine: `g2-standard-8`
- accelerator: `nvidia-l4 x1`

That run proved launch, bootstrap, archive, cold restore, final manifest, and
teardown truth on real Google infrastructure.

### 3. Plugin substrate and plugin-conditioned training lane

The repo now has a bounded but real starter-plugin platform and a bounded but
real plugin-conditioned learned lane on top of it.

Implemented starter-plugin and controller surfaces now include:

- starter-plugin runtime registry
- shared starter-plugin bridge
- deterministic workflow controller
- router-owned tool loop
- Apple FM plugin session lane
- canonical weighted controller trace
- starter-plugin catalog
- lane-neutral multi-plugin trace corpus

Implemented bounded plugin classes now include:

- host-native capability-free local deterministic
- one narrow manual `networked_read_only` proof
- one narrow digest-bound guest-artifact proof

Implemented plugin-conditioned learned surfaces now include:

- plugin training-record schema
- plugin trace derivation pipeline
- plugin-conditioned dataset builder
- plugin-aware contamination controls
- shared plugin benchmark packages
- host-native plugin-conditioned SFT lane
- mixed host-native plus guest-artifact plugin-conditioned lane
- bounded capability matrices and served posture docs
- route/refusal hardening bundle

### 4. Google operator system

The repo now owns the full bounded Google single-node operator lane:

- quota preflight
- input-package materialization
- launch manifests
- startup bootstrap
- host observability capture
- checkpoint archive
- cold restore
- final manifest and manifest-of-manifests
- teardown

The infra lane also now has:

- dedicated training bucket posture
- dedicated service account posture
- training-specific firewall and host tags
- launch-profile fixtures
- host-observability fixtures
- budget and price-profile posture
- dedicated runbook and audits

## How The System Operates Now

The current system is best understood as four connected operating lanes.

### Lane 1: Bounded reference pretraining

The reference learned-model lane now operates as:

1. admitted sources are tracked under explicit lifecycle and isolation rules
2. raw-source ingestion emits normalized artifacts
3. tokenizer-training and tokenized-corpus stages bind artifact identity
4. the reference pilot consumes the committed reference corpus
5. the pilot emits stage receipts, replay receipts, observability receipts, and
   checkpoint lineage
6. archive helpers package checkpoint and run artifacts
7. restore helpers replay the last stable checkpoint boundary

This lane is now truthful as a bounded reference system. It is not yet the
large curated accelerator-backed pretraining lane from the original ambition.

### Lane 2: Plugin runtime truth generation

The plugin substrate now operates as:

1. runtime registry defines plugin identity, capability class, schemas, replay
   class, bundle identity, and exposure posture
2. starter plugins execute through host-owned runtime code and emit typed
   receipts
3. the shared bridge projects the same plugin truth into controller surfaces
4. deterministic, router, Apple FM, and weighted-controller lanes all preserve
   the same receipt-linked tool truth
5. the starter catalog and weighted-controller traces freeze the admitted
   internal plugin set

This matters because the training lane no longer invents fake string tool calls
for supervision. It derives training truth from committed runtime receipts and
controller traces.

### Lane 3: Plugin-conditioned learned training

The plugin-conditioned learned lane now operates as:

1. multi-plugin controller traces are normalized into canonical training
   records
2. dataset builders preserve plugin ids, controller context, route labels,
   outcome labels, and receipt linkage
3. contamination controls keep train and held-out traces disjoint
4. benchmark packages test discovery, arguments, sequencing, refusal, result
   interpretation, and bounded guest capability posture
5. the bounded SFT stage trains a compact decoder against those plugin-aware
   records
6. evaluation receipts compare trained results to the named baseline or prior
   lane
7. served posture docs freeze what claims the learned lane may and may not make

The two current plugin-conditioned learned lanes are:

- host-native reference lane
- mixed host-native plus bounded guest-artifact reference lane

Both are real, but both are still tiny proof-sized lanes.

### Lane 4: Google-host execution and evidence retention

The Google operator lane now operates as:

1. preflight checks quota, launch profile, and cost ceiling posture
2. the repo packages immutable training inputs and launch manifests
3. a Google VM boots with the committed startup bundle
4. the host records timeline events, runtime facts, stdout or stderr, and GPU
   samples
5. the training or evaluation command runs from the repo-owned bundle
6. archive helpers upload checkpoint or logical checkpoint evidence
7. finalizers write one final manifest with per-object digests
8. teardown deletes the host after evidence is retained

That lane is now real for:

- generic bounded reference pilot execution
- host-native plugin-conditioned execution
- mixed plugin-conditioned execution

## What The Current Runs Actually Proved

The repo now has three important real Google-hosted run proofs.

### 1. Reference pilot proof

The first bounded reference pilot run proved:

- real Google VM allocation
- successful startup and bootstrap after several fixed live failures
- successful reference pilot execution
- checkpoint archive upload
- cold restore from the archived checkpoint boundary
- final manifest upload before teardown

Important retained metrics from the successful run:

- optimizer steps: `16`
- train tokens: `32768`
- validation tokens: `530`
- held-out tokens: `161`
- benchmark pass rates: `10000 bps` on architecture, normative-spec, held-out,
  route, and refusal receipts

### 2. Host-native plugin-conditioned proof

The host-native plugin-conditioned Google run proved:

- one real host-native plugin-conditioned run exists
- the run stayed inside the fully proved host-native capability-free class
- the run retained launch, stage, evaluation, archive, and teardown evidence

Important retained facts:

- run id: `psion-plugin-host-native-g2-l4-20260323t015231z`
- training example count: `3`
- learned plugin ids: `4`
- benchmark family count: `5`

### 3. Mixed plugin-conditioned proof

The mixed plugin-conditioned Google run proved:

- one real mixed host-native plus guest-artifact run exists
- the run retained class-specific guest capability-boundary evidence
- the mixed capability matrix stayed explicit about supported, blocked, and
  unsupported rows

Important retained facts:

- run id: `psion-plugin-mixed-g2-l4-20260323t021022z`
- training example count: `4`
- guest-artifact training examples: `1`
- supported capability rows: `9`
- blocked rows: `4`
- unsupported rows: `1`

## Hard Boundaries That Still Matter

The repo is stronger now mostly because it stopped pretending broad closure.

### 1. Accelerator truth is still missing

The reference pilot and both plugin-conditioned Google runs used real L4
machines, but all retained GPU summaries showed `0%` utilization and `0 MiB`
observed memory use.

That means:

- the cloud operator lane is proved
- the current bounded training commands are still CPU-bound
- the repo does not yet have accelerator-throughput proof for either the
  generic reference lane or the plugin-conditioned lane

### 2. Cluster-scale plugin-conditioned training is correctly blocked

The cluster decision is now explicitly frozen at `not_warranted_yet`.

That is the correct current posture because:

- the current single-node proofs are still CPU-bound
- realized cost truth is still partial
- the current plugin-conditioned datasets are still proof-sized

### 3. Plugin publication and universality remain blocked

The plugin system is real, but still intentionally narrow.

The current live posture is still:

- operator-internal
- publication-blocked
- non-marketplace
- non-universal
- non-arbitrary-software

The bounded guest-artifact lane does not change that boundary.

### 4. Cost truth is operational, not invoice-complete

The current Google lane has:

- declared run ceilings
- retained launch manifests
- retained observability
- run-specific evidence bundles

But later audits still correctly note missing machine-queryable billing-export
truth for the plugin-conditioned single-node runs. The repo can bound cost and
retain evidence, but it is not yet the final FinOps truth source for those
executions.

## Where The Current System Matches The Alpha Plans Well

The repo stayed surprisingly faithful to the original alpha intent.

Strong matches:

- explicit source and rights discipline instead of opportunistic scraping
- benchmark isolation and contamination control
- compact decoder before wider architecture ambition
- route, refusal, and claim publication as separate evidence surfaces
- learned-versus-runtime claim separation for plugin use
- bounded small pilot before any scale-up claim
- operator evidence retention instead of "it ran on my machine" inference

The plugin-conditioned tranche also preserved the central alpha law correctly:

- the learned model may choose, plan, sequence, and interpret
- the runtime still owns actual execution, capability mediation, and receipts

## Where The Current System Is Still Narrower Than The Original Ambition

The biggest remaining gaps are not conceptual anymore. They are execution and
scale gaps.

Current narrow points:

- no materially accelerator-using single-node pretraining lane yet
- no larger curated corpus run that makes the single-node host itself the
  bottleneck
- no query-backed realized cost truth for every Google training audit
- no cluster-scale plugin-conditioned justification yet
- no widened publication posture for plugin-conditioned capability claims

So the repo is now much closer to "truthful reference system" than to
"finished production-scale training stack."

## Current Operational Reading

If the question is "can the repo now run real bounded training-related work and
retain evidence honestly?", the answer is yes.

If the question is "is the repo ready to start spending on a broader
accelerator-backed training campaign?", the answer is not yet.

The current operating posture should be read as:

- yes for bounded single-node reference and plugin-conditioned proofs
- yes for real Google-hosted evidence-retaining execution
- yes for continued single-node accelerator-using bring-up work
- no for claiming GPU-efficient pretraining closure
- no for claiming cluster-scale plugin-conditioned readiness
- no for widening plugin publication, universality, or arbitrary software
  claims

## Recommended Next Steps

The next honest implementation tranche should be short and focused.

1. Land the first materially accelerator-using single-node `Psion` training
   lane and retain non-zero GPU utilization plus throughput evidence.
2. Move at least one bounded plugin-conditioned lane onto that same truthful
   accelerator-using path instead of the current CPU-bound reference bundle.
3. Close machine-queryable realized cost truth for the Google training lane so
   later audits can cite billing-backed spend rather than only declared
   ceilings.
4. Grow beyond the current proof-sized corpora into the first materially larger
   curated single-node run while preserving the current lineage, checkpoint,
   route, refusal, and contamination posture.
5. Revisit cluster-scale decisions only after steps 1 through 4 are real.

## Final Conclusion

The important shift is not that the system became broad. It is that the system
became honest.

`Psion` now has a real bounded training system, a real bounded plugin system, a
real bounded plugin-conditioned learned lane, and a real bounded Google
operator lane with retained evidence.

That is a substantial change from the original planning state.

The next job is no longer to invent the system. The next job is to move the
current truthful bounded lanes onto real accelerator-backed training paths
without losing the claim discipline that made the current system credible.
