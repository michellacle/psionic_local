# PSION Plugin Cluster-Scale Decision

> Status: canonical `PSION_PLUGIN-33` decision record for whether
> plugin-conditioned training should widen from the first single-node proofs to
> a bounded trusted-cluster run, written 2026-03-22 after the host-native and
> mixed Google audits plus the route/refusal hardening tranche landed.

## Decision

Current decision: `not_warranted_yet`

The repo should **not** start a trusted-cluster plugin-conditioned training run
yet.

The current single-node proofs are real and useful, but they are still proof of
operator lane truth and evidence retention, not proof that cluster-scale
plugin-conditioned training would buy honest new capability today.

## Decision Inputs

This decision is based directly on:

- `docs/audits/2026-03-22-openagentsgemini-first-google-host-native-plugin-conditioned-run-audit.md`
- `docs/audits/2026-03-22-openagentsgemini-first-google-mixed-plugin-conditioned-run-audit.md`
- `docs/PSION_GOOGLE_SINGLE_GPU_RUNBOOK.md`
- `docs/PSION_PLUGIN_ROUTE_REFUSAL_HARDENING.md`
- `docs/PSION_TRUSTED_CLUSTER_RUN.md`
- `docs/PSION_RENTED_CLUSTER_RUNBOOK.md`

## What The Single-Node Proofs Established

The two Google audits did establish real bounded operator truth:

- one real host-native plugin-conditioned Google run exists
- one real mixed host-native plus guest-artifact Google run exists
- both runs preserved launch, input, stage, checkpoint-evidence, and final
  manifest truth
- both runs preserved the explicit capability boundary instead of widening into
  publication, universality, or arbitrary software claims
- the mixed run retained real class-specific guest benchmark, capability
  matrix, and served-posture artifacts
- the hardening tranche now freezes route/refusal regression rows, a zero-bps
  overdelegation budget, and explicit no-implicit-execution cases

That is enough to prove the single-node plugin-conditioned operator lane is
real.

It is not enough to justify cluster widening yet.

## Why Cluster Scale Is Not Warranted Yet

### 1. There is still no accelerator-backed plugin-conditioned throughput proof

Both Google plugin-conditioned audits stayed CPU-bound on L4 hosts:

- host-native audit: `102` GPU samples with `0` average and max utilization
- mixed audit: `98` GPU samples with `0` average and max utilization

That means the current proofs do **not** show a real accelerator-using
plugin-conditioned lane. Scaling that lane out to a trusted cluster would spend
more machines to prove something the current single-node path is not yet doing.

### 2. Cost truth is still partial

Both Google audits kept the same cost boundary explicit:

- the expected billing-export table was not present
- query-backed realized cost truth for the runs was therefore missing

Cluster-scale work without query-backed realized cost truth would widen spend
before the repo can honestly say what the single-node plugin-conditioned runs
actually cost.

### 3. The current learned lanes are still tiny bounded reference lanes

The current plugin-conditioned runs are still the bounded reference lanes:

- host-native training example count: `3`
- mixed training example count: `4`
- mixed guest-artifact training example count: `1`

Those are valid proof-sized lanes. They are not yet the kind of materially
larger accelerator-using corpus that would justify the operational complexity
of a trusted-cluster plugin-conditioned run.

### 4. The current cluster substrate is generic Psion truth, not plugin-specific need

The repo does already have generic trusted-cluster and rented-cluster
contracts.

But those documents are generic Psion cluster substrate truth. They do not, by
themselves, prove that plugin-conditioned training currently needs cluster
scale, or that cluster scale is the next honest frontier for the plugin lane.

### 5. The hardening tranche is a stability gate, not a scale-up green light

`docs/PSION_PLUGIN_ROUTE_REFUSAL_HARDENING.md` now makes the current
route/refusal posture machine-readable.

That is a necessary hardening step.

It is not a reason to widen to a trusted cluster. The hardening bundle freezes
today's bounded lane; it does not prove a larger accelerator-backed lane exists
yet.

## What Would Change This Decision

Cluster-scale plugin-conditioned training becomes worth reconsidering only after
all of the following are true:

1. One real single-node plugin-conditioned run uses the accelerator materially
   instead of staying CPU-bound, and the run retains throughput evidence plus
   non-zero GPU utilization.
2. Query-backed realized cost truth exists for the plugin-conditioned single-
   node lane through billing export or an equivalent machine-queryable surface.
3. The plugin-conditioned lane grows beyond the current proof-sized reference
   datasets into a materially larger curated corpus where single-node runtime is
   actually the bottleneck.
4. The larger single-node lane still retains the current checkpoint archive,
   cold-restore, hardening, capability-boundary, and no-implicit-execution
   posture.
5. One plugin-conditioned cluster launch profile exists with an explicit cost
   ceiling, input package, checkpoint durability posture, and abort conditions
   before any paid trusted-cluster launch.

## Explicit Non-Implications

This decision does **not** mean:

- the single-node plugin-conditioned work failed
- trusted-cluster Psion substrate is fake
- plugin-conditioned training can never justify cluster scale

It means only this:

- the next honest frontier for plugin-conditioned training is still a better
  single-node accelerator-using lane, not a trusted-cluster launch

## Follow-On Direction

The next plugin-conditioned execution priorities should be:

1. land the first accelerator-using plugin-conditioned single-node lane
2. close query-backed realized cost truth for that lane
3. rerun the cluster-scale decision only after those two facts are real

Until then, the default posture stays:

- no trusted-cluster plugin-conditioned run
- no widening from single-node proof to cluster-scale readiness by implication
