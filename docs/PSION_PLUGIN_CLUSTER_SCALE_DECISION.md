# PSION Plugin Cluster-Scale Decision

> Status: canonical `PSION_PLUGIN-33` decision record for whether
> plugin-conditioned training should widen from the first single-node proofs to
> a bounded trusted-cluster run, written 2026-03-22 after the host-native and
> mixed Google audits plus the route/refusal hardening tranche landed, then
> updated 2026-03-23 after the first generic and host-native accelerated
> single-node Google audits closed.

## Decision

Current decision: `not_warranted_yet`

The repo should **not** start a trusted-cluster plugin-conditioned training run
yet.

The current single-node proofs are real and useful, but they are still proof of
operator lane truth and evidence retention, not proof that cluster-scale
plugin-conditioned training would buy honest new capability today.

## Decision Inputs

This decision is based directly on:

- `docs/audits/2026-03-23-openagentsgemini-first-google-accelerator-backed-single-node-psion-training-audit.md`
- `docs/audits/2026-03-23-openagentsgemini-first-google-accelerator-backed-host-native-plugin-conditioned-run-audit.md`
- `docs/audits/2026-03-22-openagentsgemini-first-google-host-native-plugin-conditioned-run-audit.md`
- `docs/audits/2026-03-22-openagentsgemini-first-google-mixed-plugin-conditioned-run-audit.md`
- `docs/PSION_GOOGLE_SINGLE_GPU_RUNBOOK.md`
- `docs/PSION_PLUGIN_ROUTE_REFUSAL_HARDENING.md`
- `docs/PSION_TRUSTED_CLUSTER_RUN.md`
- `docs/PSION_RENTED_CLUSTER_RUNBOOK.md`

## What The Single-Node Proofs Established

The current Google audits now establish real bounded single-node truth:

- one real generic accelerator-backed single-node `Psion` run exists
- one real host-native plugin-conditioned accelerator-backed Google run exists
- one real mixed host-native plus guest-artifact Google run exists
- all runs preserved launch, input, stage, checkpoint-evidence, and final
  manifest truth
- all runs preserved the explicit capability boundary instead of widening into
  publication, universality, or arbitrary software claims
- the mixed run retained real class-specific guest benchmark, capability
  matrix, and served-posture artifacts
- the mixed run still remained CPU-bound, so mixed guest-artifact acceleration
  is not yet proved
- the hardening tranche now freezes route/refusal regression rows, a zero-bps
  overdelegation budget, and explicit no-implicit-execution cases

That is enough to prove the bounded single-node accelerator lane is real for
generic `Psion` and for the proved host-native plugin-conditioned class.

It is not enough to justify cluster widening yet.

## Why Cluster Scale Is Not Warranted Yet

### 1. Accelerated single-node proof is real, but only in the narrow host-native form

The March 23 accelerated audits did close the first honest single-node
accelerator proof for:

- the generic `Psion` lane
- the host-native plugin-conditioned lane

But they did **not** close mixed guest-artifact acceleration:

- the mixed Google audit is still CPU-bound
- the mixed guest-artifact lane still has only one bounded guest training
  example
- widening from accelerated host-native proof to accelerated mixed or
  guest-artifact cluster claims would outrun the current evidence

So the single-node accelerator prerequisite is now satisfied in the narrow
generic and host-native form, but not in the mixed guest-artifact form that a
stronger cluster-scale plugin claim would naturally invite.

### 2. Cost truth is still partial

Both Google audits kept the same cost boundary explicit:

- the expected billing-export table was not present
- query-backed realized cost truth for the runs was therefore missing

Cluster-scale work without query-backed realized cost truth would widen spend
before the repo can honestly say what the single-node plugin-conditioned runs
actually cost.

### 3. The current learned lanes are still tiny bounded lanes

The current plugin-conditioned runs are still proof-sized bounded lanes:

- host-native training example count: `3`
- host-native accelerated optimizer steps: `8`
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

1. Query-backed realized cost truth exists for the generic and host-native
   accelerated single-node lanes through billing export or an equivalent
   machine-queryable surface.
2. The mixed guest-artifact lane is either kept explicitly out of any
   scale-up claim or lands its own accelerated single-node proof with the same
   backend, utilization, and residency gates as the host-native accelerated
   lane.
3. The plugin-conditioned lane grows beyond the current proof-sized datasets
   into a materially larger curated corpus where single-node runtime is
   actually the bottleneck.
4. The larger single-node lane still retains the current checkpoint archive,
   hardening, capability-boundary, and no-implicit-execution posture.
5. One plugin-conditioned cluster launch profile exists with an explicit cost
   ceiling, input package, checkpoint durability posture, and abort conditions
   before any paid trusted-cluster launch.

## Explicit Non-Implications

This decision does **not** mean:

- the single-node plugin-conditioned work failed
- trusted-cluster Psion substrate is fake
- plugin-conditioned training can never justify cluster scale
- the host-native accelerated proof silently widens the mixed guest-artifact
  lane into cluster-ready training

It means only this:

- the next honest frontier for plugin-conditioned training is still better
  single-node cost truth plus a clear mixed-lane acceleration decision, not a
  trusted-cluster launch

## Follow-On Direction

The next plugin-conditioned execution priorities should be:

1. close query-backed realized cost truth for the generic and host-native
   accelerated single-node lanes
2. decide whether to keep mixed guest-artifact training explicitly bounded or
   land a real accelerated mixed lane
3. grow the plugin-conditioned corpus beyond the current proof-sized bounded
   datasets
4. rerun the cluster-scale decision only after those three facts are real

Until then, the default posture stays:

- no trusted-cluster plugin-conditioned run
- no widening from single-node proof to cluster-scale readiness by implication
