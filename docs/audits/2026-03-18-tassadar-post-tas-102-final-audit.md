# Tassadar Post-TAS-102 Final Audit

## Scope

This audit closes the public `TAS-084` through `TAS-102` issue tranche.
It checks the repo state after landing `TAS-102` on `main`, confirms the
current broad internal-compute publication posture, and records what still is
not honestly claimable.

## Commands Run

- `gh issue list --repo OpenAgentsInc/psionic --state open --limit 200 --label tassadar`
- `cargo run -p psionic-eval --example tassadar_broad_internal_compute_profile_publication_report`
- `cargo run -p psionic-router --example tassadar_broad_internal_compute_route_policy_report`
- `cargo run -p psionic-eval --example tassadar_internal_compute_profile_ladder_report`
- `cargo run -p psionic-research --example tassadar_internal_compute_profile_ladder_summary`
- `cargo run -p psionic-eval --example tassadar_broad_internal_compute_acceptance_gate`
- `cargo test -p psionic-models internal_compute_profile_ladder -- --nocapture`
- `cargo test -p psionic-eval broad_internal_compute_profile -- --nocapture`
- `cargo test -p psionic-eval broad_internal_compute_acceptance_gate -- --nocapture`
- `cargo test -p psionic-eval internal_compute_profile_ladder_report -- --nocapture`
- `cargo test -p psionic-router broad_internal_compute_route -- --nocapture`
- `cargo test -p psionic-serve broad_internal_compute_profile -- --nocapture`
- `cargo test -p psionic-serve executor_service_capability_publication_serializes_benchmark_gated_matrix -- --nocapture`
- `cargo check -p psionic-provider`
- `cargo test -p psionic-provider broad_internal_compute_profile_publication -- --nocapture`
- `cargo test -p psionic-provider tassadar_capability_envelope -- --nocapture`

## Audit Result

### 1. Open-issue state

`gh issue list --state open --label tassadar` returned no rows.

That means the current public Tassadar issue queue is closed through
`TAS-102`.

### 2. Current served internal-compute claim

The current served profile is still:

- `tassadar.internal_compute.article_closeout.v1`

The committed ladder report at
`fixtures/tassadar/reports/tassadar_internal_compute_profile_ladder_report.json`
now records:

- `implemented_profiles=2`
- `planned_profiles=6`
- `current_claim_green=true`

This is the right posture. The repo still names one current served claim and
does not flatten the broader ladder into a generic `supports Rust/Wasm`
statement.

### 3. Broad publication posture

The committed broad acceptance gate at
`fixtures/tassadar/reports/tassadar_broad_internal_compute_acceptance_gate.json`
now records:

- `green_profiles=1`
- `suppressed_profiles=2`
- `failed_profiles=5`
- `overall_green=false`

The committed broad publication report at
`fixtures/tassadar/reports/tassadar_broad_internal_compute_profile_publication_report.json`
matches that posture:

- published: `tassadar.internal_compute.article_closeout.v1`
- suppressed: `tassadar.internal_compute.generalized_abi.v1`,
  `tassadar.internal_compute.public_broad_family.v1`
- failed: `tassadar.internal_compute.deterministic_import_subset.v1`,
  `tassadar.internal_compute.portable_broad_family.v1`,
  `tassadar.internal_compute.resumable_multi_slice.v1`,
  `tassadar.internal_compute.runtime_support_subset.v1`,
  `tassadar.internal_compute.wider_numeric_data_layout.v1`

The important tightening from `TAS-102` is that
`public_broad_family.v1` is no longer blocked by an `issue://182` placeholder.
It now has real publication artifacts and stays suppressed for the correct
reason: it still needs broader mount-policy and accepted-outcome closure,
not a missing issue stub.

### 4. Route policy posture

The committed route policy at
`fixtures/tassadar/reports/tassadar_broad_internal_compute_route_policy_report.json`
now records:

- `selected_routes=1`
- `suppressed_routes=2`
- `refused_routes=5`

The one selected route is the current article-closeout profile.

The two suppressed routes are:

- `generalized_abi.v1`
- `public_broad_family.v1`

That is the correct split:

- `generalized_abi.v1` is benchmarked and named, but still not promoted into
  the current served/public route posture.
- `public_broad_family.v1` is now nameable with real publication artifacts, but
  still requires profile-specific world-mount and accepted-outcome closure.

The five failed profiles remain explicit route refusals instead of being
silently widened.

### 5. Served and provider propagation

`TassadarExecutorCapabilityPublication` now carries the broad profile
publication object and route-policy ref, and the provider envelope now rejects
publication drift across:

- current served profile id
- published-profile membership
- route-policy ref presence
- agreement with the served internal-compute claim id

That means the selected profile id is now preserved end to end through the
served and provider publication surfaces.

## What Is Still Not True

After `TAS-102`, the repo still does **not** honestly claim:

- broad public internal-compute execution
- portable broad-family closure
- resumable multi-slice public route promotion
- deterministic-import public route promotion
- wider numeric/data-layout public route promotion
- settlement-ready or market-ready broad-profile closure

The only currently selected served internal-compute profile remains the
Rust-only article closeout lane.

## Verdict

The public Tassadar queue is closed through `TAS-102`, and the repo is now in
the right post-substrate posture:

- one current served profile
- explicit broader named profiles
- explicit suppressed versus refused route policy
- explicit world-mount and accepted-outcome dependency boundaries

That is a credible stopping point for the current public tranche. Any next
phase should start from new issues, not implicit widening of the current claim
surface.
