# HOMEGOLF Tailrun Daily Loop Operator Fix Audit

Date: 2026-03-28

## Scope

This audit records the next HOMEGOLF integration pass after the earlier strict
preflight and local-wallclock honesty fixes.

The new target was narrower and more operational:

- make the currently reachable consumer home-cluster daily loop actually run
- keep the retained H100-backed HOMEGOLF dense surfaces honest about what they
  are and what they are not

The reachable device set for this pass was:

- local `macbook-pro-m5`
- remote `archlinux` with `NVIDIA GeForce RTX 4080`

There was still no reachable H100-class Tailnet node.

## What Was Broken

`scripts/run-tailrun-daily-loop.sh` depended on
`scripts/run-open-adapter-tailnet-matrix.sh` for the admitted-device matrix.

Fresh live reruns exposed four real operator breakages in that matrix runner:

1. the remote staging step archived the whole repo into remote `/tmp`, which
   failed with `Disk quota exceeded`
2. the remote build target lived under `/tmp/psionic-tailrun-target`, which
   also hit quota during `cargo build`
3. `rustc` still spilled temporary files under remote `/tmp`, which hit quota
   even after the target path was moved
4. the remote benchmark wrote its retained output bundle under `/tmp`, which
   hit quota again after the benchmark itself had already started

Those were operator-path bugs, not model-quality results.

## What Changed

In `scripts/run-open-adapter-tailnet-matrix.sh`:

- remote staging now archives only the subset needed by the benchmark instead
  of archiving the whole repo
- remote build outputs now live under persistent `~/code/psionic/target`
  storage instead of remote `/tmp`
- remote `TMPDIR` now also lives under persistent
  `~/code/psionic/target/tmp`
- remote retained benchmark outputs now live under
  `~/code/psionic/target/tailrun-runs`
- the runner now removes the remote staged worktree and copied-back output root
  after success
- the remote target cache is now configured for a shared
  `tailrun-matrix-shared` directory so later reruns do not cold-build from a
  new `run_id` path every time

In the HOMEGOLF retained generators and docs:

- the retained dense HOMEGOLF surface, runtime, comparison, and accounting
  language now says `H100-backed` explicitly where that was the true source
- that matters because the retained dense mixed-device surface is still:
  - one local MLX rank
  - one optional-H100 CUDA submesh
  - not the currently reachable Apple-plus-home-RTX exact score lane

Affected HOMEGOLF files:

- `crates/psionic-train/src/parameter_golf_homegolf_clustered.rs`
- `crates/psionic-train/src/parameter_golf_homegolf_score_runtime.rs`
- `crates/psionic-train/src/parameter_golf_homegolf_comparison.rs`
- `crates/psionic-train/src/parameter_golf_homegolf_accounting.rs`
- `crates/psionic-train/src/parameter_golf_homegolf.rs`
- `docs/HOMEGOLF_TRACK.md`

Minor cleanup:

- `crates/psionic-train/src/bin/open_adapter_same_node_wallclock_benchmark.rs`
  no longer seeds `final_mean_loss` with a throwaway assignment before
  overwriting it

## Live Proof

The first fully successful admitted-device daily loop after these fixes was:

- run id: `tailrun-daily-20260328e`
- root: `/tmp/psionic_tailrun_daily_20260328e`

Retained outputs:

- matrix report:
  `/tmp/psionic_tailrun_daily_20260328e/matrix/matrix_report.json`
- quality report:
  `/tmp/psionic_tailrun_daily_20260328e/quality/quality_report.json`
- near-equivalent report:
  `/tmp/psionic_tailrun_daily_20260328e/near_equivalent/near_equivalent_report.json`
- daily scoreboard:
  `/tmp/psionic_tailrun_daily_20260328e/daily_scoreboard.json`

Retained scoreboard verdict:

- overall: `throughput_improved`
- M5 throughput: `meaningful_improvement`
- RTX 4080 throughput: `meaningful_improvement`
- held-out quality: `noise_band`
- near-equivalent bridge: `passed`

Matrix highlights:

- local M5:
  - `steps_per_second=309.8688123141508`
  - `completed_steps=177152`
- remote RTX 4080:
  - `steps_per_second=122.39022624246276`
  - `completed_steps=70656`
- local over remote steps gain:
  `153.18101112116676%`

Baseline comparison from the retained scoreboard:

- M5 steps gain versus baseline:
  `90.65258617541282%`
- RTX 4080 steps gain versus baseline:
  `48.527284726685025%`
- best held-out loss improvement versus baseline:
  `0%`

Near-equivalent closeout:

- expected target token: `37`
- direct predicted token: `37`
- served overlay token: `37`
- verdict: `passed`

## Why This Matters

Before this change, the reachable HOMEGOLF-adjacent home-cluster loop still
failed in environment setup and never reached a retained admitted-device daily
scoreboard.

After this change:

- the reachable local-plus-remote home-cluster loop now completes end to end
- the retained scoreboard now shows real throughput improvement against the
  earlier retained baseline
- the held-out quality surface remains honest instead of claiming a gain that
  did not happen
- the infer/serve bridge still closes on the retained M5 artifact
- the retained H100-backed HOMEGOLF dense surfaces are less likely to be read
  as if they already proved consumer home-cluster exact score closure

## Honest Boundary

What is true:

- the reachable admitted-device daily loop is now real and retained
- the current retained daily loop improved throughput materially over the prior
  retained baseline
- the current retained daily loop still keeps quality flat on the held-out
  PGOLF-ish comparison

What is not true:

- this is not an exact HOMEGOLF score
- this is not an exact Parameter Golf score
- this did not produce one mixed-device promoted runtime bundle directly from
  the admitted home cluster
- this did not improve the retained real full-validation PGOLF score, which is
  still `6.306931747817168`

## Validation

Script validation:

- `bash -n scripts/run-open-adapter-tailnet-matrix.sh`

Live validation:

- one full retained admitted-device daily loop:
  `tailrun-daily-20260328e`

Fixture and contract validation for the HOMEGOLF wording change is recorded in
the matching root-level audit for this iteration.
