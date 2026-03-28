# HOMEGOLF Local Competitive Override Lane Audit

Date: 2026-03-28

## Summary

This audit records the next HOMEGOLF integration pass after the earlier
2026-03-28 local-loop correction.

The retained exact competitive lane already exists as
`competitive_homegolf_v1`, but the local `RTX 4080` operator path was still too
rigid:

- the competitive defaults always enabled legal score-first TTT
- the final exported surface stayed pinned to `swa`
- the default SWA cadence stayed at `every_steps=50`

Those defaults are legal and useful for the exact challenge lane, but they are
not good operator defaults for short honest local iteration on a `600` second
budget. On a local `4080`, that combination can spend too much of the loop on
evaluation-style adaptation or export a weakly sampled SWA surface.

This pass does not claim a better retained PGOLF score yet. It makes the local
competitive loop tunable without patching source between runs.

## Code Changes

Files changed:

- `crates/psionic-train/src/bin/parameter_golf_homegolf_single_cuda_train.rs`
- `crates/psionic-train/src/parameter_golf_single_h100_training.rs`

Behavioral changes:

- `parameter_golf_homegolf_single_cuda_train` now accepts:
  - `PSIONIC_PARAMETER_GOLF_DISABLE_SCORE_FIRST_TTT=1`
  - `PSIONIC_PARAMETER_GOLF_FINAL_MODEL_SURFACE=raw|ema|swa`
  - `PSIONIC_PARAMETER_GOLF_SWA_EVERY_STEPS=<u64>`
- `ParameterGolfFinalModelSurface` now has a shared parser so the entrypoint can
  validate those override labels without duplicating config law
- the HOMEGOLF local CUDA runner now prints the chosen `model_variant`,
  `final_model_surface`, EMA posture, and SWA posture directly to stdout after
  the report is written

That last change matters because long local score passes can run for hours after
training. The live stdout now exposes the export posture immediately instead of
forcing a mid-run JSON read.

## Validation

The following validations passed locally:

```bash
cargo check -q -p psionic-train --bin parameter_golf_homegolf_single_cuda_train
cargo check -q -p psionic-train --bin parameter_golf_homegolf_prompt
cargo test -q -p psionic-train final_model_surface_parse_accepts_supported_labels -- --nocapture
cargo test -q -p psionic-train homegolf_local_cuda_defaults_drop_warmup_but_keep_wallclock_cap -- --nocapture
```

## Why These Overrides Matter

The retained HOMEGOLF exact-lane ablation says the best-known exact competitive
shape is `competitive_homegolf_v1`, with EMA, SWA sourced from EMA, and legal
score-first TTT.

That exact contract still stands.

But the local operator problem is different:

- legal score-first TTT expands the eval-side work materially on the `4080`
- the local loop only has `600` seconds of honest training time
- `swa.every_steps=50` can be too sparse to make the final `swa` export surface
  the best local answer in short runs

So the new local override contract is:

- keep the exact competitive model variant when it helps training quality
- disable score-first TTT when the goal is a practical local training loop
- choose `raw`, `ema`, or `swa` explicitly for the final exported surface
- tighten SWA cadence when the local run is too short for the default `50`-step
  cadence to collect enough samples

## Operational Follow-On

The first clean-worktree relaunch on `archlinux` exposed one separate operator
issue:

- the clean worktree was created under `/tmp`
- Cargo also built its target tree under that same `/tmp` worktree
- the run failed during cold build with `Disk quota exceeded`

So future clean-worktree HOMEGOLF runs on `archlinux` should keep only reports
and transient artifacts under `/tmp` while reusing a persistent target cache,
for example `CARGO_TARGET_DIR=~/code/psionic/target`.

That is an operator fix, not a trainer semantics fix.

## Improvement Over The Earlier 2026-03-28 HOMEGOLF Audit

Compared with the earlier local-loop audit, this pass improves the system in
two concrete ways:

1. The competitive HOMEGOLF local lane no longer requires source edits to try a
   score-practical no-TTT or EMA-final variant.
2. The live run log now makes the exact export surface explicit, which is
   necessary for honest long-running local score attempts.

The retained actual PGOLF score is still unchanged after this pass. The value
here is that the next competitive 10-minute HOMEGOLF reruns can now iterate on
quality with explicit posture control instead of silently inheriting one rigid
local default.
