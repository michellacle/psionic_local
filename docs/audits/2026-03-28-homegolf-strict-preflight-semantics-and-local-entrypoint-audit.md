# HOMEGOLF Strict Preflight Semantics And Local Entrypoint Audit

Date: 2026-03-28

## Scope

This audit records the next HOMEGOLF integration correction after
`docs/audits/2026-03-28-homegolf-strict-challenge-path-identity-audit.md`.

That earlier fix closed fake basename lookalikes.

The next live `archlinux` review exposed a different mismatch:

- the strict HOMEGOLF lane still emitted `ready_to_execute` when the exact
  challenge inputs were present
- that same strict report still pointed operators at
  `parameter_golf_single_h100_train`
- the actual local HOMEGOLF exact operator path is
  `parameter_golf_homegolf_single_cuda_train`
- that real local trainer still refused the declared `600` second contract on
  the same machine after the first measured micro-step

That made the strict surface too optimistic even after path identity was fixed.

## Reproduced Mismatch

Machine:

- `archlinux`
- `NVIDIA GeForce RTX 4080`

Exact challenge inputs:

- dataset root:
  `/home/christopherdavid/code/parameter-golf/data/datasets/fineweb10B_sp1024`
- tokenizer path:
  `/home/christopherdavid/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`

Strict-lane probe before this fix returned:

- `disposition=ready_to_execute`
- `dataset_root_status=present_exact_named_path`
- `tokenizer_path_status=present_exact_named_path`
- command template:
  `cargo run -q -p psionic-train --bin parameter_golf_single_h100_train -- <dataset_root> <tokenizer_path> <training_report_path> both sliding_window:64 legal_score_first_ttt`

Fresh local HOMEGOLF trainer proof on the same exact inputs returned:

- `disposition=refused_wallclock_projection`
- `executed_steps=0`
- refusal subject:
  `homegolf_local_cuda.wallclock_projection`
- refusal detail:
  `HOMEGOLF local-CUDA single-device trainer projected optimizer step 1 to require about 2287424ms after 1/64 micro-steps, which exceeds the remaining 600000ms wallclock budget on this machine posture.`

That combination was wrong operationally:

- the strict lane was only proving exact-input preflight
- but the emitted language and command template looked like real local runtime
  readiness

## What Changed

`crates/psionic-train/src/parameter_golf_homegolf_strict_challenge.rs` now:

- renames the positive strict-lane disposition from `ready_to_execute` to
  `preflight_satisfied`
- keeps the positive state limited to exact strict-input preflight rather than
  host-runtime readiness
- points the retained HOMEGOLF command template at
  `parameter_golf_homegolf_single_cuda_train`
- records the canonical local HOMEGOLF exact trainer entrypoint directly in the
  report
- updates the strict-lane claim boundary and summary so they no longer imply
  that the current host already satisfies the `600` second runtime contract

`crates/psionic-train/src/parameter_golf_homegolf.rs` now updates the HOMEGOLF
track evidence for the strict surface to the new audit language.

`docs/HOMEGOLF_TRACK.md` now describes this surface as a strict preflight lane,
not a runnable exact-score proof.

## Why This Matters

The earlier path-identity fix answered:

- are these the real challenge inputs

This fix answers the next operator question:

- does strict-input success already mean this host is ready for a real local
  HOMEGOLF exact run

The answer is no.

After this change:

- strict input identity and strict runtime viability are no longer conflated
- the retained operator command points at the actual HOMEGOLF local entrypoint
- the strict lane stops advertising `ready_to_execute` on machines that still
  need the runtime path itself to admit or refuse honestly

## Validation

Local validation:

- `rustfmt crates/psionic-train/src/parameter_golf_homegolf_strict_challenge.rs`
- `rustfmt crates/psionic-train/src/parameter_golf_homegolf.rs`
- `cargo test -q -p psionic-train strict_challenge_lane_ --manifest-path /Users/christopherdavid/work/psionic/Cargo.toml`
- `scripts/check-parameter-golf-homegolf-strict-challenge-lane.sh`
- `scripts/check-parameter-golf-homegolf-track-contract.sh`

Live validation:

- one fresh `archlinux` strict-lane probe on exact challenge inputs before this
  fix
- one fresh `archlinux` local HOMEGOLF exact trainer rerun on the same exact
  inputs, producing the retained wallclock refusal

## Honest Boundary After This Audit

What is true:

- the strict HOMEGOLF surface is now explicit strict-input preflight only
- the emitted operator command now uses the local HOMEGOLF exact trainer path
- the local `RTX 4080` exact lane still refuses the declared `600` second
  contract honestly
- the retained actual full-validation PGOLF score is still
  `6.306931747817168`

What is not true:

- this did not improve the retained PGOLF score
- this did not make the local `RTX 4080` exact lane viable
- this did not create a new mixed-device HOMEGOLF scored bundle
- XTRAIN did not change in this audit

## Improvement Over The Previous Audit

Compared with the strict path-identity audit:

- that change made fake inputs stop looking valid
- this change makes real exact inputs stop looking like automatic local runtime
  readiness
