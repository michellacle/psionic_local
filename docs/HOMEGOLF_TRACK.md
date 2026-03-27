# HOMEGOLF Track

> Status: canonical HOMEGOLF benchmark-track contract, updated 2026-03-27
> after landing `HOMEGOLF-0`.

This document freezes the first honest Psionic answer to:

- how to compete against Parameter Golf from a custom clustered home setup
- without pretending the hardware is the same as the public leaderboard

## What HOMEGOLF Means

HOMEGOLF keeps as much of the public Parameter Golf contract as possible:

- exact dense `9x512` baseline family as the initial target
- exact FineWeb `SP1024` identity in strict mode
- exact `10` minute wallclock cap
- exact tokenizer-agnostic `val_bpb` score reporting
- exact contest-style artifact-byte accounting toward the decimal `16MB` cap
- required train-to-infer bundle closure

HOMEGOLF changes one thing only:

- the compute posture is a declared custom cluster instead of public `8xH100`

So the honest reading is:

- HOMEGOLF can be compared against the public baseline and leaderboard
- HOMEGOLF is not a public leaderboard-equivalent result unless rerun under the
  official `8xH100` posture

## Canonical Contract

Machine-readable contract:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_track_contract.json`

Generator:

- `crates/psionic-train/src/parameter_golf_homegolf.rs`
- `crates/psionic-train/src/bin/parameter_golf_homegolf_track_contract.rs`

Checker:

- `scripts/check-parameter-golf-homegolf-track-contract.sh`

## Current Baseline Surfaces

The HOMEGOLF contract deliberately binds to existing Psionic surfaces instead of
inventing a second Parameter Golf stack:

- exact dense trainer entrypoint:
  `crates/psionic-train/src/bin/parameter_golf_single_h100_train.rs`
- exact dense trainer implementation:
  `crates/psionic-train/src/parameter_golf_single_h100_training.rs`
- promoted runtime bundle handoff:
  `crates/psionic-models/src/parameter_golf_promoted_bundle.rs`
- strict family contract:
  `docs/PARAMETER_GOLF_PROMOTED_FAMILY_CONTRACT.md`
- contest-style artifact accounting:
  `docs/PARAMETER_GOLF_NON_RECORD_SUBMISSION.md`

## Comparison Policy

Allowed wording:

- `public-baseline comparable`
- `not public-leaderboard equivalent`

Forbidden shortcut:

- calling a HOMEGOLF result a public leaderboard-equivalent score without
  rerunning it under the official `8xH100` execution posture

## Hardware Manifest Law

The HOMEGOLF contract admits one mixed-device cluster family:

- local Apple Silicon Metal workstation
- home consumer CUDA node
- secondary Apple Silicon peer when stable
- optional future H100 nodes

The hardware mix may change over time.

The score semantics may not.

That means:

- same model family
- same scorepath
- same wallclock budget
- same artifact accounting
- richer declared hardware manifest

## Current Honest Boundary

HOMEGOLF is frozen as a contract now, but one important surface is still
blocked:

- the exact dense PGOLF baseline is still H100-only in its live trainer
  execution surface

So HOMEGOLF is ready to guide implementation, but it does not yet claim that
the dense strict-baseline run can already execute across the full admitted home
cluster.
