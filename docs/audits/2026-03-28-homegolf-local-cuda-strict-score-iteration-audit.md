# HOMEGOLF Local-CUDA Score Claim Correction Audit

Date: 2026-03-28

## Summary

This audit corrects the earlier local `RTX 4080` HOMEGOLF "strict score"
language.

Follow-up review on 2026-03-28 established that the observed local
`6.80014851` BPB receipts were bounded local proxy evidence, not an actual
PGOLF score.

Two facts forced that correction:

- the public `parameter-golf` README still defines official scoring as
  evaluation on the full FineWeb validation split
- the current single-device trainer really does load that full validation split
  on the live local-CUDA lane

The earlier code fixes from this iteration remain valid and useful, but the
score claim boundary needed to be tightened.

## Code Changes That Still Stand

Files changed in this iteration:

- `crates/psionic-train/src/bin/parameter_golf_homegolf_single_cuda_train.rs`
- `crates/psionic-train/src/parameter_golf_reference.rs`
- `crates/psionic-train/src/parameter_golf_single_h100_training.rs`

Behavioral effect:

- `validation_eval_mode=sliding_window:64` now preserves the already-selected
  local-CUDA validation batch unless the previous value was still only the old
  default
- HOMEGOLF local-CUDA fit trials now reject invalid profiles like
  `grad_accum_steps=24`
- the trainer now persists the exported `.final_model.st` earlier and emits
  explicit closeout markers around artifact export, digesting, JSON encoding,
  and report finish

Build validation that passed:

```bash
cargo check -q -p psionic-train --bin parameter_golf_homegolf_single_cuda_train
```

The targeted local test attempt on this Mac is still blocked by an unrelated
existing arm64/macOS linker failure in the broader `psionic-train` test binary.

## Public Scoring Contract Re-Check

The public challenge README now matters more than the earlier local log
interpretation.

Verified in the local `parameter-golf` clone:

- `README.md` says validation "always runs on the full fineweb_val_* split"
- official challenge rhetoric still binds scores to FineWeb validation
  compression, not one bounded local validation slice

That means any local receipt derived from a smaller retained validation slice is
only a bounded proxy unless it is explicitly surfaced as such.

## Remote Reruns On `archlinux`

Machine used:

- `archlinux`
- `NVIDIA GeForce RTX 4080`
- dataset:
  `~/code/parameter-golf/data/datasets/fineweb10B_sp1024`
- tokenizer:
  `~/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`

### 1. Default Local HOMEGOLF Entry Point Is Not `10` Minute Honest

Command shape:

```bash
PSIONIC_PARAMETER_GOLF_HOMEGOLF_GRAD_ACCUM_STEPS=64 \
PSIONIC_PARAMETER_GOLF_HOMEGOLF_VALIDATION_BATCH_SEQUENCES=8 \
cargo run -q -p psionic-train --bin parameter_golf_homegolf_single_cuda_train -- \
  ~/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
  ~/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  /tmp/psionic_homegolf_runs/homegolf_strict_rerun_20260328_report.json \
  both \
  sliding_window:64 \
  legal_score_first_ttt:batch_sequences=4
```

Observed truth:

- the trainer started with `warmup_steps=20`
- after almost `12` minutes of process lifetime it was still only at
  `warmup_step:1/20`

That local default posture is not an honest `10` minute HOMEGOLF run on the
`4080`.

### 2. One-Step Local Fit Posture Is Real

Command shape:

```bash
PSIONIC_PARAMETER_GOLF_HOMEGOLF_GRAD_ACCUM_STEPS=64 \
PSIONIC_PARAMETER_GOLF_HOMEGOLF_VALIDATION_BATCH_SEQUENCES=8 \
cargo run -q -p psionic-train --bin parameter_golf_homegolf_single_cuda_train -- \
  ~/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
  ~/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  /tmp/psionic_homegolf_runs/homegolf_strict_1step_nottt_20260328_report.json \
  1 \
  both \
  sliding_window:64
```

Observed train-step truth from the live logs:

- `max_steps=1`
- `warmup_steps=0`
- `grad_accum_steps=64`
- `validation_batch_sequences=8`
- `mean_microbatch_loss=8.29200745`
- `train_time_ms=311252`

That is a real bounded one-step local-CUDA fit result and it is better than the
earlier observed local train-step runtime of `321606 ms`.

### 3. The Full Local Sliding-Window Scorer Is Not The Earlier `32k` Slice

Observed after that same one-step run:

- `final_validation_start sequences=60568 batch_sequences=8`
- `validation_batch_start ... batch=1/121136 ...`

So the current live trainer really is planning the full sliding-window
validation surface.

That is incompatible with the earlier `evaluated_tokens=32768` local receipt
that had been treated as a local "strict score."

### 4. The Default Legal TTT Overlay Is Also Not Local `10` Minute Honest

Observed on the 2026-03-28 rerun with
`legal_score_first_ttt:batch_sequences=4`:

- `chunk_tokens=32768`
- `epochs=3`
- `score_windows=969088`

That overlay is far too large for the current local `4080` HOMEGOLF lane to be
called a `10` minute qualifying score path.

## What The Earlier `6.80014851` Receipt Really Means

The earlier 2026-03-28 local receipt still has value, but only as bounded local
fit evidence:

- it proved the local runner no longer immediately refused or OOMed
- it proved a `g64 / v8` one-step posture was materially closer to workable
- it helped expose the post-score closeout blind spot that led to the new
  artifact/report progress markers

It does **not** currently support the stronger claim that a real local PGOLF
score was already retained on the home `4080`.

## Actual Full-Validation Score Truth Currently Retained

The real retained full-validation score still present in-repo is:

- `fixtures/parameter_golf/reports/parameter_golf_runpod_single_h100_first_real_training_report.json`

That report preserves:

- `evaluated_sequence_count=60568`
- `evaluated_token_count=62021632`
- `evaluated_byte_count=151080363`
- `final_validation_bits_per_byte=6.306931747817168`
- `compressed_model_bytes=4732744`

That is a real full-validation single-H100 result.

It is still not a public `8xH100` leaderboard-equivalent run, but it is the
current honest "actual PGOLF score" truth retained in this repo.

## Current Honest Boundary

What is true now:

- the local HOMEGOLF runner/trainer fit search is less misleading than before
- one bounded local one-step train posture is real on the `4080`
- the code now preserves the local fit profile correctly, rejects invalid
  batch geometry, and emits earlier closeout telemetry
- the repo still retains one real full-validation single-H100 score:
  `6.306931747817168`

What is not true:

- the home `4080` local lane does not currently have a retained official PGOLF
  score from the full FineWeb validation split
- the earlier local `6.80014851` receipt should not be treated as an actual
  PGOLF score
- the current local `legal_score_first_ttt` shorthand is not a viable
  `10` minute HOMEGOLF score path

## Next Moves

1. Add an explicit bounded local-proxy validation contract to the HOMEGOLF
   local lane and stop calling that posture a PGOLF score.
2. Keep actual score work anchored to the real full-validation runtime lanes:
   retained single-H100 and future `8xH100` distributed runs.
3. Only claim new PGOLF score improvements after the retained artifact/report
   pair covers the full validation split and the language model proof is bound
   to that same retained artifact.
