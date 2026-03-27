# HOMEGOLF Fresh Rerun Results Audit

This audit records one fresh HOMEGOLF rerun on 2026-03-27 and compares it
against the currently committed HOMEGOLF and public Parameter Golf surfaces.

## What Ran

Command:

```bash
cargo run -q -p psionic-serve --example parameter_golf_homegolf_dense_bundle_proof -- \
  /tmp/psionic_homegolf_run_1774647888
```

This is the current honest runnable HOMEGOLF path:

- exact `parameter-golf-sp1024-9x512` family contract
- real bounded Psionic training run
- emitted promoted runtime bundle
- direct inference from the emitted bundle
- served inference through `psionic-serve`

It is not yet the full public contest scorepath. The emitted training config
still uses:

- `evaluation_identity = local_reference_validation`
- `contest_bits_per_byte_accounting_required = false`
- `exact_compressed_artifact_cap_required = false`

So this is a real HOMEGOLF rerun, but still a bounded exact-family proof lane,
not a public-leaderboard-equivalent submission.

## Fresh Run Result

The fresh rerun emitted:

- report:
  `/tmp/psionic_homegolf_run_1774647888/parameter_golf_homegolf_dense_bundle_proof.json`
- bundle dir:
  `/tmp/psionic_homegolf_run_1774647888/bundle`

Key results:

- descriptor digest:
  `8a111f908acf02174554a75e83e13092852ee7caa534f65c43be897bd4c606ee`
- tokenizer digest:
  `49b44264442058c20b2b95a947f3aac60e8729fd57d63db8b8754de8edb98a6d`
- model artifact bytes:
  `68,248,296`
- initial validation mean loss:
  `8.611438751220703`
- final validation mean loss:
  `8.60598874092102`
- final validation bits per byte:
  `9.93265382277841`
- int8-zlib roundtrip validation mean loss:
  `8.598987579345703`
- prompt:
  `abcd`
- direct output:
  `gbgb`
- served output:
  `gbgb`
- direct/served parity:
  `true`

Training-shape details from the emitted bundle:

- `max_steps = 2`
- `train_batch_tokens = 32`
- `validation_batch_tokens = 32`
- `train_sequence_length = 4`
- `grad_accum_steps = 8`
- `max_wallclock_seconds = 600`

## Reproducibility Result

The fresh rerun matched the committed canonical HOMEGOLF proof fixture exactly:

```bash
cmp -s \
  /tmp/psionic_homegolf_run_1774647888/parameter_golf_homegolf_dense_bundle_proof.json \
  fixtures/parameter_golf/reports/parameter_golf_homegolf_dense_bundle_proof.json
```

Result:

- exact byte-for-byte match

That means today's rerun did not drift from the currently committed HOMEGOLF
proof. The canonical tracked result remains:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_dense_bundle_proof.json`

## Comparison To Public Parameter Golf

Using the frozen HOMEGOLF public-comparison surface:

- HOMEGOLF `val_bpb`:
  `9.93265382277841`
- public naive baseline `val_bpb`:
  `1.2243657`
- current public best `val_bpb`:
  `1.1194`

Current deltas:

- versus public naive baseline:
  `+8.70828812277841 val_bpb`
- versus current public best:
  `+8.813253822778409 val_bpb`

Artifact posture:

- fresh HOMEGOLF scored model bytes:
  `68,248,296`
- HOMEGOLF counted code bytes:
  `7,188,700`
- HOMEGOLF total counted bytes:
  `75,436,996`
- contest cap:
  `16,000,000`
- current cap delta:
  `+59,436,996`

So the current HOMEGOLF rerun is still:

- far worse than the public baseline on `val_bpb`
- far over the artifact cap
- not a public-leaderboard-equivalent run

## What This Run Actually Proves

This fresh rerun proves that Psionic can currently do all of these together for
the exact HOMEGOLF family:

- train a real bounded `9x512` model
- emit a promoted runtime bundle
- reload that bundle for inference
- serve that same bundle through `psionic-serve`
- reproduce the same retained result deterministically

That is real progress, but it is still the bounded proof lane, not the final
HOMEGOLF destination.

## What It Still Does Not Prove

This rerun does not prove:

- exact FineWeb `SP1024` public scorepath closure
- contest-required BPB accounting on the live rerun lane
- contest-cap artifact compliance
- one live mixed-device dense HOMEGOLF run producing the scored bundle
- one H100-backed HOMEGOLF run
- one competitive public-baseline result

## Current Read

The honest current status is:

- reproducibility is good
- train-to-infer closure is real
- serve parity is real
- competitiveness is still poor

The system is now reliable enough to iterate on, but the next work must improve
three things at once:

- quality
- artifact size
- exact contest-path fidelity

## Most Important Next Moves

1. Make the live HOMEGOLF lane use the exact contest evaluation/accounting
   policy instead of `local_reference_validation`.
2. Get one real mixed-device dense HOMEGOLF run to produce the scored bundle,
   instead of composing device receipts with a separate exact-family proof.
3. Reduce artifact size aggressively so the HOMEGOLF counted-byte report can
   move from refusal toward a real pass.
4. Improve model quality on the exact-family lane before spending time on public
   leaderboard rhetoric.
