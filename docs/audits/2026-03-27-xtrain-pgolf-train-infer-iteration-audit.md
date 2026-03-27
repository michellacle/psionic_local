# XTRAIN To PGOLF Train/Infer Iteration Audit

Date: 2026-03-27

## Scope

This audit records the current working state of the promoted PGOLF-shaped
train-to-infer lane after pushing it far enough to:

- train one promoted bundle through the bounded XTRAIN local-reference path
- load the trained checkpoint as a promoted runtime bundle
- run greedy inference through both the direct runtime and `psionic-serve`
- benchmark the bounded-window runtime against the old full-history decode path

The intent was not to prove model quality on a real corpus. The intent was to
make the first XTRAIN-trained promoted model actually runnable through the
inference stack and to measure the first quality/throughput deltas honestly.

## Code Surfaces Changed

- `crates/psionic-train/src/parameter_golf_reference.rs`
  - added the stronger `xtrain_promoted_general_small_decoder_baseline()`
  - separated strict proof replay from the XTRAIN lane so resume divergence is
    recorded instead of treated as a hard failure in the stronger baseline
  - fixed promoted checkpoint-manifest normalization so bundle emission and
    receipt digest checks no longer drift
  - emitted the bounded inference-window contract into the promoted generation
    config

- `crates/psionic-models/src/parameter_golf_promoted_bundle.rs`
  - added `bounded_attention_window_tokens` to the promoted generation config
  - taught local bundle generation to use the bounded trailing attention window
    when the bundle declares it
  - removed the previous clone-heavy full-history path for the current runtime
    decode case

- `crates/psionic-serve/src/lib.rs`
  - taught the promoted PGOLF serving path to honor the same bounded trailing
    attention window as the direct runtime bundle path

- `crates/psionic-serve/examples/xtrain_parameter_golf_train_infer.rs`
  - added one end-to-end proof that trains proof and XTRAIN bundles side by side,
    runs inference, and benchmarks throughput
  - fixed the direct benchmark path to use the same greedy token selector as the
    real runtime so direct-runtime and served-runtime parity is measured honestly

## Final Working Configuration

The retained configuration is the cheaper bounded XTRAIN baseline, not the later
bigram-expanded experiment.

Retained XTRAIN baseline:

- `max_steps = 8`
- `train_batch_tokens = 16`
- `validation_batch_tokens = 16`
- `train_sequence_length = 4`
- `grad_accum_steps = 1`
- `bounded_attention_window_tokens = 4`

Why this is the retained operating point:

- it improves validation loss and bits-per-byte over the strict proof baseline
- it improves decode throughput materially
- it keeps runtime/serve parity
- it avoids the much slower bigram-expanded budget, which did not buy a real
  model-quality gain

## Verification Run

Primary reproduction command:

```bash
target/debug/examples/xtrain_parameter_golf_train_infer /tmp/psionic_xtrain_pgolf_eval_1774597215
```

Generated report:

- `/tmp/psionic_xtrain_pgolf_eval_1774597215/xtrain_parameter_golf_train_infer_report.json`

Observed completion line:

```text
xtrain PGOLF train->infer completed: report=/tmp/psionic_xtrain_pgolf_eval_1774597215/xtrain_parameter_golf_train_infer_report.json xtrain_loss=8.447444 direct_tps=24.89
```

## Measured Results

Proof baseline:

- validation loss: `8.60598874092102`
- validation BPB: `9.93265382277841`
- generated tokens for prompt `abcd`: `952,1005,951,900,884,862,1005,862`
- exact cycle match: `false`

Bounded XTRAIN baseline:

- validation loss: `8.447443962097168`
- validation BPB: `9.749668409844002`
- generated tokens for prompt `abcd`: `952,930,936,983,794,794,794,794`
- exact cycle match: `false`

Measured improvement over proof baseline:

- validation loss delta: `0.15854477882385254`
- BPB delta: `0.18298541293440707`

Throughput on the trained XTRAIN bundle:

- legacy full-history decode: `2.760617232674363 tok/s`
- current bounded direct runtime: `24.8937880452915 tok/s`
- served runtime: `24.859472926484305 tok/s`
- direct over legacy improvement: `801.7471799658189%`

Parity:

- direct runtime vs served runtime: `match = true`
- direct runtime vs legacy full-history decode: `match = false`

That last mismatch is expected. The retained bundle now carries an explicit
bounded trailing attention-window contract, so the old full-history decode path
is not the same model behavior anymore.

## Current Honest Conclusion

This lane now works end to end in the narrow technical sense:

- XTRAIN training runs
- a promoted PGOLF-shaped bundle is emitted
- the bundle loads for local inference
- the same bundle runs through `psionic-serve`
- the direct runtime and served runtime agree on emitted tokens
- the bounded runtime is much faster than the old legacy decode path

What is still not solved:

- the trained model does not yet emit the intended `abcd -> efghabcd` cycle
- exact prefix gain is still `0`
- the current finite-difference coordinate budget is good enough to improve loss
  but not good enough to make the tiny reference task qualitatively correct

## What Was Tried And Rejected

One follow-up experiment expanded the XTRAIN coordinate budget toward PGOLF-style
bigram and additional attention/MLP surfaces. That run completed, but it
roughly doubled the training time and only changed the final loss/BPB by a tiny
amount while still failing exact cycle inference. That experiment was not kept.

## Next Technical Gap

If the goal is not merely "train and run inference" but "train and emit the
correct next-token cycle," the next change should not be another blind linear
increase in finite-difference coordinates.

The next useful step is one of:

1. replace the current tiny finite-difference update surface with a more
   capable gradient-backed trainer for the promoted PGOLF lane
2. add a deliberately tiny overfit profile whose trainable surface is chosen to
   control the token-to-logit path more directly, with acceptance measured by
   exact cycle emission rather than only by loss/BPB
3. split "throughput proof" from "correctness proof" so the bounded-window
   runtime can remain fast while a separate tiny reference lane proves exact
   inferable behavior

Until one of those lands, the honest status is:

- infrastructure path: working
- runtime parity: working
- throughput improvement: working
- exact inferable toy-task correctness: not yet working
