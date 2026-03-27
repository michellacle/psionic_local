# XTRAIN PGOLF Runtime Allocation Tuning Audit

Date: 2026-03-27

## Purpose

This audit records the first runtime-only tuning pass after the retained
XTRAIN-promoted PGOLF quality tune.

The prior retained state already had:

- working XTRAIN training
- working promoted PGOLF bundle emission
- working local inference
- direct-runtime and served-runtime parity
- tuned local-reference quality

This pass did not change model weights, trainable surfaces, or decode semantics.
It only reduced avoidable allocation churn during decode.

## Code Changes

Changed files:

- `crates/psionic-models/src/parameter_golf_promoted_bundle.rs`
- `crates/psionic-serve/src/lib.rs`
- `crates/psionic-serve/examples/xtrain_parameter_golf_train_infer.rs`

What changed:

- preallocated `history` to fit prompt plus requested completion budget
- preallocated `generated_tokens` to the requested output budget
- preallocated `bounded_history` to the bounded attention-window size
- applied the same cheaper allocation posture in the direct benchmark harness so
  the reported `current_runtime` measurement matches the retained decode path

No decode strategy changes were introduced:

- same bounded trailing attention window
- same greedy token selection
- same direct-runtime vs served-runtime parity target

## Verification

Build:

```bash
cargo build -q -p psionic-serve --example xtrain_parameter_golf_train_infer
```

End-to-end proof:

```bash
target/debug/examples/xtrain_parameter_golf_train_infer /tmp/psionic_xtrain_pgolf_eval_1774599196
```

Generated report:

- `/tmp/psionic_xtrain_pgolf_eval_1774599196/xtrain_parameter_golf_train_infer_report.json`

## Result

Tuned quality remained unchanged:

- loss: `7.634324550628662`
- BPB: `8.811201735783069`

Throughput improved relative to the previously retained tuned report:

Previous tuned runtime:

- direct runtime: `24.48918946732971 tok/s`
- served runtime: `24.429709034170017 tok/s`

Allocation-tuned runtime:

- direct runtime: `24.827612078552526 tok/s`
- served runtime: `24.70252483344532 tok/s`
- legacy full-history path: `2.747308123460452 tok/s`
- direct over legacy improvement: `803.7068636946403%`

Delta from previous tuned runtime:

- direct runtime: `+0.3384226112228157 tok/s`
- served runtime: `+0.27281579927530185 tok/s`

Parity remained intact:

- direct runtime vs served runtime: `true`

## Conclusion

This runtime pass is worth retaining.

Why:

- quality stayed flat
- direct tok/s improved
- served tok/s improved
- parity stayed intact
- the change is simple and low-risk

## Remaining Honest Gap

This pass does not change the model-quality conclusion:

- the tuned XTRAIN-promoted model is better on loss/BPB
- the runtime is faster
- exact toy-cycle correctness is still not solved

So the current state after this pass is:

- train/infer plumbing: working
- bounded runtime parity: working
- tuned quality metrics: improved
- tuned tok/s: improved
- exact `abcd -> efghabcd` behavior: still not working
