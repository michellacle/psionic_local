# 2026-03-25 Psionic Parameter Golf Same-Node H100 Parity Audit

This audit records the first real same-node H100 parity comparison between the
Psionic Rust single-H100 trainer and the upstream `train_gpt.py` baseline.

## Conclusion

`#472` is now closeable.

The repo now has the machine-readable same-node H100 parity receipt the issue
asked for. The result is bad for Psionic.

On the same RunPod `NVIDIA H100 80GB HBM3` node:

- Psionic is much slower on the first train step
- Psionic is much slower on final roundtrip validation
- Psionic ends with much worse final roundtrip `val_loss` and `val_bpb`
- Psionic does produce a smaller compressed artifact
- the parity report remains `blocked` only because Psionic still does not carry
  the peak-memory metric that the upstream log exposes

This issue closes because the comparison surface now exists and later work can
cite it directly. It does not close because Psionic achieved parity.

## Committed Evidence

- [parameter_golf_train_gpt_reference_run_receipt_same_node_h100.json](/tmp/psionic-open-issues.PzsotD/fixtures/parameter_golf/reports/parameter_golf_train_gpt_reference_run_receipt_same_node_h100.json)
- [parameter_golf_same_node_parity_report_h100.json](/tmp/psionic-open-issues.PzsotD/fixtures/parameter_golf/reports/parameter_golf_same_node_parity_report_h100.json)
- [parameter_golf_runpod_single_h100_first_real_training_report.json](/tmp/psionic-open-issues.PzsotD/fixtures/parameter_golf/reports/parameter_golf_runpod_single_h100_first_real_training_report.json)

The real pod-side comparison root was:

- `/workspace/parameter_golf_same_node_parity_20260325T122145Z`

The matched upstream log was:

- `/workspace/parameter_golf_train_gpt_reference_20260325T122145Z.log`

The matched Psionic run root was:

- `/workspace/issue454_single_h100_run_20260324T215106Z`

## Matched H100 Outcome

Matched machine identity:

- device: `NVIDIA H100 80GB HBM3`
- world size: `1`
- grad accumulation steps: `8`
- train sequence length: `1024`
- train batch tokens: `524288`
- dataset manifest digest matched
- tokenizer digest matched

Upstream `train_gpt.py` receipt:

- train-step wallclock: `474 ms`
- final roundtrip eval time: `10,723 ms`
- final roundtrip `val_loss`: `2.29099059`
- final roundtrip `val_bpb`: `1.35685367`
- peak memory allocated: `10,303 MiB`
- peak memory reserved: `10,712 MiB`
- compressed model bytes: `13,563,833`

Psionic Rust single-H100 receipt:

- train-step wallclock: `1,348,854 ms`
- final roundtrip eval time: `43,306,994 ms`
- final roundtrip `val_loss`: `10.64899007`
- final roundtrip `val_bpb`: `6.30693175`
- peak memory allocated: unavailable
- compressed model bytes: `4,732,744`

## Parity Report Judgment

The committed parity report is:

- `disposition=blocked`

Its only blocker is:

- `required parity metric peak_memory_allocated_mib is missing on one side of the receipt pair`

The actual comparison rows are still decisive:

- `train_step_wallclock_ms`: `worse`
- `validation_wallclock_ms`: `worse`
- `peak_memory_allocated_mib`: `missing`
- `final_roundtrip_val_loss`: `divergent`
- `final_roundtrip_val_bpb`: `divergent`
- `compressed_model_bytes`: `better`

This is the right fail-closed answer. The receipt does not invent a memory
comparison it does not have, but it still preserves the comparisons that are
present and makes the bad outcome explicit.

## Operator Fix That Made The Receipt Possible

The first chained parity attempt failed for an environment reason, not a model
reason: the pod system `python3` shipped `torch 2.4.1`, and that interpreter
could not execute `scaled_dot_product_attention(..., enable_gqa=...)` for the
current upstream `train_gpt.py`.

The runner is now fixed to:

- prefer `repo-root/.venv/bin/python` when it exists
- bootstrap or upgrade `repo-root/.venv` from `requirements.txt` when the
  current Python cannot satisfy the upstream contract
- launch `torch.distributed.run` through the resolved interpreter instead of a
  bare `torchrun` on `PATH`

That is why the final upstream same-node run succeeded instead of failing on the
system Python.

## What This Means For The Remaining PGOLF Stack

This audit retires one backlog question:

- the repo no longer needs to reconstruct same-node H100 parity by hand from
  separate Psionic and upstream logs

It makes the remaining work harsher and clearer:

- Psionic is not close to the upstream single-H100 baseline on time or score
- the current compression path is smaller, but that is not enough to overcome
  the runtime and loss gap
- later parity and readiness issues should cite this receipt directly instead of
  relitigating whether a same-node comparison exists

The next parity work should optimize or fix the runtime, metric, and memory
surfaces. It should not keep pretending the baseline comparison is still
unknown.
