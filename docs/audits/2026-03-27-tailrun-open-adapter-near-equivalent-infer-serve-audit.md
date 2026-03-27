# Tailrun Open-Adapter Near-Equivalent Infer/Serve Audit

> Status: retained 2026-03-27 audit for `TAILRUN-6`, covering the first honest
> train-to-infer-to-serve bridge for one bounded home-device artifact in the
> home Tailnet short-run lane.

## Purpose

The earlier Tailrun work proved that `psionic` could:

- train bounded same-node home-device artifacts
- compare the local M5 and remote RTX 4080 honestly
- complete one admitted mixed-device bounded run

But it still stopped too early in the lifecycle.

The missing question was:

- can one retained bounded home-device training artifact actually be exercised
  through the current inference and serving stack without pretending it is
  already a strict promoted PGOLF bundle

This audit records the first honest yes.

## What Was Bridged

The retained source artifact in this pass is the same-node M5 MLX bundle from:

- `fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b/m5_mlx/report.json`
- `fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b/m5_mlx/portable_bundle.safetensors`

That artifact is **not** a strict promoted Parameter Golf decoder bundle.

It is an open-adapter LM-head LoRA style training result. So the correct goal in
this pass was not to lie and call it a PGOLF promoted decoder. The correct goal
was to prove a documented near-equivalent bridge:

- reconstruct the trained LM-head LoRA weights from the retained portable bundle
- emit them as a real `safetensors` adapter artifact
- bind that adapter into `psionic-serve`
- prove one direct inference path and one served inference path on the same
  retained artifact

## Operator Command

The retained operator path is now:

```bash
cargo run -q -p psionic-serve \
  --example tailrun_open_adapter_near_equivalent_operator -- \
  --output-root fixtures/apple_adapter/promoted/tailrun_m5_near_equivalent_20260327
```

That example:

- loads the retained M5 same-node `portable_bundle.safetensors`
- reconstructs LoRA `A` and `B` matrices from the exported training groups
- writes `lm_head_lora_adapter.safetensors`
- writes a prompt-aligned dense CPU GGUF carrier model
- proves direct token selection through `LmHeadLoraAdapterArtifact`
- proves served token selection through `CpuGgufTextGenerationService`
- emits a retained manifest and report

## Retained Output

The retained near-equivalent publication lives at:

- `fixtures/apple_adapter/promoted/tailrun_m5_near_equivalent_20260327/carrier_model.gguf`
- `fixtures/apple_adapter/promoted/tailrun_m5_near_equivalent_20260327/lm_head_lora_adapter.safetensors`
- `fixtures/apple_adapter/promoted/tailrun_m5_near_equivalent_20260327/near_equivalent_manifest.json`
- `fixtures/apple_adapter/promoted/tailrun_m5_near_equivalent_20260327/near_equivalent_report.json`

## Retained Result

From the retained report:

- source training host: `ChristohersMBP2.lan`
- source training backend: `open_adapter_backend.mlx.metal.gpt_oss_lm_head`
- source training steps per second: `162.53061053630358`
- prompt token id: `7`
- expected target token id: `37`
- direct inference predicted token id: `37`
- served baseline output token: `1023`
- served overlay output token: `37`
- runtime support level: `lm_head_lora_cpu`

In plain language:

- without the trained adapter, the carrier model picks a useless baseline token
- with the trained adapter overlaid, `psionic-serve` picks the retained target
  token
- the same retained training artifact now has one honest direct inference proof
  and one honest served inference proof

## Exact Boundary

This bridge is intentionally documented as a **near-equivalent**, not as full
promotion.

What is allowed:

- the bounded home-device artifact can be reconstructed into a real LM-head
  LoRA `safetensors` file
- that adapter can be loaded directly for inference
- that adapter can be bound into `psionic-serve` over one prompt-aligned dense
  CPU GGUF carrier model

What is still refused:

- calling this artifact a strict PGOLF promoted decoder bundle
- calling it a production `gpt-oss` served artifact
- pretending the admitted mixed-device run already has this same retained bridge
- pretending this is automatic promotion

The retained manifest records that boundary explicitly:

- `strict_pgolf_promotion_disposition = refused`
- `near_equivalent_promotion_disposition = allowed`

## Why This Counts

The acceptance question for this phase was not “do we already have the final
production promotion system.”

The real question was “can a bounded home-device-trained artifact make it far
enough downstream to be judged through inference and serving.”

That answer is now yes for one retained home-device artifact.

This closes the most important operator gap between:

- “we can do bounded short-run training on home devices”
- and “we can exercise the result in the infer/serve stack without hand-waving”

## Honest Remaining Gap

This pass still does **not** prove:

- a strict promoted Parameter Golf decoder bundle from the Tailrun lane
- automatic promotion from a bounded home-device run into serving
- the same near-equivalent bridge for the admitted mixed-device swarm artifact
- useful product-model quality

So the correct next steps remain:

- freeze the daily operator loop and scoreboard
- then tune the best-known 10-minute profile
- then widen promotion from same-node artifacts toward admitted mixed-device
  artifacts

## Validation Run

The retained validation for this audit was:

```bash
cargo check -q -p psionic-serve --example tailrun_open_adapter_near_equivalent_operator

cargo run -q -p psionic-serve \
  --example tailrun_open_adapter_near_equivalent_operator -- \
  --output-root fixtures/apple_adapter/promoted/tailrun_m5_near_equivalent_20260327
```
