# Psionic MLX Workflow Package

This document defines the bounded `psionic-mlx-workflows` package that closes
`PMLX-707`.

## Scope

`psionic-mlx-workflows` is the MLX-facing workflow layer above
`psionic-data`, `psionic-mlx-recipes`, `psionic-train`, and portable
adapter/model IO.

It owns:

- deterministic synthetic SFT and preference dataset bundle generation
- reward-model and judge-model helper plans above the existing MLX recipe lane
- adapter merge/export workflows above the portable model-IO surface
- a local Hugging Face-style publish snapshot over existing safetensors export
- lineage-bound dataset, merge, and publish manifests instead of notebook-only
  side effects

It does not own:

- a second training runtime
- a second adapter or model-IO format
- remote registry authority
- synthetic-data job lifecycle authority outside this repo

## Current Truth

The package is a bounded workflow surface, not a hidden new control plane.

It reuses the existing Psionic substrate:

- `psionic-data` for dataset and split manifests
- `psionic-datastream` for payload manifests and object digests
- `psionic-mlx-recipes` for recipe planning
- `psionic-train::model_io` for adapter delta derivation plus bundle export

That keeps synthetic data, supervision helpers, adapter merge, and publish
flows tied to the same machine-legible truth as the rest of the repo.

## Synthetic Data

The package now materializes:

- SFT JSONL bundles with dataset manifests and datastream manifests per split
- preference JSONL bundles with the same lineage shape

Those bundles are deterministic local artifacts with explicit split counts,
payload digests, and dataset-manifest digests.

This is not the full platform synthetic-data job lifecycle. The broader
authority-owned create/append/finalize/verify flow remains outside this repo.

## Reward And Judge Helpers

`psionic-mlx-workflows` now plans reward-model and judge-model helpers by
reusing `psionic-mlx-recipes`.

That means:

- reward helpers stay on SFT/adapter methods
- judge helpers can reuse SFT or preference-style methods, but RL-only methods
  refuse explicitly
- helper plans emit typed environment policy references instead of hiding
  reward/judge policy posture in notebook code

## Adapter Merge And Publish

The package now derives adapter deltas and verifies merge/unmerge roundtrips on
top of portable model IO.

It also publishes one local Hugging Face-style snapshot directory over the
existing safetensors export boundary, including:

- `model.safetensors`
- bundle manifest
- tokenizer contract
- compatibility contract
- publish manifest

Direct GGUF export remains an explicit refusal because the current portable
model-IO boundary imports GGUF but does not emit GGUF.
