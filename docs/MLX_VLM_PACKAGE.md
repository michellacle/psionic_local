# Psionic MLX Multimodal Package

This document defines the bounded `psionic-mlx-vlm` package that closes
`PMLX-704`.

## Scope

`psionic-mlx-vlm` is the MLX-facing multimodal package above `psionic-models`
and the shared Psionic serving lane.

It owns:

- builtin processor registries for bounded VLM/omni-family aliases
- OpenAI-compatible image/audio/video request shapes
- multimodal prompt shaping into the shared text-serving lane
- machine-readable attachment receipts and served-request plans
- package-facing CLI entrypoints for shaping and request planning

It does not own:

- a claimed native image/audio/video encoder runtime
- TTS or speech-generation outputs
- product UX, demos, or desktop shells

## Current Truth

The current package closes the MLX multimodal ecosystem gap with one honest
strategy: prompt projection.

That means:

- media attachments are accepted as typed image/audio/video inputs
- the package computes digest-bound attachment receipts
- the package inserts typed prompt markers into projected `PromptMessage`s
- the package emits translated `/v1/responses` or `/v1/chat/completions`
  request JSON for the shared `psionic-mlx-serve` / `psionic-serve` text lane

It does not claim a native multimodal encoder. The projection notes must stay
explicit about that boundary.

## Builtin Families

The builtin processor registry currently covers one bounded set of families:

- `llava` for image inputs
- `qwen2_vl` / `qwen2-vl` for image and video inputs
- `omni` / `gpt_oss_omni` for image, audio, and video inputs

All of them currently use `prompt_projection_only`.

## Request Shapes

The library accepts OpenAI-compatible content-part shapes:

- `input_text`
- `input_image`
- `input_audio`
- `input_video`

Those parts can point at:

- remote URLs
- local file paths
- data URLs

Each attachment receives a stable digest plus one `<psionic_media ... />`
marker inserted into the projected prompt.

## Served Surface

The package plans requests for the shared text-serving lane only:

- `/v1/responses`
- `/v1/chat/completions`

The translated request is text-only because the current projection mode uses the
shared text server rather than a separate multimodal runtime.

## CLI

The package CLI is:

- `psionic-mlx-vlm shape`
- `psionic-mlx-vlm plan-request`

`shape` emits the prompt-projection report.
`plan-request` emits the translated served-request plan plus the same
projection receipts.
