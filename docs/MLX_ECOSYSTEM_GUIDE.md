# Psionic MLX Ecosystem Guide

This guide is the package-facing entrypoint for the bounded MLX ecosystem in
Psionic.

It is intentionally CLI- and fixture-driven. It does not add Gradio demos,
desktop pickers, or product UX to `crates/psionic-*`.

## Text

Load or generate from a local GGUF model:

```bash
cargo run -p psionic-mlx-lm --bin psionic-mlx-lm -- \
  generate \
  --model /path/to/model.gguf \
  --prompt "Explain replay-safe inference." \
  --json-out /tmp/mlx-lm-generate.json
```

## Multimodal

Project one multimodal request into the shared text-serving lane:

```bash
cargo run -p psionic-mlx-vlm --bin psionic-mlx-vlm -- \
  plan-request \
  --family llava \
  --model-reference hf:openagents/demo \
  --endpoint responses \
  --messages-json fixtures/mlx_examples/vlm_messages.json \
  --json-out /tmp/mlx-vlm-plan.json
```

## Audio

Synthesize one CPU-reference speech clip:

```bash
cargo run -p psionic-mlx-audio --bin psionic-mlx-audio -- \
  synthesize \
  --family kokoro \
  --text "Receipts stay explicit." \
  --wav-out /tmp/mlx-audio.wav \
  --json-out /tmp/mlx-audio.json
```

## Serving

Plan one shared OpenAI-compatible serving package without starting the server:

```bash
cargo run -p psionic-mlx-serve --bin psionic-mlx-serve -- \
  plan \
  --reference hf:openagents/demo \
  --json-out /tmp/mlx-serve-plan.json
```

## Recipes

Emit one bounded MLX training recipe plan:

```bash
cargo run -p psionic-mlx-recipes --bin psionic-mlx-recipes -- \
  plan \
  --run-id demo-recipe \
  --environment env.psionic.demo@2026.03.17 \
  --method qlora \
  --adapter-rank 16 \
  --adapter-alpha 32 \
  --json-out /tmp/mlx-recipe-plan.json
```

## Evaluation

Build one benchmark suite from the committed fixture:

```bash
cargo run -p psionic-mlx-bench --bin psionic-mlx-bench -- \
  build-suite \
  --spec-json fixtures/mlx_examples/benchmark_suite.json \
  --json-out /tmp/mlx-benchmark-suite.json
```

Run the same suite through the fixture-backed local text adapter:

```bash
cargo run -p psionic-mlx-bench --bin psionic-mlx-bench -- \
  run-text-fixture \
  --spec-json fixtures/mlx_examples/benchmark_suite.json \
  --responses-json fixtures/mlx_examples/benchmark_responses.json \
  --json-out /tmp/mlx-benchmark-receipt.json
```

If you want to shape the same suite as a served benchmark lane, swap in
`run-served-fixture` and supply `--model-reference`.

## Related Docs

- `docs/MLX_TO_PSIONIC_MIGRATION_GUIDE.md`
- `docs/MLX_LM_PACKAGE.md`
- `docs/MLX_TEXT_SERVE.md`
- `docs/MLX_VLM_PACKAGE.md`
- `docs/MLX_AUDIO_PACKAGE.md`
- `docs/MLX_RECIPE_PACKAGE.md`
- `docs/MLX_BENCH_PACKAGE.md`
