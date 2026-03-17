# Psionic MLX Model Catalog

> Status: canonical `PMLX-702` / `#15` reference record, updated 2026-03-17
> after landing `crates/psionic-mlx-catalog`.

`psionic-mlx-catalog` is the first bounded MLX-style model-catalog package in
this repo.

It does not create a second loader stack. It packages the already-landed local
blob, Ollama manifest, registry-pull, GGUF adapter, and `psionic-mlx-lm`
surfaces into one MLX-facing catalog boundary.

## Current Scope

- resolve direct local GGUF paths into architecture-aware package reports
- resolve local Ollama model ids through `psionic-catalog`
- discover local Hugging Face hub snapshots through explicit `hf:<owner>/<repo>`
  references
- keep one builtin architecture registry for the current direct text-runtime
  families
- surface one explicit conversion entrypoint for local safetensors snapshots
  without pretending the conversion itself is already shipped
- keep trust or refusal for remote processor and template metadata explicit
- hand direct GGUF sources off to `psionic-mlx-lm` when the architecture is
  registered
- expose a package-owned CLI for resolve, discover, load, and Ollama pull
  workflows

## Current Boundaries

- Hugging Face support is local-cache discovery only
- remote registry pull remains Ollama-style only through the wrapped
  `psionic-catalog` substrate
- safetensors conversion is still a planned entrypoint/report, not a shipped
  converter
- remote processor/template metadata is never trusted implicitly
- multimodal processors and prompt shapers stay outside this crate

Those later surfaces belong to `PMLX-703` through `PMLX-709`.

## Library Surface

The crate now exposes:

- `MlxCatalogWorkspace`
- `MlxCatalogRoots`
- `MlxCatalogResolutionReport`
- `HuggingFaceCacheSnapshot`
- `MlxArchitectureRegistry`
- `MlxRemoteMetadataPolicy`
- `MlxConversionEntryPoint`

## CLI

Run from the repo root:

```bash
cargo run -p psionic-mlx-catalog -- resolve --reference /path/to/model.gguf
cargo run -p psionic-mlx-catalog -- resolve --reference qwen2
cargo run -p psionic-mlx-catalog -- resolve --reference hf:mlx-community/Qwen2-0.5B-Instruct-4bit
cargo run -p psionic-mlx-catalog -- discover-hf --repo mlx-community/Qwen2-0.5B-Instruct-4bit
cargo run -p psionic-mlx-catalog -- load-text --reference qwen2
```

Allow digest-bound local cache metadata explicitly when needed:

```bash
cargo run -p psionic-mlx-catalog -- resolve \
  --reference hf:mlx-community/Qwen2-0.5B-Instruct-4bit \
  --allow-template-metadata \
  --allow-processor-metadata
```

Pull one Ollama model into the local models root through the wrapped registry
client:

```bash
cargo run -p psionic-mlx-catalog -- pull-ollama --reference qwen2
```

## Why This Exists

`PMLX-702` is not a claim that the whole MLX ecosystem is now complete.

It closes the package-level gap between one local text runtime and real
model-id resolution: callers can now resolve local GGUF paths, local Ollama
manifests, and local Hugging Face cache snapshots through one Psionic-owned
catalog/report layer while keeping architecture admission, conversion posture,
and remote-metadata trust explicit.
