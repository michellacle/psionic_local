# GPT-OSS Local Serving

This document records the current local GPT-OSS serving and benchmarking path
in Psionic.

It is the shortest operator path for proving that Psionic can load a GPT-OSS
GGUF, serve it through an OpenAI-compatible HTTP surface, and compare that path
against local `llama.cpp`.

## What Ships

The current repo-owned entrypoints are:

- server binary:
  `crates/psionic-serve/src/bin/psionic-gpt-oss-server.rs`
- HTTP routes:
  `crates/psionic-serve/src/openai_http.rs`
- benchmark harness:
  `scripts/benchmark-gpt-oss-vs-llama.sh`

The focused GPT-OSS server exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

## Build

Build the server:

```bash
cargo build -p psionic-serve --bin psionic-gpt-oss-server --release
```

## Run

### Linux NVIDIA

```bash
./target/release/psionic-gpt-oss-server \
  -m /path/to/gpt-oss-20b-mxfp4.gguf \
  --backend cuda \
  --host 127.0.0.1 \
  --port 8080 \
  -c 4096 \
  -ngl 999
```

### Apple Silicon

```bash
./target/release/psionic-gpt-oss-server \
  -m /path/to/gpt-oss-20b-mxfp4.gguf \
  --backend metal \
  --metal-mode native \
  --host 127.0.0.1 \
  --port 8080 \
  -c 1024 \
  -ngl 4
```

`--metal-mode proxy` exists only for explicit `llama.cpp` proxy or debug runs.
It is not the same claim as the native Rust/Metal path.

## Call The Server

Health:

```bash
curl -s http://127.0.0.1:8080/health | jq
```

Model inventory:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq
```

Chat completion:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gpt-oss-20b-mxfp4.gguf",
    "messages": [
      {"role": "system", "content": "You are ChatGPT."},
      {"role": "user", "content": "Why does HTTPS matter?"}
    ]
  }' | jq
```

## Benchmark Against `llama.cpp`

Use the shipped harness:

```bash
scripts/benchmark-gpt-oss-vs-llama.sh \
  --psionic-backend cuda \
  --model /path/to/gpt-oss-20b-mxfp4.gguf \
  --llama-bin /path/to/llama-server \
  --json-out /tmp/psionic-gpt-oss-bench
```

The harness:

- starts the Psionic server or the local `llama.cpp` control
- runs cold, warm non-hit, and prompt-cache-hit cases
- uses the explicit GPT-OSS system/developer/user request contract
- compares visible output, not only raw token rate
- records JSON summaries when `--json-out` is set

## Current Public Proof

The public closeout referenced in the OpenAgents issue trail is:

- https://github.com/OpenAgentsInc/openagents/issues/3248#issuecomment-4028968842

That proof records:

- Psionic `prompt_cache_hit`: `172.84 tok/s`
- `llama.cpp prompt_cache_hit`: `160.98 tok/s`
- `prompt_cache_hit_visible_output_match=true`
- visible output:
  `HTTPS protects users by encrypting traffic, preventing tampering, and confirming they are connected to the right website.`

The benchmark result matters because it binds the claim to:

- the shipped server binary
- the shipped benchmark script
- the same visible output
- a Psionic-only execution path

## Related Docs

- `docs/INFERENCE_ENGINE.md`
- `docs/HARDWARE_VALIDATION_MATRIX.md`
- `docs/LLAMA_VLLM_SGLANG_INFERENCE_SPEC.md`
- `docs/ROADMAP_METAL.md`
