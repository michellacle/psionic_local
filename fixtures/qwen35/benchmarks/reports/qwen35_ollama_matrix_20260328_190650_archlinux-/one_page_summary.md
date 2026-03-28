# Qwen35 Native CUDA vs Ollama Matrix Summary

Run ID: `qwen35_ollama_matrix_20260328_190650_archlinux-`

Generated at: `2026-03-28T19:06:50Z`

Psionic commit: `f09a997e917a84cba1d8d9259fd8117da41c24f5`

Ollama version: `ollama version is 0.17.5`

Host label: `archlinux-`

Host GPU: `NVIDIA GeForce RTX 4080`

Host GPU memory total: `16376 MiB`

Host power limit: `320.00 W`

Host default power limit: `320.00 W`

Repeats per row: `3`

CARGO_INCREMENTAL: `0`

Change rationale: `Michel follow-up rerun with repo-owned per-run evidence capture`

Ollama comparison rationale: `Local Ollama comparison on the same host and prompt contract`

| Contract | Model | Psionic tok/s | Ollama tok/s | Ratio | Psionic output tokens | Ollama output tokens | Psionic termination | Ollama termination | First divergence | Strength |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `greedy` | `qwen3.5:0.8b` | `535.3762446599115` | `335.93144381165865` | `1.59` | `37,37,37` | `28,28,28` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `23,23,23` | `mismatched` |
| `greedy` | `qwen3.5:2b` | `260.1194118079032` | `206.2035528140798` | `1.26` | `33,33,33` | `34,34,34` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `none` | `strong` |
| `greedy` | `qwen3.5:4b` | `173.86916609640676` | `144.4117890190071` | `1.20` | `51,51,51` | `35,35,35` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `18,18,18` | `mismatched` |
| `greedy` | `qwen3.5:9b` | `107.74401283606898` | `97.43164811693903` | `1.11` | `41,41,41` | `42,42,42` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `5,5,5` | `mismatched` |
| `sampled_topk40` | `qwen3.5:0.8b` | `470.4921905680291` | `336.967178708952` | `1.40` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `3,3,3` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:2b` | `243.56417296303576` | `207.17201168756537` | `1.18` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `4,4,4` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:4b` | `175.06163697112007` | `144.17417749863205` | `1.21` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `4,4,4` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:9b` | `108.35583944817206` | `96.47632486815007` | `1.12` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `3,3,3` | `weak_length_matched_only` |
| `sampled_topk100` | `qwen3.5:0.8b` | `446.26124990451467` | `328.40760590107925` | `1.36` | `33,33,33` | `20,20,20` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `4,4,4` | `mismatched` |
| `sampled_topk100` | `qwen3.5:2b` | `236.1743124371845` | `203.5590099419148` | `1.16` | `27,27,27` | `38,38,38` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `2,2,2` | `mismatched` |
| `sampled_topk100` | `qwen3.5:4b` | `171.24118032885724` | `145.150270899278` | `1.18` | `41,41,41` | `37,37,37` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `5,5,5` | `mismatched` |
| `sampled_topk100` | `qwen3.5:9b` | `106.62317084372086` | `97.99855242725852` | `1.09` | `34,34,34` | `37,37,37` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `6,6,6` | `mismatched` |
