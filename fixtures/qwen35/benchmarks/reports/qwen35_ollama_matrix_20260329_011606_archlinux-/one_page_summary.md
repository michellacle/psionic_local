# Qwen35 Native CUDA vs Ollama Matrix Summary

Run ID: `qwen35_ollama_matrix_20260329_011606_archlinux-`

Generated at: `2026-03-29T01:16:06Z`

Psionic commit: `856e21767294a4a2cb682abeaabb9b862a203c08`

Ollama version: `ollama version is 0.17.5`

Host label: `archlinux-`

Host GPU: `NVIDIA GeForce RTX 4080`

Host GPU memory total: `16376 MiB`

Host power limit: `320.00 W`

Host default power limit: `320.00 W`

Repeats per row: `3`

CARGO_INCREMENTAL: `0`

Change rationale: `partitioned_top_k_block_shape_256x8`

Ollama comparison rationale: `Local Ollama comparison on the same host and prompt contract`

| Contract | Model | Psionic tok/s | Ollama tok/s | Ratio | Psionic output tokens | Ollama output tokens | Psionic termination | Ollama termination | First divergence | Strength |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `sampled_topk40` | `qwen3.5:0.8b` | `518.744765919478` | `338.97205268362205` | `1.53` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `3,3,3` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:2b` | `256.2540011103346` | `206.2233396827902` | `1.24` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `4,4,4` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:4b` | `181.13277109387755` | `144.44794190481875` | `1.25` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `4,4,4` | `weak_length_matched_only` |
| `sampled_topk40` | `qwen3.5:9b` | `110.74881588595179` | `96.60416241685304` | `1.15` | `128,128,128` | `128,128,128` | `max_output_tokens,max_output_tokens,max_output_tokens` | `max_output_tokens,max_output_tokens,max_output_tokens` | `3,3,3` | `weak_length_matched_only` |
| `sampled_topk100` | `qwen3.5:0.8b` | `513.0638725655157` | `322.0647256380718` | `1.59` | `33,33,33` | `20,20,20` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `4,4,4` | `mismatched` |
| `sampled_topk100` | `qwen3.5:2b` | `253.55419938735756` | `205.6082934546246` | `1.23` | `27,27,27` | `38,38,38` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `2,2,2` | `mismatched` |
| `sampled_topk100` | `qwen3.5:4b` | `179.73831629626423` | `144.73067787586652` | `1.24` | `41,41,41` | `37,37,37` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `5,5,5` | `mismatched` |
| `sampled_topk100` | `qwen3.5:9b` | `109.98043588091146` | `97.5937727062926` | `1.13` | `34,34,34` | `37,37,37` | `eos_token,eos_token,eos_token` | `eos_token,eos_token,eos_token` | `6,6,6` | `mismatched` |
