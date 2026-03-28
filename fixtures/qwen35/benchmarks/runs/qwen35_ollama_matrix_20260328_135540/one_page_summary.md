# Qwen3.5 Psionic vs Ollama Benchmark Summary (2026-03-28)

This report is generated automatically from the benchmark JSONL evidence.

## Run Metadata

- run id: `qwen35_ollama_matrix_20260328_135540`
- artifact: `qwen35_ollama_matrix_qwen35_ollama_matrix_20260328_135540.json`
- jsonl source: `/home/michel/code/psionic_local/fixtures/qwen35/benchmarks/runs/qwen35_ollama_matrix_20260328_135540/runs.jsonl`
- raw log: `/home/michel/code/psionic_local/fixtures/qwen35/benchmarks/runs/qwen35_ollama_matrix_20260328_135540/raw.log`
- telemetry csv: `/home/michel/code/psionic_local/fixtures/qwen35/benchmarks/runs/qwen35_ollama_matrix_20260328_135540/telemetry.csv`
- psionic commit: `9686fab0fc38a31856bf81677ba2f80a3ece65d2`
- ollama version: `0.17.7`
- gpu: `NVIDIA GeForce RTX 4070 Laptop GPU`
- gpu vram mib: `8188`
- gpu max power limit w: `90.0`

## Executive Summary

- Comparable rows: `5` of `12` (token-delta thresholds: abs `16`, ratio `0.20`).
- Non-comparable rows: `7` of `12`.
- On comparable rows only, Psionic is ahead in `1` and behind in `4`.

## Comparable Rows

| Contract | Model | Psionic tok/s (mean±std) | Ollama tok/s (mean±std) | Ratio | Output tokens P/O |
| --- | --- | ---: | ---: | ---: | --- |
| `greedy` | `qwen3.5:0.8b` | 224.41±0.72 | 165.05±8.71 | 1.36x | 23.4/27.2 |
| `sampled_topk40` | `qwen3.5:0.8b` | 150.11±0.17 | 164.19±0.30 | 0.91x | 128.0/128.0 |
| `sampled_topk40` | `qwen3.5:2b` | 81.69±0.11 | 87.02±0.14 | 0.94x | 128.0/128.0 |
| `sampled_topk40` | `qwen3.5:4b` | 53.24±0.11 | 58.67±0.05 | 0.91x | 128.0/128.0 |
| `sampled_topk100` | `qwen3.5:4b` | 56.67±0.13 | 59.88±0.09 | 0.95x | 48.4/47.0 |

## Non-Comparable Rows

| Contract | Model | Classification reason | Psionic tokens | Ollama tokens |
| --- | --- | --- | ---: | ---: |
| `greedy` | `qwen3.5:2b` | token delta 12.00 (28.85%) exceeds thresholds | 29.6 | 41.6 |
| `greedy` | `qwen3.5:4b` | token delta 11.80 (23.41%) exceeds thresholds | 50.4 | 38.6 |
| `greedy` | `qwen3.5:9b` | token delta 11.60 (30.85%) exceeds thresholds | 26.0 | 37.6 |
| `sampled_topk40` | `qwen3.5:9b` | token delta 23.40 (18.28%) exceeds thresholds | 104.6 | 128.0 |
| `sampled_topk100` | `qwen3.5:0.8b` | token delta 11.40 (33.53%) exceeds thresholds | 22.6 | 34.0 |
| `sampled_topk100` | `qwen3.5:2b` | token delta 15.20 (34.55%) exceeds thresholds | 28.8 | 44.0 |
| `sampled_topk100` | `qwen3.5:9b` | token delta 13.40 (36.22%) exceeds thresholds | 23.6 | 37.0 |

## Graphs

### Throughput by contract

![Greedy throughput](./throughput_greedy.svg)

![Sampled top-k=40 throughput](./throughput_sampled_topk40.svg)

![Sampled top-k=100 throughput](./throughput_sampled_topk100.svg)

### Ratio overview

![Psionic to Ollama ratio](./psionic_vs_ollama_ratio.svg)
