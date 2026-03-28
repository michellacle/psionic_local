#!/usr/bin/env python3

import argparse
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


CONTRACT_ORDER = ["greedy", "sampled_topk40", "sampled_topk100"]
MODEL_ORDER = ["qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build enriched qwen35 benchmark artifact and one-page report from jsonl runs."
    )
    parser.add_argument("--jsonl", required=True, help="Path to qwen35 bench jsonl runs.")
    parser.add_argument("--output-dir", required=True, help="Output directory for artifact/report.")
    parser.add_argument("--run-id", required=True, help="Stable run id.")
    parser.add_argument("--run-date", default=str(date.today()), help="Run date (YYYY-MM-DD).")
    parser.add_argument("--raw-log", default=None, help="Raw benchmark log path.")
    parser.add_argument("--telemetry-csv", default=None, help="GPU telemetry CSV path.")
    parser.add_argument("--git-commit", default=None, help="Psionic git commit used for run.")
    parser.add_argument("--ollama-version", default=None, help="Ollama version.")
    parser.add_argument("--gpu-name", default=None, help="GPU name.")
    parser.add_argument("--gpu-vram-mib", type=int, default=None, help="GPU VRAM MiB.")
    parser.add_argument("--gpu-power-limit-w", type=float, default=None, help="GPU power limit.")
    parser.add_argument(
        "--gpu-power-max-limit-w",
        type=float,
        default=None,
        help="GPU max power limit.",
    )
    parser.add_argument(
        "--token-delta-abs-threshold",
        type=int,
        default=16,
        help="Absolute output-token delta threshold for non-comparable rows.",
    )
    parser.add_argument(
        "--token-delta-ratio-threshold",
        type=float,
        default=0.20,
        help="Relative output-token delta threshold for non-comparable rows.",
    )
    return parser.parse_args()


def safe_mean(values):
    return statistics.fmean(values) if values else 0.0


def safe_stddev(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def infer_contract(record):
    contract = record.get("row_label")
    if contract:
        return contract
    request = record.get("request_contract") or {}
    decode_mode = request.get("decode_mode")
    top_k = request.get("top_k")
    if decode_mode == "greedy":
        return "greedy"
    if decode_mode == "sample" and top_k is not None:
        return f"sampled_topk{top_k}"
    return "unknown"


def infer_model_alias(record):
    alias = record.get("model_alias")
    if alias:
        return alias
    alias = record.get("ollama_model")
    if alias:
        return alias
    model_path = record.get("model_path") or ""
    name = os.path.basename(model_path)
    if "0.8b" in name:
        return "qwen3.5:0.8b"
    if "2b" in name:
        return "qwen3.5:2b"
    if "4b" in name:
        return "qwen3.5:4b"
    if "9b" in name:
        return "qwen3.5:9b"
    return name


def model_sort_key(model):
    if model in MODEL_ORDER:
        return (0, MODEL_ORDER.index(model))
    return (1, model)


def contract_sort_key(contract):
    if contract in CONTRACT_ORDER:
        return (0, CONTRACT_ORDER.index(contract))
    return (1, contract)


def parse_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {idx}: {error}") from error
            records.append(record)
    return records


def summarize_backend_runs(runs):
    decode_tok_s = [float(run.get("decode_tok_s", 0.0)) for run in runs]
    prompt_s = [float(run.get("prompt_s", 0.0)) for run in runs]
    decode_s = [float(run.get("decode_s", 0.0)) for run in runs]
    total_s = [float(run.get("total_s", 0.0)) for run in runs]
    output_tokens = [int(run.get("output_tokens", 0)) for run in runs]
    prompt_tokens = [int(run.get("prompt_tokens", 0)) for run in runs]
    terminations = Counter(
        str(run.get("termination_reason"))
        for run in runs
        if run.get("termination_reason") is not None
    )
    finish_reasons = Counter(
        str(run.get("finish_reason"))
        for run in runs
        if run.get("finish_reason") is not None
    )
    output_modes = Counter()
    readback_bytes = []
    raw_logits = False
    for run in runs:
        for mode in run.get("qwen35_output_modes") or []:
            output_modes[str(mode)] += 1
        if run.get("qwen35_readback_bytes") is not None:
            readback_bytes.append(float(run["qwen35_readback_bytes"]))
        raw_logits = raw_logits or bool(run.get("qwen35_raw_logits", False))
    return {
        "repeats": len(runs),
        "mean_decode_tok_s": safe_mean(decode_tok_s),
        "stddev_decode_tok_s": safe_stddev(decode_tok_s),
        "min_decode_tok_s": min(decode_tok_s) if decode_tok_s else 0.0,
        "max_decode_tok_s": max(decode_tok_s) if decode_tok_s else 0.0,
        "mean_prompt_s": safe_mean(prompt_s),
        "mean_decode_s": safe_mean(decode_s),
        "mean_total_s": safe_mean(total_s),
        "mean_output_tokens": safe_mean(output_tokens),
        "mean_prompt_tokens": safe_mean(prompt_tokens),
        "termination_reasons": dict(terminations),
        "finish_reasons": dict(finish_reasons),
        "qwen35_output_modes": dict(output_modes),
        "qwen35_readback_bytes_mean": safe_mean(readback_bytes),
        "qwen35_raw_logits_any": raw_logits,
        "runs": runs,
    }


def classify_row(psionic_summary, ollama_summary, abs_threshold, ratio_threshold):
    if psionic_summary is None or ollama_summary is None:
        return "missing_backend", "missing psionic or ollama row"
    p_tokens = psionic_summary["mean_output_tokens"]
    o_tokens = ollama_summary["mean_output_tokens"]
    delta = abs(p_tokens - o_tokens)
    denom = max(p_tokens, o_tokens, 1.0)
    delta_ratio = delta / denom
    if delta >= abs_threshold or delta_ratio >= ratio_threshold:
        return (
            "non_comparable_token_divergence",
            f"token delta {delta:.2f} ({delta_ratio:.2%}) exceeds thresholds",
        )
    return "comparable", "output-token deltas within comparability thresholds"


def parse_telemetry(csv_path: Path):
    if not csv_path or not csv_path.exists():
        return None
    power = []
    temp = []
    util_gpu = []
    util_mem = []
    clocks_sm = []
    clocks_mem = []
    mem_used = []
    mem_total = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = [token.strip() for token in line.split(",")]
            if len(parts) < 9:
                continue
            try:
                power.append(float(parts[1]))
                clocks_sm.append(float(parts[2]))
                clocks_mem.append(float(parts[3]))
                temp.append(float(parts[4]))
                util_gpu.append(float(parts[5]))
                util_mem.append(float(parts[6]))
                mem_used.append(float(parts[7]))
                mem_total.append(float(parts[8]))
            except ValueError:
                continue
    if not power:
        return None
    return {
        "samples": len(power),
        "power_draw_w": {"mean": safe_mean(power), "max": max(power)},
        "temperature_c": {"mean": safe_mean(temp), "max": max(temp)},
        "utilization_gpu_pct": {"mean": safe_mean(util_gpu), "max": max(util_gpu)},
        "utilization_mem_pct": {"mean": safe_mean(util_mem), "max": max(util_mem)},
        "clocks_sm_mhz": {"mean": safe_mean(clocks_sm), "max": max(clocks_sm)},
        "clocks_mem_mhz": {"mean": safe_mean(clocks_mem), "max": max(clocks_mem)},
        "memory_used_mib": {"mean": safe_mean(mem_used), "max": max(mem_used)},
        "memory_total_mib": {"mean": safe_mean(mem_total), "max": max(mem_total)},
    }


def format_model_short(model_alias):
    return model_alias.replace("qwen3.5:", "")


def write_svg(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def generate_throughput_svg(path: Path, contract: str, rows):
    w, h = 980, 450
    ml, mr, mt, mb = 90, 40, 60, 110
    cw, ch = w - ml - mr, h - mt - mb
    vals = []
    for row in rows:
        for backend in ("psionic", "ollama"):
            backend_data = row["backends"].get(backend)
            if backend_data:
                vals.append(float(backend_data["mean_decode_tok_s"]))
    ymax = max(vals) * 1.15 if vals else 100.0
    ymax = max(50.0, math.ceil(ymax / 10.0) * 10.0)

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    )
    lines.append("<style>")
    lines.append('text { font-family: "DejaVu Sans", Arial, sans-serif; fill: #1f2937; }')
    lines.append(".title { font-size: 20px; font-weight: 700; }")
    lines.append(".axis { font-size: 12px; }")
    lines.append(".tick { font-size: 11px; }")
    lines.append(".val { font-size: 11px; font-weight: 600; }")
    lines.append("</style>")
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
    lines.append(f'<text class="title" x="{ml}" y="34">Qwen3.5 throughput: {contract}</text>')
    lines.append(
        f'<text class="axis" x="{ml}" y="53">Decode throughput (tok/s), higher is better</text>'
    )

    for i in range(6):
        t = i / 5.0
        y = mt + ch * t
        yv = ymax * (1 - t)
        lines.append(
            f'<line x1="{ml}" y1="{y:.2f}" x2="{ml + cw}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text class="tick" x="{ml - 8}" y="{y + 4:.2f}" text-anchor="end">{yv:.0f}</text>'
        )
    lines.append(
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" stroke="#111827" stroke-width="1.5"/>'
    )
    lines.append(
        f'<line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" stroke="#111827" stroke-width="1.5"/>'
    )

    group_w = cw / max(len(rows), 1)
    bar_w = group_w * 0.24
    for index, row in enumerate(rows):
        gx = ml + index * group_w + group_w * 0.2
        p = row["backends"].get("psionic")
        o = row["backends"].get("ollama")
        p_val = float(p["mean_decode_tok_s"]) if p else 0.0
        o_val = float(o["mean_decode_tok_s"]) if o else 0.0
        p_h = (p_val / ymax) * ch
        o_h = (o_val / ymax) * ch
        px = gx
        ox = gx + bar_w + group_w * 0.08
        py = mt + ch - p_h
        oy = mt + ch - o_h
        lines.append(
            f'<rect x="{px:.2f}" y="{py:.2f}" width="{bar_w:.2f}" height="{p_h:.2f}" fill="#1f77b4"/>'
        )
        lines.append(
            f'<rect x="{ox:.2f}" y="{oy:.2f}" width="{bar_w:.2f}" height="{o_h:.2f}" fill="#ff7f0e"/>'
        )
        lines.append(
            f'<text class="val" x="{px + bar_w / 2:.2f}" y="{max(py - 6, mt + 12):.2f}" text-anchor="middle">{p_val:.2f}</text>'
        )
        lines.append(
            f'<text class="val" x="{ox + bar_w / 2:.2f}" y="{max(oy - 6, mt + 12):.2f}" text-anchor="middle">{o_val:.2f}</text>'
        )
        p_tok = p["mean_output_tokens"] if p else 0
        o_tok = o["mean_output_tokens"] if o else 0
        mark = " *" if row["classification"] != "comparable" else ""
        lines.append(
            f'<text class="axis" x="{gx + group_w * 0.26:.2f}" y="{mt + ch + 24}" text-anchor="middle">{format_model_short(row["model"])}{mark}</text>'
        )
        lines.append(
            f'<text class="tick" x="{gx + group_w * 0.26:.2f}" y="{mt + ch + 40}" text-anchor="middle">P:{p_tok:.1f} O:{o_tok:.1f}</text>'
        )

    lx = ml + cw - 210
    ly = mt + 8
    lines.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="#1f77b4"/>')
    lines.append(f'<text class="axis" x="{lx + 20}" y="{ly + 12}">Psionic</text>')
    lines.append(f'<rect x="{lx + 95}" y="{ly}" width="14" height="14" fill="#ff7f0e"/>')
    lines.append(f'<text class="axis" x="{lx + 115}" y="{ly + 12}">Ollama</text>')
    lines.append(
        f'<text class="tick" x="{ml}" y="{h - 18}">* non-comparable row by token-delta policy</text>'
    )
    lines.append("</svg>")
    write_svg(path, "\n".join(lines))


def generate_ratio_svg(path: Path, rows):
    items = []
    for row in rows:
        p = row["backends"].get("psionic")
        o = row["backends"].get("ollama")
        if not p or not o:
            continue
        denom = float(o["mean_decode_tok_s"])
        ratio = float(p["mean_decode_tok_s"]) / denom if denom > 0 else 0.0
        items.append(
            {
                "label": f"{row['contract']} · {format_model_short(row['model'])}",
                "ratio": ratio,
                "classification": row["classification"],
                "tokens": (p["mean_output_tokens"], o["mean_output_tokens"]),
            }
        )
    if not items:
        return
    w, h = 1060, 580
    ml, mr, mt, mb = 240, 40, 60, 30
    cw, ch = w - ml - mr, h - mt - mb
    bar_h = ch / len(items) * 0.62
    stride = ch / len(items)
    xmax = max(1.45, max(item["ratio"] for item in items) * 1.12)

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    )
    lines.append("<style>")
    lines.append('text { font-family: "DejaVu Sans", Arial, sans-serif; fill: #1f2937; }')
    lines.append(".title { font-size: 20px; font-weight: 700; }")
    lines.append(".axis { font-size: 12px; }")
    lines.append(".tick { font-size: 11px; }")
    lines.append(".val { font-size: 11px; font-weight: 600; }")
    lines.append("</style>")
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
    lines.append('<text class="title" x="240" y="32">Psionic / Ollama decode throughput ratio</text>')
    lines.append(
        '<text class="axis" x="240" y="49">1.00 means parity, above 1.00 means Psionic is faster</text>'
    )
    for i in range(6):
        value = xmax * i / 5.0
        x = ml + (value / xmax) * cw
        stroke = "#111827" if abs(value - 1.0) <= 0.05 else "#d1d5db"
        width = "1.5" if abs(value - 1.0) <= 0.05 else "1"
        lines.append(
            f'<line x1="{x:.2f}" y1="{mt}" x2="{x:.2f}" y2="{mt + ch}" stroke="{stroke}" stroke-width="{width}"/>'
        )
        lines.append(
            f'<text class="tick" x="{x:.2f}" y="{mt + ch + 16}" text-anchor="middle">{value:.2f}</text>'
        )
    for idx, item in enumerate(items):
        y = mt + idx * stride + (stride - bar_h) / 2.0
        bw = (item["ratio"] / xmax) * cw
        if item["classification"] == "comparable":
            color = "#059669" if item["ratio"] >= 1.0 else "#dc2626"
        else:
            color = "#9ca3af"
        lines.append(
            f'<rect x="{ml}" y="{y:.2f}" width="{bw:.2f}" height="{bar_h:.2f}" fill="{color}"/>'
        )
        lines.append(
            f'<text class="axis" x="{ml - 8}" y="{y + bar_h * 0.72:.2f}" text-anchor="end">{item["label"]}</text>'
        )
        token_note = ""
        p_tok, o_tok = item["tokens"]
        if abs(p_tok - o_tok) > 0.1:
            token_note = f" *P:{p_tok:.1f} O:{o_tok:.1f}"
        lines.append(
            f'<text class="val" x="{ml + bw + 6:.2f}" y="{y + bar_h * 0.72:.2f}">{item["ratio"]:.2f}x{token_note}</text>'
        )
    lines.append(
        f'<text class="tick" x="{ml}" y="{h - 8}">gray bars are non-comparable rows</text>'
    )
    lines.append("</svg>")
    write_svg(path, "\n".join(lines))


def summarize(records, args):
    runs_by_row_backend = defaultdict(list)
    request_contract_by_row = {}
    git_commit_from_runs = None
    for record in records:
        if record.get("record_type") != "run":
            continue
        backend = record.get("backend")
        if backend not in ("psionic", "ollama"):
            continue
        contract = infer_contract(record)
        model = infer_model_alias(record)
        runs_by_row_backend[(contract, model, backend)].append(record)
        if (contract, model) not in request_contract_by_row:
            request_contract_by_row[(contract, model)] = record.get("request_contract")
        if git_commit_from_runs is None and record.get("git_commit"):
            git_commit_from_runs = record.get("git_commit")

    row_keys = {(contract, model) for (contract, model, _) in runs_by_row_backend.keys()}
    sorted_keys = sorted(
        row_keys, key=lambda key: (contract_sort_key(key[0]), model_sort_key(key[1]))
    )

    rows = []
    for contract, model in sorted_keys:
        psionic_runs = runs_by_row_backend.get((contract, model, "psionic"), [])
        ollama_runs = runs_by_row_backend.get((contract, model, "ollama"), [])
        psionic_summary = summarize_backend_runs(psionic_runs) if psionic_runs else None
        ollama_summary = summarize_backend_runs(ollama_runs) if ollama_runs else None
        classification, reason = classify_row(
            psionic_summary,
            ollama_summary,
            args.token_delta_abs_threshold,
            args.token_delta_ratio_threshold,
        )
        ratio = None
        if psionic_summary and ollama_summary and ollama_summary["mean_decode_tok_s"] > 0:
            ratio = psionic_summary["mean_decode_tok_s"] / ollama_summary["mean_decode_tok_s"]
        rows.append(
            {
                "contract": contract,
                "model": model,
                "classification": classification,
                "classification_reason": reason,
                "psionic_to_ollama_ratio": ratio,
                "request_contract": request_contract_by_row.get((contract, model)),
                "backends": {
                    "psionic": psionic_summary,
                    "ollama": ollama_summary,
                },
            }
        )

    comparable = [row for row in rows if row["classification"] == "comparable"]
    non_comparable = [row for row in rows if row["classification"] != "comparable"]
    return rows, comparable, non_comparable, git_commit_from_runs


def write_markdown(path: Path, artifact_name: str, rows, comparable, non_comparable, args):
    lines = []
    lines.append(f"# Qwen3.5 Psionic vs Ollama Benchmark Summary ({args.run_date})")
    lines.append("")
    lines.append("This report is generated automatically from the benchmark JSONL evidence.")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- run id: `{args.run_id}`")
    lines.append(f"- artifact: `{artifact_name}`")
    lines.append(f"- jsonl source: `{args.jsonl}`")
    if args.raw_log:
        lines.append(f"- raw log: `{args.raw_log}`")
    if args.telemetry_csv:
        lines.append(f"- telemetry csv: `{args.telemetry_csv}`")
    if args.git_commit:
        lines.append(f"- psionic commit: `{args.git_commit}`")
    if args.ollama_version:
        lines.append(f"- ollama version: `{args.ollama_version}`")
    if args.gpu_name:
        lines.append(f"- gpu: `{args.gpu_name}`")
    if args.gpu_vram_mib is not None:
        lines.append(f"- gpu vram mib: `{args.gpu_vram_mib}`")
    if args.gpu_power_limit_w is not None:
        lines.append(f"- gpu power limit w: `{args.gpu_power_limit_w}`")
    if args.gpu_power_max_limit_w is not None:
        lines.append(f"- gpu max power limit w: `{args.gpu_power_max_limit_w}`")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"- Comparable rows: `{len(comparable)}` of `{len(rows)}` (token-delta thresholds: abs `{args.token_delta_abs_threshold}`, ratio `{args.token_delta_ratio_threshold:.2f}`)."
    )
    lines.append(f"- Non-comparable rows: `{len(non_comparable)}` of `{len(rows)}`.")
    if comparable:
        wins = 0
        losses = 0
        for row in comparable:
            ratio = row.get("psionic_to_ollama_ratio")
            if ratio is None:
                continue
            if ratio > 1.0:
                wins += 1
            elif ratio < 1.0:
                losses += 1
        lines.append(
            f"- On comparable rows only, Psionic is ahead in `{wins}` and behind in `{losses}`."
        )
    lines.append("")
    lines.append("## Comparable Rows")
    lines.append("")
    lines.append(
        "| Contract | Model | Psionic tok/s (mean±std) | Ollama tok/s (mean±std) | Ratio | Output tokens P/O |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for row in comparable:
        p = row["backends"]["psionic"]
        o = row["backends"]["ollama"]
        ratio = row["psionic_to_ollama_ratio"]
        lines.append(
            f"| `{row['contract']}` | `{row['model']}` | {p['mean_decode_tok_s']:.2f}±{p['stddev_decode_tok_s']:.2f} | {o['mean_decode_tok_s']:.2f}±{o['stddev_decode_tok_s']:.2f} | {ratio:.2f}x | {p['mean_output_tokens']:.1f}/{o['mean_output_tokens']:.1f} |"
        )
    if not comparable:
        lines.append("| _none_ | _none_ |  |  |  |  |")
    lines.append("")
    lines.append("## Non-Comparable Rows")
    lines.append("")
    lines.append("| Contract | Model | Classification reason | Psionic tokens | Ollama tokens |")
    lines.append("| --- | --- | --- | ---: | ---: |")
    for row in non_comparable:
        p = row["backends"].get("psionic")
        o = row["backends"].get("ollama")
        p_tok = p["mean_output_tokens"] if p else 0.0
        o_tok = o["mean_output_tokens"] if o else 0.0
        lines.append(
            f"| `{row['contract']}` | `{row['model']}` | {row['classification_reason']} | {p_tok:.1f} | {o_tok:.1f} |"
        )
    if not non_comparable:
        lines.append("| _none_ | _none_ |  |  |  |")
    lines.append("")
    lines.append("## Graphs")
    lines.append("")
    lines.append("### Throughput by contract")
    lines.append("")
    lines.append("![Greedy throughput](./throughput_greedy.svg)")
    lines.append("")
    lines.append("![Sampled top-k=40 throughput](./throughput_sampled_topk40.svg)")
    lines.append("")
    lines.append("![Sampled top-k=100 throughput](./throughput_sampled_topk100.svg)")
    lines.append("")
    lines.append("### Ratio overview")
    lines.append("")
    lines.append("![Psionic to Ollama ratio](./psionic_vs_ollama_ratio.svg)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(args.jsonl)
    telemetry_path = Path(args.telemetry_csv) if args.telemetry_csv else None
    records = parse_jsonl(jsonl_path)
    rows, comparable, non_comparable, git_from_runs = summarize(records, args)
    if args.git_commit is None:
        args.git_commit = git_from_runs
    telemetry = parse_telemetry(telemetry_path) if telemetry_path else None

    artifact = {
        "abi_version": "psionic.qwen35.ollama_matrix.v2",
        "run_id": args.run_id,
        "run_date": args.run_date,
        "source": {
            "jsonl_path": str(jsonl_path),
            "raw_log_path": args.raw_log,
            "telemetry_csv_path": args.telemetry_csv,
        },
        "environment": {
            "git_commit": args.git_commit,
            "ollama_version": args.ollama_version,
            "gpu": {
                "name": args.gpu_name,
                "vram_mib": args.gpu_vram_mib,
                "current_power_limit_w": args.gpu_power_limit_w,
                "max_power_limit_w": args.gpu_power_max_limit_w,
            },
            "telemetry": telemetry,
        },
        "comparability_policy": {
            "token_delta_abs_threshold": args.token_delta_abs_threshold,
            "token_delta_ratio_threshold": args.token_delta_ratio_threshold,
        },
        "matrix": {
            "rows": rows,
            "comparable_row_count": len(comparable),
            "non_comparable_row_count": len(non_comparable),
        },
    }

    artifact_name = f"qwen35_ollama_matrix_{args.run_id}.json"
    artifact_path = output_dir / artifact_name
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    report_path = output_dir / "one_page_summary.md"
    write_markdown(report_path, artifact_name, rows, comparable, non_comparable, args)

    by_contract = defaultdict(list)
    for row in rows:
        by_contract[row["contract"]].append(row)
    for contract in CONTRACT_ORDER:
        contract_rows = sorted(
            by_contract.get(contract, []),
            key=lambda row: model_sort_key(row["model"]),
        )
        generate_throughput_svg(output_dir / f"throughput_{contract}.svg", contract, contract_rows)
    generate_ratio_svg(output_dir / "psionic_vs_ollama_ratio.svg", rows)

    print(str(artifact_path))
    print(str(report_path))


if __name__ == "__main__":
    main()
