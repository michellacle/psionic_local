#!/usr/bin/env python3
import argparse
import json
import platform
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark repeated Hermes-equivalent qwen35 tool loops with prefix cache auto versus bypass."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--psionic-root", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--psionic-revision", default="")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def run_git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(args, cwd=str(cwd), text=True).strip()


def request_case_headers(headers) -> dict[str, str]:
    keep = [
        "x-psionic-backend",
        "x-psionic-execution-mode",
        "x-psionic-execution-engine",
        "x-psionic-residency-mode",
        "x-psionic-fallback-policy",
        "x-psionic-performance-class",
        "x-psionic-prefix-cache-state",
        "x-psionic-prefix-cache-refusal",
        "x-psionic-prefix-cache-reused-tokens",
        "x-psionic-ttft-ns",
        "x-psionic-itl-ns",
    ]
    output: dict[str, str] = {}
    for name in keep:
        value = headers.get(name)
        if value is not None:
            output[name] = value
    return output


def post_chat_completion(
    *,
    base_url: str,
    body: dict,
    timeout_seconds: float,
) -> dict:
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Authorization": "Bearer dummy",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            text = response.read().decode("utf-8")
            wallclock_s = time.perf_counter() - started
            return {
                "http_status": response.status,
                "headers": request_case_headers(response.headers),
                "payload": json.loads(text),
                "error": None,
                "wallclock_s": wallclock_s,
            }
    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8")
        try:
            payload_json = json.loads(text)
        except json.JSONDecodeError:
            payload_json = {"raw_body": text}
        return {
            "http_status": error.code,
            "headers": request_case_headers(error.headers),
            "payload": payload_json,
            "error": payload_json.get("error", {}).get("message", text),
            "wallclock_s": time.perf_counter() - started,
        }


def decode_metrics_summary(payload: dict) -> dict:
    metrics = payload.get("psionic_metrics") or {}
    decode = metrics.get("qwen35_cuda_decode")
    if decode is None:
        return {
            "present": False,
            "output_modes": [],
            "readback_bytes": None,
            "raw_logits_materialized": None,
        }
    output_modes = []
    for mode in decode.get("output_modes", []):
        if isinstance(mode, dict):
            kind = mode.get("kind", "unknown")
            if kind == "top_k_candidates":
                output_modes.append(f"top_k_candidates:{mode.get('top_k')}")
            elif kind == "sparse_logits":
                output_modes.append(f"sparse_logits:{mode.get('token_count')}")
            else:
                output_modes.append(str(kind))
        else:
            output_modes.append(str(mode))
    return {
        "present": True,
        "output_modes": output_modes,
        "readback_bytes": decode.get("readback_bytes"),
        "raw_logits_materialized": decode.get("raw_logits_materialized"),
    }


def tool_call_names(payload: dict) -> list[str]:
    tool_calls = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("tool_calls")
    ) or []
    names: list[str] = []
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def assistant_text(payload: dict) -> str:
    content = payload.get("choices", [{}])[0].get("message", {}).get("content")
    return content if isinstance(content, str) else ""


def prefix_tokens_reused(headers: dict[str, str]) -> int:
    value = headers.get("x-psionic-prefix-cache-reused-tokens")
    if value is None:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def ttft_ms(headers: dict[str, str]) -> float | None:
    value = headers.get("x-psionic-ttft-ns")
    if value is None:
        return None
    try:
        return int(value) / 1_000_000.0
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def build_prefix_cache(mode: str) -> dict:
    return {"mode": mode, "tenant_id": "hermes-qwen35-reuse"}


def required_tool_body(model: str, mode: str) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are Hermes. Use tools when needed."},
            {"role": "user", "content": "Use the weather tool for Paris."},
        ],
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": 128,
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current Paris weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "enum": ["Paris"]},
                        },
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "psionic_prefix_cache": build_prefix_cache(mode),
    }


def continuation_body(model: str, mode: str) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are Hermes. Use tools when needed."},
            {"role": "user", "content": "Use the weather tool for Paris."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Paris\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": json.dumps(
                    {"city": "Paris", "condition": "sunny", "temperature_c": 18}
                ),
            },
        ],
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": 128,
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current Paris weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "enum": ["Paris"]},
                        },
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "psionic_prefix_cache": build_prefix_cache(mode),
    }


def summarize_iteration(label: str, response: dict) -> dict:
    payload = response["payload"]
    return {
        "request_label": label,
        "http_status": response["http_status"],
        "error": response["error"],
        "wallclock_s": response["wallclock_s"],
        "psionic_headers": response["headers"],
        "prefix_cache_state": response["headers"].get("x-psionic-prefix-cache-state"),
        "prefix_cache_refusal": response["headers"].get("x-psionic-prefix-cache-refusal"),
        "prefix_tokens_reused": prefix_tokens_reused(response["headers"]),
        "ttft_ms": ttft_ms(response["headers"]),
        "decode_metrics": decode_metrics_summary(payload),
        "tool_call_names": tool_call_names(payload),
        "assistant_text": assistant_text(payload),
        "finish_reason": (
            payload.get("choices", [{}])[0].get("finish_reason")
            if isinstance(payload.get("choices"), list)
            else None
        ),
        "payload": payload,
    }


def aggregate_iterations(iterations: list[dict]) -> dict:
    warm = iterations[1:] if len(iterations) > 1 else iterations
    return {
        "iteration_count": len(iterations),
        "mean_wallclock_s_all": mean([row["wallclock_s"] for row in iterations]),
        "mean_wallclock_s_warm": mean([row["wallclock_s"] for row in warm]),
        "mean_ttft_ms_all": mean(
            [row["ttft_ms"] for row in iterations if row["ttft_ms"] is not None]
        ),
        "mean_ttft_ms_warm": mean(
            [row["ttft_ms"] for row in warm if row["ttft_ms"] is not None]
        ),
        "mean_prefix_tokens_reused_all": mean(
            [float(row["prefix_tokens_reused"]) for row in iterations]
        ),
        "mean_prefix_tokens_reused_warm": mean(
            [float(row["prefix_tokens_reused"]) for row in warm]
        ),
    }


def percent_improvement(baseline: float | None, improved: float | None) -> float | None:
    if baseline is None or improved is None or baseline <= 0.0:
        return None
    return ((baseline - improved) / baseline) * 100.0


def main() -> int:
    args = parse_args()
    report_path = Path(args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    psionic_root = Path(args.psionic_root).resolve()
    psionic_revision = args.psionic_revision or run_git(["git", "rev-parse", "HEAD"], psionic_root)

    row_reports = []
    for mode in ["bypass", "auto"]:
        required_iterations = []
        continuation_iterations = []
        for iteration in range(args.iterations):
            required_response = summarize_iteration(
                f"required_tool_turn_loop_{iteration}",
                post_chat_completion(
                    base_url=args.base_url,
                    body=required_tool_body(args.model, mode),
                    timeout_seconds=args.timeout_seconds,
                ),
            )
            required_iterations.append(required_response)
            continuation_response = summarize_iteration(
                f"tool_result_continuation_loop_{iteration}",
                post_chat_completion(
                    base_url=args.base_url,
                    body=continuation_body(args.model, mode),
                    timeout_seconds=args.timeout_seconds,
                ),
            )
            continuation_iterations.append(continuation_response)

        required_ok = all(
            row["http_status"] == 200 and row["tool_call_names"] == ["get_weather"]
            for row in required_iterations
        )
        continuation_ok = all(
            row["http_status"] == 200 and "paris" in row["assistant_text"].lower()
            for row in continuation_iterations
        )
        row_reports.append(
            {
                "row_id": mode,
                "pass": required_ok and continuation_ok,
                "required_tool_turn": {
                    "iterations": required_iterations,
                    "aggregate": aggregate_iterations(required_iterations),
                },
                "tool_result_continuation": {
                    "iterations": continuation_iterations,
                    "aggregate": aggregate_iterations(continuation_iterations),
                },
            }
        )

    row_by_id = {row["row_id"]: row for row in row_reports}
    bypass = row_by_id["bypass"]
    auto = row_by_id["auto"]
    comparison = {
        "required_tool_turn_warm_wallclock_improvement_pct": percent_improvement(
            bypass["required_tool_turn"]["aggregate"]["mean_wallclock_s_warm"],
            auto["required_tool_turn"]["aggregate"]["mean_wallclock_s_warm"],
        ),
        "tool_result_continuation_warm_wallclock_improvement_pct": percent_improvement(
            bypass["tool_result_continuation"]["aggregate"]["mean_wallclock_s_warm"],
            auto["tool_result_continuation"]["aggregate"]["mean_wallclock_s_warm"],
        ),
        "required_tool_turn_warm_ttft_improvement_pct": percent_improvement(
            bypass["required_tool_turn"]["aggregate"]["mean_ttft_ms_warm"],
            auto["required_tool_turn"]["aggregate"]["mean_ttft_ms_warm"],
        ),
        "tool_result_continuation_warm_ttft_improvement_pct": percent_improvement(
            bypass["tool_result_continuation"]["aggregate"]["mean_ttft_ms_warm"],
            auto["tool_result_continuation"]["aggregate"]["mean_ttft_ms_warm"],
        ),
    }
    report = {
        "report_kind": "psionic_hermes_qwen35_reuse_benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "base_url": args.base_url,
        "model": args.model,
        "model_path": args.model_path,
        "psionic_root": str(psionic_root),
        "psionic_revision": psionic_revision,
        "iterations_per_row": args.iterations,
        "workload_shape": {
            "api_mode": "chat_completions",
            "provider_shape": "direct Hermes-equivalent custom-provider loop",
            "turns_per_loop": [
                "required_tool_turn",
                "tool_result_continuation",
            ],
            "control_row": "psionic_prefix_cache=bypass",
            "improved_row": "psionic_prefix_cache=auto",
        },
        "rows": row_reports,
        "comparison": comparison,
        "overall_pass": all(row["pass"] for row in row_reports),
        "improvement_status": "improved"
        if (comparison["required_tool_turn_warm_wallclock_improvement_pct"] or 0.0) > 0.0
        and (comparison["tool_result_continuation_warm_wallclock_improvement_pct"] or 0.0) > 0.0
        else "not_improved",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
