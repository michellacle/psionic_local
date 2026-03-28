#!/usr/bin/env python3
import argparse
import json
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe qwen35 Hermes-equivalent tool turns for fast-path versus fallback truth."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--psionic-root", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--psionic-revision", default="")
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
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            text = response.read().decode("utf-8")
            return {
                "http_status": response.status,
                "headers": request_case_headers(response.headers),
                "payload": json.loads(text),
                "error": None,
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
            "path_class": "missing_metrics",
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
    raw_logits = bool(decode.get("raw_logits_materialized", False))
    return {
        "present": True,
        "output_modes": output_modes,
        "readback_bytes": decode.get("readback_bytes"),
        "raw_logits_materialized": raw_logits,
        "path_class": "dense_fallback"
        if raw_logits or "raw_logits" in output_modes
        else "bounded_fast_path",
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


def make_weather_tool() -> dict:
    return {
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


def make_parallel_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_paris_weather",
                "description": "Get the current Paris weather.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_tokyo_weather",
                "description": "Get the current Tokyo weather.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    ]


def summarize_response(label: str, response: dict) -> dict:
    payload = response["payload"]
    return {
        "request_label": label,
        "http_status": response["http_status"],
        "psionic_headers": response["headers"],
        "decode_metrics": decode_metrics_summary(payload),
        "tool_call_names": tool_call_names(payload),
        "assistant_text": assistant_text(payload),
        "finish_reason": (
            payload.get("choices", [{}])[0].get("finish_reason")
            if isinstance(payload.get("choices"), list)
            else None
        ),
        "error": response["error"],
        "payload": payload,
    }


def supported_fast_path_case(case_id: str, requests: list[dict], summary: str, details: dict) -> dict:
    all_bounded = all(
        request["decode_metrics"]["path_class"] == "bounded_fast_path" for request in requests
    )
    return {
        "case_id": case_id,
        "envelope_state": "supported_fast_path",
        "pass": all_bounded and details.get("behavior_ok", False),
        "summary": summary,
        "requests": requests,
        "details": details,
    }


def boundary_case(case_id: str, requests: list[dict], summary: str, details: dict) -> dict:
    return {
        "case_id": case_id,
        "envelope_state": "outside_supported_envelope",
        "pass": details.get("boundary_truthful", False),
        "summary": summary,
        "requests": requests,
        "details": details,
    }


def main() -> int:
    args = parse_args()
    report_path = Path(args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    psionic_root = Path(args.psionic_root).resolve()
    psionic_revision = args.psionic_revision or run_git(["git", "rev-parse", "HEAD"], psionic_root)

    cases: list[dict] = []

    required_body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are Hermes. Use tools when needed."},
            {"role": "user", "content": "Use the weather tool for Paris."},
        ],
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": 128,
        "stream": False,
        "tools": [make_weather_tool()],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }
    required_response = summarize_response(
        "required_tool_turn",
        post_chat_completion(
            base_url=args.base_url,
            body=required_body,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    cases.append(
        supported_fast_path_case(
            "required_tool_turn_fast_path",
            [required_response],
            "required tool turn stayed on the bounded fast path",
            {
                "behavior_ok": required_response["tool_call_names"] == ["get_weather"],
                "expected_tool_call_names": ["get_weather"],
            },
        )
    )

    auto_body = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "You are Hermes. If the user asks for a greeting, answer directly and do not call tools.",
            },
            {"role": "user", "content": "Say hello in one short sentence. Do not use tools."},
        ],
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": 64,
        "stream": False,
        "tools": [make_weather_tool()],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    }
    auto_response = summarize_response(
        "auto_plain_text_turn",
        post_chat_completion(
            base_url=args.base_url,
            body=auto_body,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    cases.append(
        supported_fast_path_case(
            "auto_message_fast_path",
            [auto_response],
            "auto mode answered directly and stayed on the bounded fast path",
            {
                "behavior_ok": not auto_response["tool_call_names"] and bool(auto_response["assistant_text"]),
                "expected_tool_call_names": [],
            },
        )
    )

    tool_call = (
        required_response["payload"].get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
    )
    continuation_body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are Hermes. Use tools when needed."},
            {"role": "user", "content": "Use the weather tool for Paris."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_call,
            },
            {
                "role": "tool",
                "tool_call_id": tool_call[0]["id"] if tool_call else "missing-tool-call-id",
                "content": json.dumps({"city": "Paris", "condition": "sunny", "temperature_c": 18}),
            },
        ],
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": 128,
        "stream": False,
        "tools": [make_weather_tool()],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    }
    continuation_response = summarize_response(
        "tool_result_continuation",
        post_chat_completion(
            base_url=args.base_url,
            body=continuation_body,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    cases.append(
        supported_fast_path_case(
            "tool_result_continuation_fast_path",
            [required_response, continuation_response],
            "tool-result continuation stayed on the bounded fast path",
            {
                "behavior_ok": "paris" in continuation_response["assistant_text"].lower(),
                "expected_final_text_contains": "paris",
            },
        )
    )

    parallel_body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are Hermes. Call both tools in the same turn."},
            {"role": "user", "content": "Call both weather tools, one for Paris and one for Tokyo."},
        ],
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": 192,
        "stream": False,
        "tools": make_parallel_tools(),
        "tool_choice": "required",
        "parallel_tool_calls": True,
    }
    parallel_response = summarize_response(
        "parallel_tool_turn",
        post_chat_completion(
            base_url=args.base_url,
            body=parallel_body,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    observed_parallel_names = parallel_response["tool_call_names"]
    cases.append(
        boundary_case(
            "parallel_tool_turn_model_boundary",
            [parallel_response],
            "same-turn parallel tools are still outside the proven qwen35 fast-path envelope",
            {
                "boundary_truthful": observed_parallel_names != ["get_paris_weather", "get_tokyo_weather"],
                "expected_tool_call_names": ["get_paris_weather", "get_tokyo_weather"],
                "observed_tool_call_names": observed_parallel_names,
                "degradation_reason": "current qwen35 row still emits fewer than two tools on the same-turn parallel request",
            },
        )
    )

    mirostat_body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are Hermes. Use tools when needed."},
            {"role": "user", "content": "Use the weather tool for Paris."},
        ],
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.9,
        "mirostat": 1,
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1,
        "seed": 42,
        "max_tokens": 128,
        "stream": False,
        "tools": [make_weather_tool()],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }
    mirostat_response = summarize_response(
        "mirostat_required_tool_turn",
        post_chat_completion(
            base_url=args.base_url,
            body=mirostat_body,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    cases.append(
        boundary_case(
            "mirostat_tool_turn_dense_fallback",
            [mirostat_response],
            "mirostat tool turn leaves the bounded lane and materializes dense logits",
            {
                "boundary_truthful": mirostat_response["decode_metrics"]["path_class"] == "dense_fallback",
                "expected_path_class": "dense_fallback",
                "observed_path_class": mirostat_response["decode_metrics"]["path_class"],
                "degradation_reason": "mirostat currently routes qwen35 CUDA tool turns through raw-logit fallback",
            },
        )
    )

    supported_cases = [case for case in cases if case["envelope_state"] == "supported_fast_path"]
    fallback_cases = [
        case
        for case in cases
        if any(
            request["decode_metrics"]["path_class"] == "dense_fallback" for request in case["requests"]
        )
    ]
    boundary_cases = [case for case in cases if case["envelope_state"] == "outside_supported_envelope"]

    report = {
        "report_kind": "psionic_hermes_qwen35_fast_path_benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "base_url": args.base_url,
        "model": args.model,
        "model_path": args.model_path,
        "psionic_root": str(psionic_root),
        "psionic_revision": psionic_revision,
        "supported_fast_path_envelope": {
            "api_mode": "chat_completions",
            "include_debug_fields": True,
            "tool_selection": "temperature=0, seed=0, tool_choice=required or auto, no structured outputs",
            "tool_result_continuation": "assistant tool_call replay plus tool result replay on chat.completions",
            "bounded_decode_modes": ["argmax_only", "top_k_candidates:*", "sparse_logits:*"],
            "dense_fallback_mode": "raw_logits",
            "current_outside_envelope_rows": [
                "parallel_tool_turn_model_boundary",
                "mirostat_tool_turn_dense_fallback",
            ],
        },
        "fast_path_health": {
            "supported_case_count": len(supported_cases),
            "supported_cases_all_bounded": all(case["pass"] for case in supported_cases),
            "dense_fallback_case_count": len(fallback_cases),
            "outside_envelope_case_count": len(boundary_cases),
            "current_status": "bounded_fast_path_green"
            if all(case["pass"] for case in supported_cases)
            else "partial",
        },
        "cases": cases,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
