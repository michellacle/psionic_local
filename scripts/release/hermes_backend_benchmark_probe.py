#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Hermes backend benchmark row against an OpenAI-compatible endpoint."
    )
    parser.add_argument("--hermes-root", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--psionic-root", required=True)
    parser.add_argument("--backend-label", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--psionic-revision", default="")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=float(os.environ.get("PSIONIC_HERMES_REQUEST_TIMEOUT_SECONDS", "60")),
    )
    return parser.parse_args()


def run_git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(args, cwd=str(cwd), text=True).strip()


def import_hermes(hermes_root: Path):
    sys.path.insert(0, str(hermes_root))
    os.chdir(hermes_root)
    from run_agent import AIAgent
    from toolsets import create_custom_toolset
    from tools.registry import registry

    return AIAgent, create_custom_toolset, registry


def first_assistant_tool_turn(messages: list[dict]) -> dict | None:
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return message
    return None


def final_assistant_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


def lower_text(value: str) -> str:
    return value.lower().strip()


def usage_to_dict(usage) -> dict:
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = (
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", 0)
        or 0
    )
    total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
    }


def sum_usage(usages: list[dict]) -> dict:
    return {
        "prompt_tokens": sum(item["prompt_tokens"] for item in usages),
        "completion_tokens": sum(item["completion_tokens"] for item in usages),
        "total_tokens": sum(item["total_tokens"] for item in usages),
    }


def usage_dict_from_response_usage(usage: dict | None) -> dict:
    if not isinstance(usage, dict):
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def collect_tool_call_names(messages: list[dict]) -> list[str]:
    names: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls", []):
            name = tool_call.get("function", {}).get("name")
            if name:
                names.append(name)
    return names


def collect_tool_result_names(messages: list[dict]) -> list[str]:
    tool_call_id_to_name: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls", []):
            call_id = tool_call.get("id")
            name = tool_call.get("function", {}).get("name")
            if call_id and name:
                tool_call_id_to_name[call_id] = name
    names: list[str] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        name = message.get("name")
        if not name:
            name = tool_call_id_to_name.get(message.get("tool_call_id", ""))
        if name:
            names.append(name)
    return names


def run_direct_summary_case(
    *,
    base_url: str,
    model: str,
    system_message: str,
    user_prompt: str,
    request_timeout_seconds: float,
):
    payload = {
        "model": model,
        "temperature": 0,
        "seed": 0,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
    }
    wallclock_start = time.monotonic()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer dummy",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=request_timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        wallclock_s = time.monotonic() - wallclock_start
        usage = usage_dict_from_response_usage(body.get("usage"))
        completion_tok_s = (
            usage["completion_tokens"] / wallclock_s
            if wallclock_s > 0 and usage["completion_tokens"] > 0
            else None
        )
        choice = body.get("choices", [{}])[0]
        message = choice.get("message", {})
        final_text = str(message.get("content") or "")
        return {
            "ok": True,
            "request_payload": payload,
            "response_body": body,
            "wallclock_s": wallclock_s,
            "usage": usage,
            "completion_tok_s": completion_tok_s,
            "final_text": final_text,
            "error_type": None,
            "error_message": None,
        }
    except Exception as exc:
        wallclock_s = time.monotonic() - wallclock_start
        return {
            "ok": False,
            "request_payload": payload,
            "response_body": None,
            "wallclock_s": wallclock_s,
            "usage": usage_dict_from_response_usage(None),
            "completion_tok_s": None,
            "final_text": "",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def run_serialized_two_city_case(
    *,
    model: str,
    base_url: str,
    AIAgent,
    request_timeout_seconds: float,
):
    expected_final_text = "Paris is sunny at 18C. Tokyo is rainy at 12C."
    agent = AIAgent(
        base_url=base_url,
        api_key="dummy",
        provider="custom",
        api_mode="chat_completions",
        model=model,
        enabled_toolsets=["psionic_hermes_backend_serialized_two_city"],
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    obs = {"stream_calls": 0, "deltas": [], "request_summaries": [], "usage_reports": []}
    orig_stream = agent._interruptible_streaming_api_call
    orig_build = agent._build_api_kwargs
    wallclock_start = time.monotonic()
    first_delta_at = {"value": None}

    def wrapped_stream(self, api_kwargs, on_first_delta=None):
        obs["stream_calls"] += 1
        response = orig_stream(api_kwargs, on_first_delta=on_first_delta)
        obs["usage_reports"].append(usage_to_dict(getattr(response, "usage", None)))
        return response

    def wrapped_build(self, api_messages):
        kwargs = orig_build(api_messages)
        tool_result_names = collect_tool_result_names(api_messages)
        distinct_results = []
        for name in tool_result_names:
            if name not in distinct_results:
                distinct_results.append(name)
        remaining_tool_names = [
            name
            for name in ["get_paris_weather", "get_tokyo_weather"]
            if name not in distinct_results
        ]
        if remaining_tool_names:
            kwargs["tools"] = [
                tool
                for tool in kwargs.get("tools", [])
                if tool.get("function", {}).get("name") in remaining_tool_names
            ]
            kwargs["tool_choice"] = "required"
            kwargs["parallel_tool_calls"] = False
        else:
            kwargs["tools"] = []
            kwargs.pop("tool_choice", None)
            kwargs.pop("parallel_tool_calls", None)
        kwargs["temperature"] = 0
        kwargs["seed"] = 0
        obs["request_summaries"].append(
            {
                "tool_choice": kwargs.get("tool_choice"),
                "parallel_tool_calls": kwargs.get("parallel_tool_calls"),
                "tool_results_seen": distinct_results,
                "remaining_tool_names": remaining_tool_names,
                "tool_names": [
                    tool.get("function", {}).get("name")
                    for tool in kwargs.get("tools", [])
                    if isinstance(tool, dict)
                ],
            }
        )
        return kwargs

    agent._interruptible_streaming_api_call = MethodType(wrapped_stream, agent)
    agent._build_api_kwargs = MethodType(wrapped_build, agent)

    def stream_callback(text: str):
        if text:
            obs["deltas"].append(text)
            if first_delta_at["value"] is None:
                first_delta_at["value"] = time.monotonic()

    result = agent.run_conversation(
        "Check Paris weather, then Tokyo weather, then summarize both cities in one answer.",
        system_message=(
            "You are Hermes. Follow this exact plan. First call `get_paris_weather`. "
            "After that tool result is returned, call `get_tokyo_weather`. After both "
            "tool results are returned, answer briefly. If only one tool is currently "
            "offered, that is the only valid next tool."
        ),
        stream_callback=stream_callback,
    )
    tool_loop_wallclock_s = time.monotonic() - wallclock_start
    tool_loop_usage = sum_usage(obs["usage_reports"])

    messages = result.get("messages", [])
    assistant_tool_call_names = collect_tool_call_names(messages)
    tool_result_names = collect_tool_result_names(messages)
    tool_result_texts = [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "tool"
    ]
    distinct_tool_results = []
    for name in tool_result_names:
        if name not in distinct_tool_results:
            distinct_tool_results.append(name)

    summary_case = None
    final_text = ""
    if distinct_tool_results == ["get_paris_weather", "get_tokyo_weather"]:
        summary_case = run_direct_summary_case(
            base_url=base_url,
            model=model,
            system_message=(
                "You are Hermes. Use only the provided tool results. Reply with exactly "
                "the requested final sentence and nothing else."
            ),
            user_prompt=(
                "Return exactly this sentence and nothing else: "
                f"{expected_final_text}\n"
                "Authoritative tool results:\n"
                + "\n".join(f"- {text}" for text in tool_result_texts)
            ),
            request_timeout_seconds=request_timeout_seconds,
        )
        final_text = summary_case["final_text"]

    total_usage = tool_loop_usage
    total_wallclock_s = tool_loop_wallclock_s
    if summary_case is not None:
        total_usage = sum_usage([tool_loop_usage, summary_case["usage"]])
        total_wallclock_s += summary_case["wallclock_s"]
    completion_tok_s = (
        total_usage["completion_tokens"] / total_wallclock_s
        if total_wallclock_s > 0 and total_usage["completion_tokens"] > 0
        else None
    )
    passed = (
        assistant_tool_call_names == ["get_paris_weather", "get_tokyo_weather"]
        and distinct_tool_results == ["get_paris_weather", "get_tokyo_weather"]
        and summary_case is not None
        and summary_case["ok"]
        and final_text.strip() == expected_final_text
    )
    return {
        "case_id": "serialized_two_city_tool_loop",
        "stage": "tool_loop",
        "pass": passed,
        "summary": "serialized two-city controller completed both tools and direct summary closeout",
        "wallclock_s": total_wallclock_s,
        "first_delta_s": (
            first_delta_at["value"] - wallclock_start
            if first_delta_at["value"] is not None
            else None
        ),
        "completion_tok_s": completion_tok_s,
        "usage": total_usage,
        "usage_reports": obs["usage_reports"],
        "stream_calls": obs["stream_calls"],
        "request_summaries": obs["request_summaries"],
        "api_calls": result.get("api_calls"),
        "completed": result.get("completed"),
        "failed": result.get("failed", False),
        "final_response": final_text,
        "messages": messages,
        "details": {
            "assistant_tool_call_sequence": assistant_tool_call_names,
            "tool_result_sequence": tool_result_names,
            "tool_result_texts": tool_result_texts,
            "expected_final_text": expected_final_text,
            "summary_phase": summary_case or {},
            "request_timeout_seconds": request_timeout_seconds,
        },
    }


def run_case(
    *,
    case_id: str,
    stage: str,
    user_prompt: str,
    system_message: str,
    enabled_toolsets: list[str],
    tool_policy,
    max_iterations: int,
    model: str,
    base_url: str,
    AIAgent,
):
    agent = AIAgent(
        base_url=base_url,
        api_key="dummy",
        provider="custom",
        api_mode="chat_completions",
        model=model,
        enabled_toolsets=enabled_toolsets,
        max_iterations=max_iterations,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    obs = {"stream_calls": 0, "deltas": [], "request_summaries": [], "usage_reports": []}
    orig_stream = agent._interruptible_streaming_api_call
    orig_build = agent._build_api_kwargs
    wallclock_start = time.monotonic()
    first_delta_at = {"value": None}

    def wrapped_stream(self, api_kwargs, on_first_delta=None):
        obs["stream_calls"] += 1
        response = orig_stream(api_kwargs, on_first_delta=on_first_delta)
        obs["usage_reports"].append(usage_to_dict(getattr(response, "usage", None)))
        return response

    def wrapped_build(self, api_messages):
        kwargs = orig_build(api_messages)
        tool_choice, parallel_tool_calls = tool_policy(api_messages)
        kwargs["tool_choice"] = tool_choice
        kwargs["parallel_tool_calls"] = parallel_tool_calls
        kwargs["temperature"] = 0
        kwargs["seed"] = 0
        obs["request_summaries"].append(
            {
                "tool_choice": tool_choice,
                "parallel_tool_calls": parallel_tool_calls,
                "tool_names": [
                    tool.get("function", {}).get("name")
                    for tool in kwargs.get("tools", [])
                    if isinstance(tool, dict)
                ],
            }
        )
        return kwargs

    agent._interruptible_streaming_api_call = MethodType(wrapped_stream, agent)
    agent._build_api_kwargs = MethodType(wrapped_build, agent)

    def stream_callback(text: str):
        if text:
            obs["deltas"].append(text)
            if first_delta_at["value"] is None:
                first_delta_at["value"] = time.monotonic()

    result = agent.run_conversation(
        user_prompt,
        system_message=system_message,
        stream_callback=stream_callback,
    )
    wallclock_s = time.monotonic() - wallclock_start
    usage = sum_usage(obs["usage_reports"])
    completion_tok_s = (
        usage["completion_tokens"] / wallclock_s if wallclock_s > 0 and usage["completion_tokens"] > 0 else None
    )
    return {
        "case_id": case_id,
        "stage": stage,
        "stream_calls": obs["stream_calls"],
        "stream_deltas": obs["deltas"],
        "request_summaries": obs["request_summaries"],
        "usage_reports": obs["usage_reports"],
        "usage": usage,
        "wallclock_s": wallclock_s,
        "first_delta_s": (
            first_delta_at["value"] - wallclock_start if first_delta_at["value"] is not None else None
        ),
        "completion_tok_s": completion_tok_s,
        "result": result,
    }


def benchmark_case_result(case: dict, passed: bool, summary: str, details: dict) -> dict:
    result = case["result"]
    return {
        "case_id": case["case_id"],
        "stage": case["stage"],
        "pass": passed,
        "summary": summary,
        "wallclock_s": case["wallclock_s"],
        "first_delta_s": case["first_delta_s"],
        "completion_tok_s": case["completion_tok_s"],
        "usage": case["usage"],
        "usage_reports": case["usage_reports"],
        "stream_calls": case["stream_calls"],
        "request_summaries": case["request_summaries"],
        "api_calls": result.get("api_calls"),
        "completed": result.get("completed"),
        "failed": result.get("failed", False),
        "final_response": result.get("final_response"),
        "messages": result.get("messages", []),
        "details": details,
    }


def main() -> int:
    args = parse_args()
    hermes_root = Path(args.hermes_root).resolve()
    psionic_root = Path(args.psionic_root).resolve()
    report_path = Path(args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    AIAgent, create_custom_toolset, registry = import_hermes(hermes_root)

    psionic_revision = args.psionic_revision or run_git(
        ["git", "rev-parse", "HEAD"], psionic_root
    )
    hermes_revision = run_git(["git", "rev-parse", "HEAD"], hermes_root)

    def required_only(_api_messages):
        return "required", True

    def auto_only(_api_messages):
        return "auto", True

    def required_then_auto(api_messages):
        saw_tool = any(message.get("role") == "tool" for message in api_messages)
        return ("auto" if saw_tool else "required"), True

    reports: list[dict] = []

    registry.register(
        name="get_weather_required",
        toolset="psionic_hermes_backend_required_tool_turn",
        schema={
            "name": "get_weather_required",
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
        handler=lambda tool_args, **kwargs: json.dumps(
            {"city": tool_args.get("city"), "condition": "sunny", "temperature_c": 18}
        ),
        check_fn=lambda: True,
    )
    create_custom_toolset(
        "psionic_hermes_backend_required_tool_turn",
        "Required tool turn benchmark",
        tools=["get_weather_required"],
    )
    case = run_case(
        case_id="required_tool_turn",
        stage="tool_loop",
        user_prompt="Use the weather tool for Paris.",
        system_message="You are Hermes. Use tools when needed.",
        enabled_toolsets=["psionic_hermes_backend_required_tool_turn"],
        tool_policy=required_then_auto,
        max_iterations=2,
        model=args.model,
        base_url=args.base_url,
        AIAgent=AIAgent,
    )
    tool_turn = first_assistant_tool_turn(case["result"].get("messages", []))
    tool_names = [
        tool_call.get("function", {}).get("name")
        for tool_call in (tool_turn or {}).get("tool_calls", [])
    ]
    reports.append(
        benchmark_case_result(
            case,
            passed=tool_names == ["get_weather_required"],
            summary="required tool turn emitted one weather tool call",
            details={"assistant_tool_names": tool_names},
        )
    )

    registry.register(
        name="get_weather_auto",
        toolset="psionic_hermes_backend_auto_text_turn",
        schema={
            "name": "get_weather_auto",
            "description": "Get weather information if explicitly requested.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
        handler=lambda tool_args, **kwargs: json.dumps(
            {"city": tool_args.get("city"), "unused": True}
        ),
        check_fn=lambda: True,
    )
    create_custom_toolset(
        "psionic_hermes_backend_auto_text_turn",
        "Auto text turn benchmark",
        tools=["get_weather_auto"],
    )
    case = run_case(
        case_id="auto_plain_text_turn",
        stage="provider_sanity",
        user_prompt="Say hello in one short sentence. Do not use tools.",
        system_message="You are Hermes. If the user asks for a greeting, answer directly and do not call tools.",
        enabled_toolsets=["psionic_hermes_backend_auto_text_turn"],
        tool_policy=auto_only,
        max_iterations=args.max_iterations,
        model=args.model,
        base_url=args.base_url,
        AIAgent=AIAgent,
    )
    messages = case["result"].get("messages", [])
    tool_turn = first_assistant_tool_turn(messages)
    final_text = final_assistant_text(messages)
    reports.append(
        benchmark_case_result(
            case,
            passed=tool_turn is None and bool(final_text),
            summary="auto mode answered with plain text and no tool call",
            details={"final_text": final_text},
        )
    )

    registry.register(
        name="get_weather_loop",
        toolset="psionic_hermes_backend_tool_loop",
        schema={
            "name": "get_weather_loop",
            "description": "Get the current weather for Paris.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "enum": ["Paris"]}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
        handler=lambda tool_args, **kwargs: json.dumps(
            {"city": tool_args.get("city"), "condition": "sunny", "temperature_c": 22}
        ),
        check_fn=lambda: True,
    )
    create_custom_toolset(
        "psionic_hermes_backend_tool_loop",
        "Tool loop benchmark",
        tools=["get_weather_loop"],
    )
    case = run_case(
        case_id="multi_turn_tool_loop",
        stage="tool_loop",
        user_prompt="Use the weather tool for Paris and then summarize the result.",
        system_message="You are Hermes. Use tools when needed and summarize tool results plainly.",
        enabled_toolsets=["psionic_hermes_backend_tool_loop"],
        tool_policy=required_then_auto,
        max_iterations=args.max_iterations,
        model=args.model,
        base_url=args.base_url,
        AIAgent=AIAgent,
    )
    messages = case["result"].get("messages", [])
    final_text = final_assistant_text(messages)
    saw_tool_result = any(message.get("role") == "tool" for message in messages)
    reports.append(
        benchmark_case_result(
            case,
            passed=saw_tool_result and "paris" in lower_text(final_text),
            summary="multi-turn tool loop replay reached a final answer",
            details={"final_text": final_text, "saw_tool_result": saw_tool_result},
        )
    )

    registry.register(
        name="get_weather_stream",
        toolset="psionic_hermes_backend_streamed_turn",
        schema={
            "name": "get_weather_stream",
            "description": "Get the current weather for Paris.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "enum": ["Paris"]}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
        handler=lambda tool_args, **kwargs: json.dumps(
            {"city": tool_args.get("city"), "condition": "sunny", "temperature_c": 21}
        ),
        check_fn=lambda: True,
    )
    create_custom_toolset(
        "psionic_hermes_backend_streamed_turn",
        "Streamed tool benchmark",
        tools=["get_weather_stream"],
    )
    case = run_case(
        case_id="streamed_tool_turn",
        stage="tool_loop",
        user_prompt="Use the weather tool for Paris.",
        system_message="You are Hermes. Use tools when needed.",
        enabled_toolsets=["psionic_hermes_backend_streamed_turn"],
        tool_policy=required_then_auto,
        max_iterations=2,
        model=args.model,
        base_url=args.base_url,
        AIAgent=AIAgent,
    )
    tool_turn = first_assistant_tool_turn(case["result"].get("messages", []))
    reports.append(
        benchmark_case_result(
            case,
            passed=(
                case["stream_calls"] >= 1
                and tool_turn is not None
                and not case["result"].get("failed", False)
            ),
            summary="streaming chat loop preserved a tool-call turn",
            details={
                "assistant_tool_names": [
                    tool_call.get("function", {}).get("name")
                    for tool_call in (tool_turn or {}).get("tool_calls", [])
                ]
            },
        )
    )

    registry.register(
        name="get_paris_weather",
        toolset="psionic_hermes_backend_serialized_two_city",
        schema={
            "name": "get_paris_weather",
            "description": (
                "Get the weather for Paris only. Use this only for Paris. "
                "Never use this tool for Tokyo."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        handler=lambda tool_args, **kwargs: "Paris is sunny at 18C. Next tool: get_tokyo_weather.",
        check_fn=lambda: True,
    )
    registry.register(
        name="get_tokyo_weather",
        toolset="psionic_hermes_backend_serialized_two_city",
        schema={
            "name": "get_tokyo_weather",
            "description": (
                "Get the weather for Tokyo only. Use this only for Tokyo. "
                "Never use this tool for Paris."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        handler=lambda tool_args, **kwargs: (
            "Tokyo is rainy at 12C. Final answer: Paris is sunny at 18C. "
            "Tokyo is rainy at 12C."
        ),
        check_fn=lambda: True,
    )
    create_custom_toolset(
        "psionic_hermes_backend_serialized_two_city",
        "Serialized two-city benchmark",
        tools=["get_paris_weather", "get_tokyo_weather"],
    )
    reports.append(
        run_serialized_two_city_case(
            model=args.model,
            base_url=args.base_url,
            AIAgent=AIAgent,
            request_timeout_seconds=args.request_timeout_seconds,
        )
    )

    overall_pass = all(case["pass"] for case in reports)
    mean_completion_tok_s = [
        case["completion_tok_s"] for case in reports if case["completion_tok_s"] is not None
    ]
    output = {
        "report_kind": "psionic_hermes_backend_benchmark_row",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend_label": args.backend_label,
        "base_url": args.base_url,
        "model": args.model,
        "model_path": args.model_path,
        "psionic_root": str(psionic_root),
        "psionic_revision": psionic_revision,
        "hermes_root": str(hermes_root),
        "hermes_revision": hermes_revision,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "host": platform.node(),
        "cases": reports,
        "overall_pass": overall_pass,
        "passing_case_count": sum(1 for case in reports if case["pass"]),
        "total_case_count": len(reports),
        "mean_case_wallclock_s": (
            sum(case["wallclock_s"] for case in reports) / len(reports) if reports else None
        ),
        "mean_completion_tok_s": (
            sum(mean_completion_tok_s) / len(mean_completion_tok_s) if mean_completion_tok_s else None
        ),
    }
    report_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
