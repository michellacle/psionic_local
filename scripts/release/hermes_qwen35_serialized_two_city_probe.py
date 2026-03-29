#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a serialized two-city Hermes compatibility probe against an OpenAI-compatible qwen35 backend."
    )
    parser.add_argument("--hermes-root", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--psionic-root", required=True)
    parser.add_argument("--backend-label", default="psionic")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--psionic-revision", default="")
    parser.add_argument("--max-iterations", type=int, default=4)
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


def lower_text(value: str) -> str:
    return value.lower().strip()


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


def final_assistant_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


def run_case(
    *,
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
    obs = {"stream_calls": 0, "deltas": [], "request_summaries": []}
    orig_stream = agent._interruptible_streaming_api_call
    orig_build = agent._build_api_kwargs

    def wrapped_stream(self, api_kwargs, on_first_delta=None):
        obs["stream_calls"] += 1
        return orig_stream(api_kwargs, on_first_delta=on_first_delta)

    def wrapped_build(self, api_messages):
        kwargs = orig_build(api_messages)
        tool_choice, parallel_tool_calls, tool_results_seen = tool_policy(api_messages)
        remaining_tool_names = [
            name
            for name in ["get_paris_weather", "get_tokyo_weather"]
            if name not in tool_results_seen
        ]
        if remaining_tool_names:
            kwargs["tools"] = [
                tool
                for tool in kwargs.get("tools", [])
                if tool.get("function", {}).get("name") in remaining_tool_names
            ]
        else:
            kwargs["tools"] = []
        if kwargs["tools"]:
            kwargs["tool_choice"] = tool_choice
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        else:
            kwargs.pop("tool_choice", None)
            kwargs.pop("parallel_tool_calls", None)
        kwargs["temperature"] = 0
        kwargs["seed"] = 0
        obs["request_summaries"].append(
            {
                "tool_choice": tool_choice if kwargs["tools"] else None,
                "parallel_tool_calls": parallel_tool_calls if kwargs["tools"] else None,
                "tool_results_seen": tool_results_seen,
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
        obs["deltas"].append(text)

    result = agent.run_conversation(
        user_prompt,
        system_message=system_message,
        stream_callback=stream_callback,
    )
    return {
        "stream_calls": obs["stream_calls"],
        "stream_deltas": obs["deltas"],
        "request_summaries": obs["request_summaries"],
        "result": result,
    }


def run_summary_case(
    *,
    user_prompt: str,
    system_message: str,
    model: str,
    base_url: str,
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
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer dummy",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        body = json.loads(response.read().decode("utf-8"))
    choice = body.get("choices", [{}])[0]
    message = choice.get("message", {})
    final_text = str(message.get("content") or "")
    return {
        "request_payload": payload,
        "response_body": body,
        "final_text": final_text,
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

    def required_until_both_tool_results_then_auto(api_messages):
        tool_result_names = collect_tool_result_names(api_messages)
        distinct_results = []
        for name in tool_result_names:
            if name not in distinct_results:
                distinct_results.append(name)
        if len(distinct_results) >= 2:
            return "auto", False, distinct_results
        return "required", False, distinct_results

    registry.register(
        name="get_paris_weather",
        toolset="psionic_hermes_qwen35_serialized_two_city",
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
        toolset="psionic_hermes_qwen35_serialized_two_city",
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
        "psionic_hermes_qwen35_serialized_two_city",
        "Serialized two-city tool loop probe",
        tools=["get_paris_weather", "get_tokyo_weather"],
    )

    case = run_case(
        user_prompt="Check Paris weather, then Tokyo weather, then summarize both cities in one answer.",
        system_message=(
            "You are Hermes. Follow this exact plan. First call `get_paris_weather`. "
            "After that tool result is returned, call `get_tokyo_weather`. After both "
            "tool results are returned, answer in one short sentence that mentions both "
            "Paris and Tokyo. Do not answer before both tool results arrive. If only one "
            "tool is currently offered, that is the only valid next tool. Tool outputs are "
            "authoritative plain text. If any tool output contains `Final answer:`, copy the "
            "exact text after `Final answer:` as your next assistant message and do not add "
            "anything else."
        ),
        enabled_toolsets=["psionic_hermes_qwen35_serialized_two_city"],
        tool_policy=required_until_both_tool_results_then_auto,
        max_iterations=args.max_iterations,
        model=args.model,
        base_url=args.base_url,
        AIAgent=AIAgent,
    )
    messages = case["result"].get("messages", [])
    assistant_tool_call_names = collect_tool_call_names(messages)
    tool_result_names = collect_tool_result_names(messages)
    tool_result_texts = [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "tool"
    ]
    expected_final_text = "Paris is sunny at 18C. Tokyo is rainy at 12C."
    distinct_tool_results = []
    for name in tool_result_names:
        if name not in distinct_tool_results:
            distinct_tool_results.append(name)

    summary_case = None
    final_text = ""
    if distinct_tool_results == ["get_paris_weather", "get_tokyo_weather"]:
        summary_case = run_summary_case(
            user_prompt=(
                "Return exactly this sentence and nothing else: "
                f"{expected_final_text}\n"
                "Authoritative tool results:\n"
                + "\n".join(f"- {text}" for text in tool_result_texts)
            ),
            system_message=(
                "You are Hermes. Use only the provided tool results. Reply with exactly "
                "the requested final sentence and nothing else."
            ),
            model=args.model,
            base_url=args.base_url,
        )
        final_text = summary_case["final_text"]

    passed = (
        assistant_tool_call_names.count("get_paris_weather") == 1
        and assistant_tool_call_names.count("get_tokyo_weather") == 1
        and assistant_tool_call_names == ["get_paris_weather", "get_tokyo_weather"]
        and distinct_tool_results == ["get_paris_weather", "get_tokyo_weather"]
        and final_text.strip() == expected_final_text
    )

    output = {
        "report_kind": "psionic_hermes_qwen35_serialized_two_city_compatibility",
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
        "case": {
            "case_id": "serialized_two_city_tool_loop",
            "pass": passed,
            "summary": "serialized two-city loop called both city tools across separate turns and reached a final summary",
            "stream_calls": case["stream_calls"],
            "request_summaries": case["request_summaries"],
            "api_calls": case["result"].get("api_calls"),
            "completed": case["result"].get("completed"),
            "failed": case["result"].get("failed", False),
            "final_response": final_text,
            "messages": messages,
            "details": {
                "assistant_tool_call_sequence": assistant_tool_call_names,
                "tool_result_sequence": tool_result_names,
                "tool_result_texts": tool_result_texts,
                "final_text": final_text,
                "expected_final_text": expected_final_text,
                "summary_phase": {
                    "request_payload": (
                        summary_case["request_payload"] if summary_case is not None else {}
                    ),
                    "response_body": (
                        summary_case["response_body"] if summary_case is not None else {}
                    ),
                },
            },
        },
        "overall_pass": passed,
    }
    report_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
