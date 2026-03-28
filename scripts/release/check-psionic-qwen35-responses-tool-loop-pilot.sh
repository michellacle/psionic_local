#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

cargo test -p psionic-serve --lib generic_responses_qwen35_tool_result_messages_preserve_role_and_name_through_prompt_conversion -- --test-threads=1
cargo test -p psionic-serve --lib generic_responses_native_qwen35_tool_turn_stores_replayable_response_state -- --test-threads=1
cargo test -p psionic-serve --lib generic_responses_native_qwen35_tool_result_replay_reaches_final_assistant_answer -- --test-threads=1
