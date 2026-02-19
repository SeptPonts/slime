"""Custom rollout logic for holdem agentic RL training."""

from __future__ import annotations

import json
import re
from typing import Any

from holdem_core import AGENTIC_SYSTEM_PROMPT, POKER_TOOLS, HoldemGameState, execute_tool
from slime.utils.http_utils import post
from slime.utils.types import Sample

DEFAULT_MAX_TURNS = 10


def _get_max_turns(args) -> int:
    return int(getattr(args, "poker_max_turns", DEFAULT_MAX_TURNS))


def _format_prompt() -> str:
    tools_text = "\n".join([json.dumps(spec, ensure_ascii=True) for spec in POKER_TOOLS])
    return f"""<|im_start|>system
{AGENTIC_SYSTEM_PROMPT}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_text}
</tools>

For each function call, return a json object with function name and arguments
within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
<|im_end|>
<|im_start|>user
It's your turn to act. What do you do?<|im_end|>
<|im_start|>assistant
"""


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    segments = re.findall(pattern, text, re.DOTALL)

    decoder = json.JSONDecoder()
    tool_calls: list[dict[str, Any]] = []
    for segment in segments:
        start = segment.find("{")
        if start == -1:
            continue
        try:
            obj, _ = decoder.raw_decode(segment[start:].lstrip())
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict):
            continue

        name = str(obj.get("name", "")).strip()
        arguments = obj.get("arguments", {})
        if not name:
            continue
        if not isinstance(arguments, dict):
            arguments = {}

        tool_calls.append({"name": name, "arguments": arguments})

    return tool_calls


def _extract_output_tokens_and_logprobs(output: dict, tokenizer) -> tuple[list[int], list[float]]:
    meta_info = output.get("meta_info", {})
    entries = meta_info.get("output_token_logprobs") or []

    token_ids: list[int] = []
    log_probs: list[float] = []
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        try:
            log_probs.append(float(entry[0]))
            token_ids.append(int(entry[1]))
        except (TypeError, ValueError):
            continue

    if token_ids:
        return token_ids, log_probs

    text = output.get("text", "")
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    log_probs = [0.0] * len(token_ids)
    return token_ids, log_probs


def _init_or_align_response_fields(sample: Sample) -> None:
    if sample.loss_mask is None:
        sample.loss_mask = [1] * sample.response_length
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = [0.0] * sample.response_length

    if len(sample.loss_mask) != sample.response_length:
        if len(sample.loss_mask) < sample.response_length:
            sample.loss_mask = sample.loss_mask + [1] * (sample.response_length - len(sample.loss_mask))
        else:
            sample.loss_mask = sample.loss_mask[: sample.response_length]

    if len(sample.rollout_log_probs) != sample.response_length:
        if len(sample.rollout_log_probs) < sample.response_length:
            sample.rollout_log_probs = sample.rollout_log_probs + [0.0] * (
                sample.response_length - len(sample.rollout_log_probs)
            )
        else:
            sample.rollout_log_probs = sample.rollout_log_probs[: sample.response_length]


async def generate(args, sample: Sample, sampling_params) -> Sample:
    from slime.rollout.sglang_rollout import GenerateState

    if not isinstance(sample.prompt, str):
        sample.status = Sample.Status.FAILED
        sample.metadata["generate_error"] = "sample.prompt must be str for holdem rollout"
        return sample

    try:
        game_state = HoldemGameState.from_dataset(
            {"instruction": sample.prompt, "output": sample.label or ""},
            player_id=0,
        )
    except Exception as exc:
        sample.status = Sample.Status.FAILED
        sample.metadata["generate_error"] = f"failed to parse game state: {exc}"
        return sample

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    base_prompt = _format_prompt()
    base_prompt_token_ids = state.tokenizer(base_prompt, add_special_tokens=False)["input_ids"]

    if not sample.tokens:
        sample.tokens = list(base_prompt_token_ids)
        sample.response = ""
        sample.response_length = 0
        sample.loss_mask = []
        sample.rollout_log_probs = []
    else:
        _init_or_align_response_fields(sample)

    max_turns = _get_max_turns(args)

    for turn_idx in range(max_turns):
        output = await post(
            url,
            {
                "input_ids": sample.tokens,
                "sampling_params": sampling_params,
                "return_logprob": True,
            },
        )

        finish_reason = output.get("meta_info", {}).get("finish_reason", {}).get("type", "stop")
        if finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_text = output.get("text", "")
        cur_token_ids, cur_log_probs = _extract_output_tokens_and_logprobs(output, state.tokenizer)

        if sample.loss_mask is None:
            sample.loss_mask = []
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []

        sample.tokens += cur_token_ids
        sample.response += cur_text
        sample.response_length += len(cur_token_ids)
        sample.loss_mask += [1] * len(cur_token_ids)
        sample.rollout_log_probs += cur_log_probs

        if finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
            sample.metadata["tool_turns"] = turn_idx + 1
            return sample

        tool_calls = _parse_tool_calls(cur_text)
        if not tool_calls:
            sample.status = Sample.Status.COMPLETED
            sample.metadata["tool_turns"] = turn_idx + 1
            return sample

        finished_by_action = False
        for tool_call in tool_calls:
            result = await execute_tool(
                tool_call["name"],
                game_state,
                tool_call["arguments"],
            )
            if result is None:
                finished_by_action = True
                break

            obs = "\n" + json.dumps(result, ensure_ascii=True) + "\n"
            obs_token_ids = state.tokenizer(obs, add_special_tokens=False)["input_ids"]
            sample.tokens += obs_token_ids
            sample.response += obs
            sample.response_length += len(obs_token_ids)
            sample.loss_mask += [0] * len(obs_token_ids)
            sample.rollout_log_probs += [0.0] * len(obs_token_ids)

        if finished_by_action:
            sample.status = Sample.Status.COMPLETED
            sample.metadata["tool_turns"] = turn_idx + 1
            return sample

    sample.status = Sample.Status.TRUNCATED
    sample.metadata["max_turns_reached"] = max_turns
    return sample


__all__ = ["generate"]

