"""Custom reward logic for holdem agentic RL training."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from holdem_core import HoldemGameState, openspiel_chips_to_dataset_decimal

from holdem_reward_config import ACTION_TYPE_DISTANCE, RewardWeights


@dataclass(frozen=True)
class PokerAction:
    type: Literal["fold", "check", "call", "bet", "raise"]
    amount: float | None = None

    def __str__(self) -> str:
        if self.amount is None:
            return self.type
        return f"{self.type} {self.amount:.1f}"


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


def extract_tool_calls(response: str) -> list[tuple[str, dict[str, Any]]]:
    calls = _parse_tool_calls(response)
    return [(item["name"], item["arguments"]) for item in calls]


def parse_action_string(action_str: str) -> PokerAction:
    text = (action_str or "").strip().lower()
    if not text:
        raise ValueError("empty action string")

    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
            action_type = str(obj.get("action_type", "")).strip().lower()
            amount = obj.get("amount")
            if action_type in {"fold", "check", "call", "bet", "raise"}:
                if amount is not None:
                    return PokerAction(action_type, float(amount))
                return PokerAction(action_type, None)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    match = re.search(r"\b(fold|check|call|bet|raise)\b(?:\s*(?:to|for|:)?\s*([0-9]+(?:\.[0-9]+)?))?", text)
    if not match:
        raise ValueError(f"cannot parse action: {action_str!r}")

    action_type = match.group(1)
    amount_group = match.group(2)
    if action_type in {"fold", "check", "call"}:
        return PokerAction(action_type, None)
    if amount_group is None:
        return PokerAction(action_type, None)
    return PokerAction(action_type, float(amount_group))


def extract_action_from_trace(tool_calls: list[tuple[str, dict[str, Any]]]) -> PokerAction | None:
    make_action_calls = [args for name, args in tool_calls if name == "make_action"]
    if not make_action_calls:
        return None

    last_call = make_action_calls[-1]
    action_type = str(last_call.get("action_type", "")).strip().lower()
    if action_type not in {"fold", "check", "call", "bet", "raise"}:
        return None

    amount = last_call.get("amount")
    if amount is None:
        return PokerAction(action_type, None)
    try:
        return PokerAction(action_type, float(amount))
    except (TypeError, ValueError):
        return PokerAction(action_type, None)


def amount_distance(pred: float, gt: float, legal_range: tuple[float, float]) -> float:
    min_legal, max_legal = legal_range

    if pred < min_legal:
        return 1.0 + (min_legal - pred) / max(min_legal, 1e-6)
    if pred > max_legal:
        return 1.0 + (pred - max_legal) / max(max_legal, 1e-6)

    if pred == gt:
        return 0.0
    relative_error = abs(pred - gt) / max(abs(pred), abs(gt), 1e-6)
    return min(relative_error, 1.0)


def action_quality_score(
    pred: PokerAction | None,
    gt: PokerAction,
    legal_range: tuple[float, float],
) -> tuple[float, dict[str, Any]]:
    details: dict[str, Any] = {}

    if pred is None:
        details["missing_action"] = True
        return 0.0, details

    type_dist = ACTION_TYPE_DISTANCE[(pred.type, gt.type)]
    details["type_distance"] = type_dist

    amount_dist = 0.0
    if pred.type == gt.type and pred.type in ("bet", "raise"):
        if pred.amount is None or gt.amount is None:
            amount_dist = 1.0
            details["missing_amount"] = True
        else:
            amount_dist = amount_distance(pred.amount, gt.amount, legal_range)
            details["amount_distance"] = amount_dist
            details["relative_error"] = abs(pred.amount - gt.amount) / max(abs(pred.amount), abs(gt.amount), 1e-6)

    total_dist = amount_dist if type_dist == 0.0 else type_dist
    details["total_distance"] = total_dist

    score = max(0.0, 1.0 - total_dist)
    details["score"] = score
    return score, details


class ProtocolValidator:
    def __init__(self, game_state: HoldemGameState):
        self.game_state = game_state
        self.expected_streets = self._get_expected_streets()

    def _get_expected_streets(self) -> list[str]:
        stage = str(self.game_state.current_stage).lower()
        stages = ["preflop", "flop", "turn", "river"]
        if stage not in stages:
            return ["preflop"]
        return stages[: stages.index(stage) + 1]

    def validate(self, tool_calls: list[tuple[str, dict[str, Any]]]) -> tuple[float, dict[str, Any]]:
        violations: dict[str, Any] = {}
        score = 1.0

        call_counts = Counter(name for name, _ in tool_calls)

        if call_counts["get_hand_info"] == 0:
            violations["missing_hand_info"] = True
            score -= 0.3
        elif call_counts["get_hand_info"] > 1:
            violations["duplicate_hand_info"] = call_counts["get_hand_info"] - 1
            score -= 0.2 * (call_counts["get_hand_info"] - 1)

        if call_counts["submit_analysis"] == 0:
            violations["missing_analysis"] = True
            score -= 0.2
        elif call_counts["submit_analysis"] > 1:
            violations["duplicate_analysis"] = call_counts["submit_analysis"] - 1
            score -= 0.2 * (call_counts["submit_analysis"] - 1)

        if call_counts["make_action"] == 0:
            violations["missing_action"] = True
            score -= 0.3
        elif call_counts["make_action"] > 1:
            violations["duplicate_action"] = call_counts["make_action"] - 1
            score -= 0.3 * (call_counts["make_action"] - 1)

        streets = ["preflop", "flop", "turn", "river"]
        observed_streets: list[str] = []
        invalid_round_info_streets: list[Any] = []

        for name, args in tool_calls:
            if name != "get_round_info":
                continue
            street = args.get("street")
            if street in streets:
                observed_streets.append(street)
            else:
                invalid_round_info_streets.append(street)

        if invalid_round_info_streets:
            violations["invalid_round_info_street"] = invalid_round_info_streets
            score -= 0.1

        observed_counts = Counter(observed_streets)
        duplicate_streets = {k: v for k, v in observed_counts.items() if v > 1}
        if duplicate_streets:
            violations["duplicate_round_info"] = duplicate_streets
            score -= 0.05 * sum(v - 1 for v in duplicate_streets.values())

        expected_set = set(self.expected_streets)
        observed_set = set(observed_streets)

        missing_streets = [street for street in self.expected_streets if street not in observed_set]
        if missing_streets:
            violations["missing_round_info"] = missing_streets
            score -= 0.1 * len(missing_streets)

        unexpected_streets = [street for street in observed_set if street not in expected_set]
        if unexpected_streets:
            violations["unexpected_round_info"] = sorted(unexpected_streets)
            score -= 0.05 * len(unexpected_streets)

        first_seen: list[str] = []
        for street in observed_streets:
            if street not in first_seen:
                first_seen.append(street)

        if first_seen and first_seen != self.expected_streets[: len(first_seen)]:
            violations["wrong_order_round_info"] = {
                "expected_prefix": self.expected_streets[: len(first_seen)],
                "actual": first_seen,
            }
            score -= 0.1

        tool_names = [name for name, _ in tool_calls]

        if "get_hand_info" in tool_names and tool_names.index("get_hand_info") > 0:
            violations["wrong_order_hand_info"] = True
            score -= 0.2

        if "submit_analysis" in tool_names and "make_action" in tool_names:
            analysis_idx = tool_names.index("submit_analysis")
            action_idx = tool_names.index("make_action")
            if action_idx < analysis_idx:
                violations["wrong_order_analysis"] = True
                score -= 0.2

        if "submit_analysis" in tool_names:
            analysis_idx = tool_names.index("submit_analysis")
            if "get_round_info" in tool_names[analysis_idx + 1 :]:
                violations["round_info_after_analysis"] = True
                score -= 0.1

        if "make_action" in tool_names:
            action_idx = tool_names.index("make_action")
            if action_idx < len(tool_names) - 1:
                violations["calls_after_action"] = tool_names[action_idx + 1 :]
                score -= 0.3

        return max(0.0, score), violations


def efficiency_penalty(num_turns: int) -> float:
    if num_turns <= 7:
        return 0.0
    if num_turns == 8:
        return 0.1
    if num_turns == 9:
        return 0.2
    return 1.0


def combine_rewards(protocol_score: float, action_score: float, efficiency_pen: float, weights: RewardWeights) -> float:
    base_score = weights.protocol * protocol_score + weights.action * action_score
    normalized = base_score * 2.0 - 1.0
    return max(-1.0, min(1.0, normalized - efficiency_pen))


def get_legal_bet_range(game_state: HoldemGameState) -> tuple[float, float]:
    pot_size_dataset = float(openspiel_chips_to_dataset_decimal(game_state.pot_size))
    min_bet = 2.0
    max_bet = max(min_bet, pot_size_dataset * 2.0)
    return min_bet, max_bet


def _validate_sample_like(sample: Any) -> tuple[str, str, str | None]:
    response = getattr(sample, "response", "")
    prompt = getattr(sample, "prompt", "")
    label = getattr(sample, "label", None)

    if not isinstance(response, str):
        raise TypeError("sample.response must be str")
    if not isinstance(prompt, str):
        raise TypeError("sample.prompt must be str")
    if label is not None and not isinstance(label, str):
        raise TypeError("sample.label must be str or None")

    return response, prompt, label


async def reward_func(args, sample: Any, **kwargs) -> dict[str, Any]:
    del args, kwargs

    try:
        response, prompt, label = _validate_sample_like(sample)

        tool_trace = extract_tool_calls(response)
        pred_action = extract_action_from_trace(tool_trace)

        try:
            gt_action = parse_action_string(label or "")
        except ValueError:
            gt_action = PokerAction("fold", None)

        try:
            game_state = HoldemGameState.from_dataset(
                {"instruction": prompt, "output": label or ""},
                player_id=0,
            )
            legal_range = get_legal_bet_range(game_state)
            protocol_score, protocol_details = ProtocolValidator(game_state).validate(tool_trace)
        except Exception as exc:
            legal_range = (2.0, 100.0)
            protocol_score, protocol_details = 1.0, {"warning": f"protocol_check_skipped: {exc}"}

        action_score, action_details = action_quality_score(pred_action, gt_action, legal_range)
        efficiency_pen = efficiency_penalty(len(tool_trace))

        final_score = combine_rewards(
            protocol_score=protocol_score,
            action_score=action_score,
            efficiency_pen=efficiency_pen,
            weights=RewardWeights(protocol=0.3, action=0.7),
        )

        return {
            "score": float(final_score),
            "pred": str(pred_action) if pred_action else "no_action",
            "details": {
                "protocol": {"score": protocol_score, **protocol_details},
                "action": {"score": action_score, **action_details},
                "efficiency": {"penalty": efficiency_pen, "turns": len(tool_trace)},
                "label": label,
            },
        }
    except Exception as exc:
        return {
            "score": -1.0,
            "pred": "error",
            "details": {"error": str(exc)},
        }


__all__ = ["reward_func"]

