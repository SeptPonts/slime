"""Shared poker tools for all backends (OpenAI API, Slime, VERL, etc).

This module defines tool specifications and a small execution shim.

Good taste here is about eliminating backend special-casing:
- Backends own their game-state representation.
- The "core" layer only requires a tiny tool-facing interface.
"""

from typing import Any, Literal, Protocol

Street = Literal["preflop", "flop", "turn", "river"]


class ToolGameState(Protocol):
    """Minimal interface required to execute tool calls."""

    def tool_get_hand_info(self) -> dict[str, Any]:
        ...

    def tool_get_round_info(self, *, street: Street) -> dict[str, Any]:
        ...


# Tool definitions in OpenAI format
POKER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_hand_info",
            "description": (
                "Get per-hand static information (blinds, stacks, positions, "
                "and hero seat/holding). Call this exactly once."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_round_info",
            "description": (
                "Get structured information for one street of play. "
                "Start at street=preflop, then follow next_street until "
                "has_next_round is false. pot_size is only included for the "
                "decision street (current pot at decision time)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "street": {
                        "type": "string",
                        "enum": ["preflop", "flop", "turn", "river"],
                        "description": "Street to fetch",
                    }
                },
                "required": ["street"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_analysis",
            "description": (
                "Submit your poker analysis before making a decision. "
                "This should be called after gathering hand and street "
                "information, but before calling make_action."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "initiative_context": {
                        "type": "string",
                        "description": (
                            "Initiative & Context: Who has initiative, "
                            "board texture changes. Max 3 sentences."
                        ),
                    },
                    "range_assessment": {
                        "type": "string",
                        "description": (
                            "Range Assessment: Opponent's likely range "
                            "(VALUE vs BLUFF). Max 3 sentences."
                        ),
                    },
                    "action_logic": {
                        "type": "string",
                        "description": (
                            "Action Logic: Whether your action is for Value "
                            "or as Bluff, target hands. Max 3 sentences."
                        ),
                    },
                    "sizing_plan": {
                        "type": "string",
                        "description": (
                            "Sizing & Plan: Bet sizing reasoning, "
                            "response to raises. Max 3 sentences."
                        ),
                    },
                    "final_action": {
                        "type": "string",
                        "description": (
                            "Final optimal action summary in one sentence."
                        ),
                    },
                },
                "required": [
                    "initiative_context",
                    "range_assessment",
                    "action_logic",
                    "sizing_plan",
                    "final_action",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_action",
            "description": (
                "Execute a poker action after analyzing game state. "
                "Call this after gathering hand and street information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["fold", "check", "call", "bet", "raise"],
                        "description": "Type of action to take",
                    },
                    "amount": {
                        "type": "number",
                        "description": (
                            "Amount of chips to bet or raise. "
                            "Required for bet/raise, "
                            "ignored for fold/check/call."
                        ),
                    },
                },
                "required": ["action_type"],
            },
        },
    },
]


async def execute_tool(
    tool_name: str, game_state: ToolGameState, arguments: dict[str, Any]
) -> dict | None:
    """Execute a tool call asynchronously."""
    return _execute_tool_impl(tool_name, game_state, arguments)


def execute_tool_sync(
    tool_name: str, game_state: ToolGameState, arguments: dict[str, Any]
) -> dict | None:
    """Execute a tool call synchronously (for OpenAI API backend).

    This is a synchronous version of execute_tool() for backends that
    don't use async/await (like the OpenAI API client).

    Args:
        tool_name: Name of the tool to execute
        game_state: Current game state
        arguments: Tool arguments

    Returns:
        dict: Tool execution result
        None: For make_action (termination signal)
    """
    return _execute_tool_impl(tool_name, game_state, arguments)


def _execute_tool_impl(
    tool_name: str, game_state: ToolGameState, arguments: dict[str, Any]
) -> dict | None:
    if not isinstance(arguments, dict):
        return {"error": "Tool arguments must be a JSON object"}

    if tool_name == "make_action":
        return None

    if tool_name == "get_hand_info":
        try:
            return game_state.tool_get_hand_info()
        except Exception as exc:
            return {"error": str(exc)}

    if tool_name == "get_round_info":
        street = arguments.get("street")
        if street not in ("preflop", "flop", "turn", "river"):
            return {"error": f"Invalid street: {street}"}
        try:
            return game_state.tool_get_round_info(street=street)
        except Exception as exc:
            return {"error": str(exc)}

    if tool_name == "submit_analysis":
        return {
            "status": "analysis_submitted",
            "message": "Analysis received. You can now call make_action.",
        }

    return {"error": f"Unknown tool: {tool_name}"}
