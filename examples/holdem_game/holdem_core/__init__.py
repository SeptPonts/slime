"""Shared components for poker agent backends.

This package provides backend-agnostic tools and prompts that can be
used by OpenAI API, Slime, VERL, and other RL training frameworks.
"""

from .chips import (
    DATASET_BIG_BLIND,
    DATASET_SMALL_BLIND,
    DATASET_STARTING_STACK,
    dataset_chips_to_openspiel,
    openspiel_chips_to_dataset_decimal,
)
from .game_state import HoldemGameState
from .openspiel_betting import current_round_index, round_start_contributions
from .pokerbench_prompt_parser import (
    PokerBenchParseError,
    decision_street,
    parse_hand_info,
)
from .prompts import AGENTIC_SYSTEM_PROMPT, DIRECT_SYSTEM_PROMPT
from .tool_state import build_hand_info, build_round_info
from .tools import POKER_TOOLS, execute_tool, execute_tool_sync

__all__ = [
    "DATASET_BIG_BLIND",
    "DATASET_SMALL_BLIND",
    "DATASET_STARTING_STACK",
    "dataset_chips_to_openspiel",
    "openspiel_chips_to_dataset_decimal",
    "HoldemGameState",
    "current_round_index",
    "round_start_contributions",
    "PokerBenchParseError",
    "decision_street",
    "parse_hand_info",
    "AGENTIC_SYSTEM_PROMPT",
    "DIRECT_SYSTEM_PROMPT",
    "build_hand_info",
    "build_round_info",
    "POKER_TOOLS",
    "execute_tool",
    "execute_tool_sync",
]
