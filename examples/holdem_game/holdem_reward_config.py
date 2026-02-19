"""Reward configuration constants for holdem RL."""

from dataclasses import dataclass

ACTION_TYPE_DISTANCE: dict[tuple[str, str], float] = {
    ("fold", "fold"): 0.0,
    ("fold", "check"): 0.9,
    ("fold", "call"): 0.7,
    ("fold", "bet"): 1.0,
    ("fold", "raise"): 1.0,
    ("check", "fold"): 0.9,
    ("check", "check"): 0.0,
    ("check", "call"): 0.3,
    ("check", "bet"): 0.8,
    ("check", "raise"): 0.9,
    ("call", "fold"): 0.7,
    ("call", "check"): 0.3,
    ("call", "call"): 0.0,
    ("call", "bet"): 0.6,
    ("call", "raise"): 0.5,
    ("bet", "fold"): 1.0,
    ("bet", "check"): 0.8,
    ("bet", "call"): 0.6,
    ("bet", "bet"): 0.0,
    ("bet", "raise"): 0.2,
    ("raise", "fold"): 1.0,
    ("raise", "check"): 0.9,
    ("raise", "call"): 0.5,
    ("raise", "bet"): 0.2,
    ("raise", "raise"): 0.0,
}


@dataclass
class RewardWeights:
    protocol: float = 0.3
    action: float = 0.7

