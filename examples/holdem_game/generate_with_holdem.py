"""Compatibility entrypoint for holdem custom rollout and reward.

Keep this module path stable for CLI usage:
- --custom-generate-function-path generate_with_holdem.generate
- --custom-rm-path generate_with_holdem.reward_func
"""

from holdem_rollout import generate
from holdem_reward import reward_func

__all__ = ["generate", "reward_func"]

