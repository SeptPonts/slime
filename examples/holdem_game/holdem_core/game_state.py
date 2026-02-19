"""Unified game state representation for Texas Hold'em.

This module provides HoldemGameState, a single data structure that supports
both OpenSpiel and PokerBench dataset sources, eliminating code path duplication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .chips import openspiel_chips_to_dataset_str
from .pokerbench_prompt_parser import (
    PokerBenchParseError,
)
from .pokerbench_prompt_parser import (
    decision_street as dataset_decision_street,
)
from .state_formatter import (
    card_to_text,
    card_to_text_title_case,
    extract_game_history,
    get_position_name,
)


@dataclass
class HoldemGameState:
    """
    统一的德州扑克游戏状态表示

    支持两种初始化方式：
    1. HoldemGameState.from_openspiel(state, player_id) 用于 online continual learning
    2. HoldemGameState.from_dataset(dataset_entry, player_id) 用于 rl training
    """

    player_id: int
    hole_cards: str  # e.g., "AsKd"
    community_cards: str  # e.g., "9h7c4h2s" or ""
    pot_size: int  # OpenSpiel 单位（整数）
    position: str  # "SB", "BB", "UTG", etc.
    num_players: int
    action_history: str  # 完整历史（自然语言）
    current_stage: str  # "preflop", "flop", "turn", "river"

    # 用于 direct mode 的完整 prompt（可选）
    full_prompt: str | None = None

    # OpenSpiel state 引用（用于正确转换 action）
    openspiel_state: Any | None = None

    @classmethod
    def from_openspiel(cls, state, player_id: int) -> HoldemGameState:
        """
        从 OpenSpiel State 构造游戏状态

        Args:
            state: OpenSpiel State 对象
            player_id: 玩家 ID (0-5)

        Returns:
            HoldemGameState 实例
        """
        struct = state.to_struct()

        # 提取核心属性
        num_players = len(struct.player_hands)
        hole_cards = struct.player_hands[player_id]
        community_cards = struct.board_cards or ""
        pot_size = struct.pot_size
        position = get_position_name(player_id, num_players)

        # 确定当前阶段
        current_stage = _determine_stage_from_board(community_cards)

        # 提取游戏历史
        action_history = extract_game_history(state, struct, player_id)

        # 为 direct mode 生成完整 prompt（复用现有逻辑）
        from .state_formatter import format_state_for_llm

        full_prompt = format_state_for_llm(state, player_id)

        return cls(
            player_id=player_id,
            hole_cards=hole_cards,
            community_cards=community_cards,
            pot_size=pot_size,
            position=position,
            num_players=num_players,
            action_history=action_history,
            current_stage=current_stage,
            full_prompt=full_prompt,
            openspiel_state=state,
        )

    @classmethod
    def from_dataset(cls, dataset_entry: dict, player_id: int = 0) -> HoldemGameState:
        """
        从 PokerBench 数据集条目构造游戏状态

        Args:
            dataset_entry: PokerBench 数据集的一条记录 (dict)
                包含 instruction, output 等字段
            player_id: 玩家 ID（默认 0）

        Returns:
            HoldemGameState 实例

        Note:
            使用简单正则解析 instruction，实用主义方法。
        """
        instruction = dataset_entry["instruction"]

        # 解析手牌
        hole_cards = _extract_hole_cards(instruction)

        # 解析公共牌
        community_cards = _extract_community_cards(instruction)

        # 解析底池大小
        pot_size = _extract_pot_size(instruction)

        # 解析位置
        position = _extract_position(instruction)

        # 解析玩家数量
        num_players = _extract_num_players(instruction)

        # 优先使用与 tool_state 一致的决策街道解析逻辑；
        # 非标准文本（如缺少 decision anchor）时回退到关键词启发式。
        try:
            current_stage = dataset_decision_street(instruction)
        except PokerBenchParseError:
            current_stage = _determine_stage_from_instruction(instruction)

        # 游戏历史 = instruction 本身（简化）
        action_history = instruction

        # full_prompt = instruction（用于 direct mode）
        full_prompt = instruction

        return cls(
            player_id=player_id,
            hole_cards=hole_cards,
            community_cards=community_cards,
            pot_size=pot_size,
            position=position,
            num_players=num_players,
            action_history=action_history,
            current_stage=current_stage,
            full_prompt=full_prompt,
        )

    def get_my_hand(self) -> str:
        """返回手牌的自然语言描述"""
        if not self.hole_cards:
            return "Unknown hand"

        # 每两个字符是一张牌
        cards = [self.hole_cards[i : i + 2] for i in range(0, len(self.hole_cards), 2)]
        card_texts = [card_to_text(c) for c in cards if c]

        if len(card_texts) == 2:
            return f"{card_texts[0]} and {card_texts[1]}"
        else:
            return ", ".join(card_texts)

    def get_community_cards(self) -> str:
        """返回公共牌的描述"""
        if not self.community_cards:
            return "No community cards yet (preflop)"

        # 每两个字符是一张牌
        cards = [
            self.community_cards[i : i + 2]
            for i in range(0, len(self.community_cards), 2)
        ]
        card_texts = [card_to_text_title_case(c) for c in cards if c]

        if len(card_texts) == 0:
            return "No community cards"
        elif len(card_texts) == 1:
            return card_texts[0]
        else:
            return ", ".join(card_texts[:-1]) + f", and {card_texts[-1]}"

    def get_pot_info(self) -> dict:
        """
        返回底池信息

        Returns:
            dict 包含 pot_size 和可读格式
        """
        pot_str = openspiel_chips_to_dataset_str(self.pot_size, force_one_decimal=True)
        return {
            "pot_size": self.pot_size,
            "pot_size_str": f"{pot_str} chips",
        }

    def get_action_history(self) -> str:
        """返回完整行动历史"""
        return self.action_history

    def get_position_info(self) -> dict:
        """返回位置信息"""
        return {
            "position": self.position,
            "num_players": self.num_players,
        }

    def tool_get_hand_info(self) -> dict[str, Any]:
        """Return tool-facing per-hand info as a plain dict.

        This keeps holdem_core backend-agnostic: the game-state object owns the
        concrete representation and just exposes a minimal tool interface.
        """
        from .tool_state import build_hand_info

        return build_hand_info(self).to_dict()

    def tool_get_round_info(self, *, street: str) -> dict[str, Any]:
        """Return tool-facing per-street info as a plain dict."""
        from .tool_state import Street, build_round_info

        if street not in ("preflop", "flop", "turn", "river"):
            raise ValueError(f"Invalid street: {street}")
        street_typed: Street = street
        info = build_round_info(self, street=street_typed)
        return info.to_dict()


def _determine_stage_from_board(board_cards: str) -> str:
    """根据公共牌确定阶段"""
    if not board_cards:
        return "preflop"

    card_count = len(board_cards) // 2

    if card_count >= 5:
        return "river"
    elif card_count == 4:
        return "turn"
    elif card_count == 3:
        return "flop"
    else:
        return "preflop"


def _determine_stage_from_instruction(instruction: str) -> str:
    """从 instruction 文本确定阶段"""
    text_lower = instruction.lower()

    # 查找最后提到的阶段
    if "the river comes" in text_lower:
        return "river"
    elif "the turn comes" in text_lower:
        return "turn"
    elif "the flop comes" in text_lower:
        return "flop"
    else:
        return "preflop"


def _extract_hole_cards(instruction: str) -> str:
    """
    从 instruction 提取手牌

    Example:
        "your holding is [Ace of Spade and King of Diamond]"
        -> "AsKd"
    """
    # 正则匹配 "your holding is [...]"
    match = re.search(r"your holding is \[(.*?)\]", instruction, re.IGNORECASE)
    if not match:
        return ""

    holding_text = match.group(1)

    # 解析卡牌文本 -> 缩写
    cards = _parse_card_text_to_abbrev(holding_text)
    return "".join(cards)


def _extract_community_cards(instruction: str) -> str:
    """
    从 instruction 提取公共牌

    Examples:
        "The flop comes Nine Of Heart, Seven Of Club, and Four Of Heart"
        "The turn comes Two Of Spade"
        "The river comes Jack Of Diamond"
    """
    community = []

    # Flop
    flop_match = re.search(
        r"The flop comes (.*?)(?:\.|,\s+then|$)", instruction, re.IGNORECASE
    )
    if flop_match:
        flop_text = flop_match.group(1)
        community.extend(_parse_card_text_to_abbrev(flop_text))

    # Turn
    turn_match = re.search(
        r"The turn comes (.*?)(?:\.|,\s+then|$)", instruction, re.IGNORECASE
    )
    if turn_match:
        turn_text = turn_match.group(1)
        community.extend(_parse_card_text_to_abbrev(turn_text))

    # River
    river_match = re.search(
        r"The river comes (.*?)(?:\.|,\s+then|Now|$)", instruction, re.IGNORECASE
    )
    if river_match:
        river_text = river_match.group(1)
        community.extend(_parse_card_text_to_abbrev(river_text))

    return "".join(community)


def _extract_pot_size(instruction: str) -> int:
    """
    从 instruction 提取底池大小（返回 OpenSpiel 整数单位）

    Example:
        "the current pot size is 18.0 chips"
        -> 180 (OpenSpiel units)
    """
    match = re.search(r"pot size is ([\d.]+) chips", instruction, re.IGNORECASE)
    if not match:
        return 0

    pot_str = match.group(1)

    # Dataset uses 0.1 = 1 OpenSpiel unit
    # So 18.0 chips = 180 OpenSpiel units
    from decimal import Decimal

    from .chips import dataset_chips_to_openspiel

    pot_decimal = Decimal(pot_str)
    return dataset_chips_to_openspiel(pot_decimal)


def _extract_position(instruction: str) -> str:
    """
    从 instruction 提取位置

    Example:
        "your position is UTG"
    """
    match = re.search(r"your position is (\w+)", instruction, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "UNKNOWN"


def _extract_num_players(instruction: str) -> int:
    """
    从 instruction 提取玩家数量

    Example:
        "6-handed No Limit Texas Holdem"
    """
    match = re.search(r"(\d+)-handed", instruction)
    if match:
        return int(match.group(1))
    return 6  # 默认 6 人


def _parse_card_text_to_abbrev(card_text: str) -> list[str]:
    """
    解析自然语言卡牌文本为缩写

    Examples:
        "Ace of Spade and King of Diamond"
        -> ["As", "Kd"]

        "Nine Of Heart, Seven Of Club, and Four Of Heart"
        -> ["9h", "7c", "4h"]
    """
    # 卡牌映射
    rank_map = {
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "T",
        "jack": "J",
        "queen": "Q",
        "king": "K",
        "ace": "A",
    }
    suit_map = {"spade": "s", "heart": "h", "diamond": "d", "club": "c"}

    cards = []

    # 正则提取所有 "Rank of Suit" 模式
    pattern = (
        r"(Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Jack|Queen|King|Ace)"
        r"\s+[Oo]f\s+(Spade|Heart|Diamond|Club)"
    )
    matches = re.findall(pattern, card_text, re.IGNORECASE)

    for rank_str, suit_str in matches:
        rank = rank_map.get(rank_str.lower(), "")
        suit = suit_map.get(suit_str.lower(), "")
        if rank and suit:
            cards.append(f"{rank}{suit}")

    return cards
