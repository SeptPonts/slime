"""将 OpenSpiel State 转为 PokerBench 格式的自然语言提示"""

from __future__ import annotations

from functools import partial
from typing import Callable

from .chips import (
    format_postflop_amount,
    format_preflop_amount,
    openspiel_chips_to_dataset_str,
)
from .openspiel_betting import RoundEvent, first_player_pos, parse_round_events


def _round_events_to_actions(
    *,
    events: list[RoundEvent],
    round_start_contribs: list[int],
    round_index: int,
    num_players: int,
) -> list[tuple[str, str, int | None]]:
    """将 round events 转换为 (position, action, amount) 元组列表"""
    out: list[tuple[str, str, int | None]] = []
    saw_raise = False
    for ev in events:
        position = get_position_name(ev.player_id, num_players)
        if ev.token == "f":
            out.append((position, "fold", None))
            continue
        if ev.token == "c":
            action = "check" if ev.contrib_after == ev.contrib_before else "call"
            out.append((position, action, None))
            continue
        if ev.token == "r":
            if ev.amount_total is None:
                continue
            if round_index == 0:
                action = "raise"
                amount_units = ev.amount_total
            else:
                action = "raise" if saw_raise else "bet"
                amount_units = ev.amount_total - round_start_contribs[ev.player_id]
            saw_raise = True
            out.append((position, action, amount_units))
    return out


def _extract_preflop_board(board_cards: str) -> str | None:
    """Preflop 无公共牌"""
    return None


def _extract_flop_board(board_cards: str) -> str | None:
    """提取 flop 的3张公共牌"""
    if not board_cards or len(board_cards) < 6:
        return None
    return format_board_cards(board_cards[:6])


def _extract_turn_board(board_cards: str) -> str | None:
    """提取 turn 牌"""
    if not board_cards or len(board_cards) < 8:
        return None
    return format_single_card(board_cards[6:8])


def _extract_river_board(board_cards: str) -> str | None:
    """提取 river 牌"""
    if not board_cards or len(board_cards) < 10:
        return None
    return format_single_card(board_cards[8:10])


def _format_preflop_message(board: str | None, actions: str) -> str:
    """格式化 preflop 消息"""
    return (
        f"\nBefore the flop, {actions}. "
        "Assume that all other players that is not mentioned folded."
    )


def _format_postflop_message(board: str | None, actions: str, round_name: str) -> str:
    """格式化 postflop 消息"""
    if not board:
        return ""
    if actions:
        return f"\nThe {round_name} comes {board}, then {actions}."
    else:
        return f"\nThe {round_name} comes {board}."


def _process_round(
    *,
    round_index: int,
    rounds: list[str],
    board_cards: str,
    num_players: int,
    folded_player_ids: set[int],
    contributions: list[int],
    extract_board: Callable[[str], str | None],
    format_message: Callable[[str | None, str], str],
    stage_name: str,
) -> tuple[str | None, set[int], list[int]]:
    """
    处理单个 betting round。

    Returns:
        (formatted_message, updated_folded, updated_contributions)
    """
    if len(rounds) <= round_index:
        return None, folded_player_ids, contributions

    if round_index == 0 and not rounds[0]:
        return None, folded_player_ids, contributions

    # 提取 board cards
    board = extract_board(board_cards)
    if round_index > 0 and not board:  # Postflop 必须有公共牌
        return None, folded_player_ids, contributions

    # 解析 betting history
    actions_text = ""
    if rounds[round_index]:
        round_start = list(contributions)
        events, folded_player_ids, contributions = parse_round_events(
            rounds[round_index],
            first_player=first_player_pos(num_players, round_index),
            num_players=num_players,
            folded_player_ids=folded_player_ids,
            contributions=contributions,
        )
        actions = _round_events_to_actions(
            events=events,
            round_start_contribs=round_start,
            round_index=round_index,
            num_players=num_players,
        )
        if actions:
            actions_text = format_actions_list(actions, stage=stage_name)

    if actions_text:
        return format_message(board, actions_text), folded_player_ids, contributions
    if round_index == 0:
        return None, folded_player_ids, contributions

    return format_message(board, ""), folded_player_ids, contributions


def format_state_for_llm(state, player_id):
    """
    将 OpenSpiel State 对象转换为 PokerBench 格式的自然语言提示

    Args:
        state: OpenSpiel State 对象
        player_id: 当前玩家 ID (0-5)

    Returns:
        str: PokerBench 格式的自然语言描述
    """
    struct = state.to_struct()

    # 提取基础信息
    num_players = len(struct.player_hands)
    small_blind_openspiel = min([b for b in struct.blinds if b > 0])
    big_blind_openspiel = max(struct.blinds)
    starting_stack_openspiel = struct.starting_stacks[player_id]

    small_blind = openspiel_chips_to_dataset_str(
        small_blind_openspiel, force_one_decimal=False
    )
    big_blind = openspiel_chips_to_dataset_str(
        big_blind_openspiel, force_one_decimal=False
    )
    starting_stack = openspiel_chips_to_dataset_str(
        starting_stack_openspiel, force_one_decimal=False
    )

    # 提取手牌
    hole_cards = struct.player_hands[player_id]
    hand_str = format_cards(hole_cards)

    # 提取位置
    position = get_position_name(player_id, num_players)

    # 提取游戏历史（关键）
    history_text = extract_game_history(state, struct, player_id)

    # 构建 PokerBench 格式 prompt
    prompt = (
        f"You are a specialist in playing {num_players}-handed "
        "No Limit Texas Holdem. The following will be a game scenario "
        "and you need to make the optimal decision.\n\n"
        "Here is a game summary:\n\n"
        f"The small blind is {small_blind} chips and the big blind is "
        f"{big_blind} chips. Everyone started with {starting_stack} "
        "chips.\n"
        f"The player positions involved in this game are "
        f"{', '.join(get_all_positions(num_players))}.\n"
        f"In this hand, your position is {position}, and your holding is "
        f"{hand_str}.{history_text}\n\n"
        "Now it is your turn to make a move.\n"
        f"To remind you, the current pot size is "
        f"{openspiel_chips_to_dataset_str(struct.pot_size, force_one_decimal=True)} "
        f"chips, and your holding is {hand_str}.\n\n"
        "Decide on an action based on the strength of your hand on this "
        "board, your position, and actions before you. "
        "Do not explain your answer.\n"
        "Your optimal action is:"
    )

    return prompt


def extract_game_history(state, struct, player_id):
    """
    从 OpenSpiel State 提取完整游戏历史，格式化为 PokerBench 风格

    Args:
        state: OpenSpiel State 对象
        struct: UniversalPokerStateStruct
        player_id: 当前玩家 ID

    Returns:
        str: 游戏历史的自然语言描述
    """
    betting_history = struct.betting_history
    board_cards = struct.board_cards
    num_players = len(struct.player_hands)

    if not betting_history:
        return ""

    rounds = betting_history.split("/")
    folded_player_ids: set[int] = set()
    contributions = list(struct.blinds)
    history_parts = []

    # 定义 round 配置
    round_configs = [
        (0, "preflop", _extract_preflop_board, _format_preflop_message),
        (
            1,
            "flop",
            _extract_flop_board,
            partial(_format_postflop_message, round_name="flop"),
        ),
        (
            2,
            "turn",
            _extract_turn_board,
            partial(_format_postflop_message, round_name="turn"),
        ),
        (
            3,
            "river",
            _extract_river_board,
            partial(_format_postflop_message, round_name="river"),
        ),
    ]

    # 处理每个 round
    for round_index, stage_name, extract_board, format_msg in round_configs:
        message, folded_player_ids, contributions = _process_round(
            round_index=round_index,
            rounds=rounds,
            board_cards=board_cards,
            num_players=num_players,
            folded_player_ids=folded_player_ids,
            contributions=contributions,
            extract_board=extract_board,
            format_message=format_msg,
            stage_name=stage_name,
        )
        if message:
            history_parts.append(message)

    return "".join(history_parts) if history_parts else ""


def format_actions_list(
    actions: list[tuple[str, str, int | None]], *, stage: str
) -> str:
    """
    格式化动作列表为自然语言

    Args:
        actions: [(position, action, amount), ...]

    Returns:
        str: 例如 "UTG fold, HJ raise 10 chips, and CO call"
    """
    if not actions:
        return ""

    formatted = []
    for position, action, amount in actions:
        if action in ["fold", "check", "call"]:
            formatted.append(f"{position} {action}")
        elif action in ["bet", "raise"]:
            if amount is None:  # pragma: no cover - defensive
                continue
            amount_str = (
                format_preflop_amount(amount)
                if stage == "preflop"
                else format_postflop_amount(amount)
            )
            formatted.append(f"{position} {action} {amount_str} chips")

    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]}, and {formatted[1]}"
    else:
        return f"{', '.join(formatted[:-1])}, and {formatted[-1]}"


def format_board_cards(cards_str):
    """
    格式化公共牌（Flop 3张）

    Args:
        cards_str: 例如 "9h7c4h" (6个字符)

    Returns:
        str: 例如 "Nine Of Heart, Seven Of Club, and Four Of Heart"
    """
    if not cards_str or len(cards_str) < 6:
        return ""

    card1 = format_single_card(cards_str[0:2])
    card2 = format_single_card(cards_str[2:4])
    card3 = format_single_card(cards_str[4:6])

    return f"{card1}, {card2}, and {card3}"


def format_single_card(card_str):
    """
    格式化单张牌

    Args:
        card_str: 例如 "9h"

    Returns:
        str: 例如 "Nine Of Heart"
    """
    if not card_str or len(card_str) < 2:
        return ""

    return card_to_text_title_case(card_str)


def format_cards(cards_str):
    """
    将 OpenSpiel 牌格式转为可读文本（手牌格式）

    Args:
        cards_str: e.g., "AsKd"

    Returns:
        e.g., "[Ace of Diamond and Six of Spade]"
    """
    if not cards_str:
        return "[]"

    # 每两个字符是一张牌
    cards = [cards_str[i : i + 2] for i in range(0, len(cards_str), 2)]
    card_texts = [card_to_text(c) for c in cards]

    if len(card_texts) == 1:
        return f"[{card_texts[0]}]"
    elif len(card_texts) == 2:
        return f"[{card_texts[0]} and {card_texts[1]}]"
    else:
        return f"[{', '.join(card_texts[:-1])} and {card_texts[-1]}]"


def card_to_text(card):
    """
    将单张牌转为文本（小写格式，用于手牌）

    Args:
        card: e.g., "As", "Kd", "Tc"

    Returns:
        e.g., "Ace of Spade", "King of Diamond"
    """
    ranks = {
        "2": "Two",
        "3": "Three",
        "4": "Four",
        "5": "Five",
        "6": "Six",
        "7": "Seven",
        "8": "Eight",
        "9": "Nine",
        "T": "Ten",
        "J": "Jack",
        "Q": "Queen",
        "K": "King",
        "A": "Ace",
    }
    suits = {"s": "Spade", "h": "Heart", "d": "Diamond", "c": "Club"}

    rank = ranks.get(card[0], card[0])
    suit = suits.get(card[1].lower(), card[1])

    return f"{rank} of {suit}"


def card_to_text_title_case(card):
    """
    将单张牌转为文本（Title Case 格式，用于公共牌）

    Args:
        card: e.g., "As", "Kd", "Tc"

    Returns:
        e.g., "Ace Of Spade", "King Of Diamond"
    """
    ranks = {
        "2": "Two",
        "3": "Three",
        "4": "Four",
        "5": "Five",
        "6": "Six",
        "7": "Seven",
        "8": "Eight",
        "9": "Nine",
        "T": "Ten",
        "J": "Jack",
        "Q": "Queen",
        "K": "King",
        "A": "Ace",
    }
    suits = {"s": "Spade", "h": "Heart", "d": "Diamond", "c": "Club"}

    rank = ranks.get(card[0], card[0])
    suit = suits.get(card[1].lower(), card[1])

    return f"{rank} Of {suit}"


def get_position_name(player_id, num_players):
    """
    获取玩家位置名称（使用数据集标准）

    Dataset standard position mapping for 6-max:
    - Position 0: SB (Small Blind)
    - Position 1: BB (Big Blind)
    - Position 2: UTG (Under the Gun)
    - Position 3: HJ (Hijack)
    - Position 4: CO (Cutoff)
    - Position 5: BTN (Button)

    Args:
        player_id: 玩家 ID (0-5)
        num_players: 玩家总数

    Returns:
        str: 位置名称（SB, BB, UTG, HJ, CO, BTN）
    """
    if num_players == 6:
        positions = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
        return positions[player_id]
    elif num_players == 2:
        return "SB" if player_id == 0 else "BB"
    else:
        # 其他玩家数的通用逻辑
        return f"P{player_id}"


def get_all_positions(num_players):
    """
    获取所有位置名称列表

    Args:
        num_players: 玩家总数

    Returns:
        list[str]: 位置名称列表
    """
    if num_players == 6:
        return ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
    elif num_players == 2:
        return ["SB", "BB"]
    else:
        return [f"P{i}" for i in range(num_players)]
