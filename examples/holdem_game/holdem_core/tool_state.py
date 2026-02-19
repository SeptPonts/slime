"""Tool-facing structured state for the LLM agent.

This module provides a single, deterministic way to present game state to an LLM
via tools, for both OpenSpiel states and PokerBench dataset instructions.

The agent gets:
- Per-hand static info once (blinds, positions, hero seat/holding).
- Per-street info in a controlled loop (board/actions), stopping at the current
  decision street to avoid leaking future information.

Usage example:
    # From PokerBench dataset
    game_state = HoldemGameState.from_dataset(dataset_entry)
    hand_info = build_hand_info(game_state)

    # Street iteration protocol
    street = "preflop"
    while True:
        info = build_round_info(game_state, street=street)
        # ... process info ...
        if not info.has_next_round:
            break
        street = info.next_street
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .chips import (
    format_postflop_amount,
    format_preflop_amount,
    openspiel_chips_to_dataset_str,
)
from .openspiel_betting import (
    RoundEvent,
    first_player_pos,
    parse_round_events,
)
from .pokerbench_prompt_parser import (
    ParsedAction,
    PokerBenchParseError,
    parse_hand_info,
    parse_round_info,
)
from .pokerbench_prompt_parser import (
    decision_street as dataset_decision_street,
)
from .state_formatter import (
    format_board_cards,
    format_single_card,
    get_all_positions,
    get_position_name,
)

Street = Literal["preflop", "flop", "turn", "river"]
ActionType = Literal["fold", "call", "check", "bet", "raise"]

_STREETS: list[Street] = ["preflop", "flop", "turn", "river"]


class ToolStateError(ValueError):
    """Base exception for tool state building errors."""


@dataclass(frozen=True, slots=True)
class ToolAction:
    actor: str
    action: ActionType
    amount_chips: str | None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"actor": self.actor, "action": self.action}
        if self.amount_chips is not None:
            out["amount_chips"] = self.amount_chips
        return out


@dataclass(frozen=True, slots=True)
class ToolHandInfo:
    small_blind: str
    big_blind: str
    starting_stack: str
    num_players: int
    positions: list[str]
    hero_position: str
    hero_holding: str
    unit: str = "chips"

    def to_dict(self) -> dict[str, Any]:
        return {
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "starting_stack": self.starting_stack,
            "num_players": self.num_players,
            "positions": self.positions,
            "hero_position": self.hero_position,
            "hero_holding": self.hero_holding,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class ToolRoundInfo:
    street: Street
    board: str | None
    actions: list[ToolAction]
    pot_size: str | None
    has_next_round: bool
    next_street: Street | None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "street": self.street,
            "actions": [a.to_dict() for a in self.actions],
            "has_next_round": self.has_next_round,
            "next_street": self.next_street,
        }
        if self.board is not None:
            out["board"] = self.board
        if self.pot_size is not None:
            out["pot_size"] = self.pot_size
        return out


def build_hand_info(game_state) -> ToolHandInfo:
    """Build per-hand static info for tools from HoldemGameState."""
    if game_state.openspiel_state is not None:
        return _hand_info_from_openspiel(game_state)
    return _hand_info_from_instruction(
        game_state.full_prompt or game_state.action_history
    )


def build_round_info(game_state, *, street: Street) -> ToolRoundInfo:
    """Build per-street info for tools from HoldemGameState."""
    decision = _decision_street(game_state)
    if _street_index(street) > _street_index(decision):
        raise ValueError(f"Street '{street}' is beyond decision street '{decision}'")

    if game_state.openspiel_state is not None:
        return _round_info_from_openspiel(game_state, street=street, decision=decision)

    instruction = game_state.full_prompt or game_state.action_history
    return _round_info_from_instruction(instruction, street=street, decision=decision)


def _street_index(street: Street) -> int:
    return _STREETS.index(street)


def _decision_street(game_state) -> Street:
    if game_state.openspiel_state is not None:
        struct = game_state.openspiel_state.to_struct()
        board_cards = struct.board_cards or ""
        card_count = len(board_cards) // 2
        if card_count >= 5:
            return "river"
        if card_count == 4:
            return "turn"
        if card_count == 3:
            return "flop"
        return "preflop"
    instruction = game_state.full_prompt or game_state.action_history
    return dataset_decision_street(instruction)


def _hand_info_from_instruction(instruction: str) -> ToolHandInfo:
    try:
        parsed = parse_hand_info(instruction)
    except PokerBenchParseError as exc:
        raise ToolStateError(str(exc)) from exc

    num_players = len(parsed.positions)
    return ToolHandInfo(
        small_blind=parsed.small_blind,
        big_blind=parsed.big_blind,
        starting_stack=parsed.starting_stack,
        num_players=num_players,
        positions=parsed.positions,
        hero_position=parsed.hero_position,
        hero_holding=parsed.hero_holding,
    )


def _round_info_from_instruction(
    instruction: str, *, street: Street, decision: Street
) -> ToolRoundInfo:
    try:
        parsed = parse_round_info(instruction, street=street)
    except PokerBenchParseError as exc:
        raise ToolStateError(str(exc)) from exc

    actions = [_tool_action_from_parsed(a) for a in parsed.actions]
    has_next, next_street = _next_street(street, decision=decision)
    return ToolRoundInfo(
        street=street,
        board=parsed.board,
        actions=actions,
        pot_size=parsed.pot_size,
        has_next_round=has_next,
        next_street=next_street,
    )


def _tool_action_from_parsed(action: ParsedAction) -> ToolAction:
    return ToolAction(
        actor=action.actor,
        action=action.action,
        amount_chips=action.amount_chips,
    )


def _hand_info_from_openspiel(game_state) -> ToolHandInfo:
    state = game_state.openspiel_state
    struct = state.to_struct()
    num_players = len(struct.player_hands)

    # Validate blinds
    positive_blinds = [b for b in struct.blinds if b > 0]
    if not positive_blinds:
        raise ToolStateError(f"No positive blinds in OpenSpiel state: {struct.blinds}")

    small_blind_openspiel = min(positive_blinds)
    big_blind_openspiel = max(struct.blinds)

    # Validate player_id
    if not (0 <= game_state.player_id < len(struct.starting_stacks)):
        raise ToolStateError(
            f"player_id {game_state.player_id} out of range "
            f"[0, {len(struct.starting_stacks)})"
        )

    starting_stack_openspiel = struct.starting_stacks[game_state.player_id]

    return ToolHandInfo(
        small_blind=openspiel_chips_to_dataset_str(
            small_blind_openspiel, force_one_decimal=False
        ),
        big_blind=openspiel_chips_to_dataset_str(
            big_blind_openspiel, force_one_decimal=False
        ),
        starting_stack=openspiel_chips_to_dataset_str(
            starting_stack_openspiel, force_one_decimal=False
        ),
        num_players=num_players,
        positions=get_all_positions(num_players),
        hero_position=get_position_name(game_state.player_id, num_players),
        hero_holding=game_state.get_my_hand(),
    )


def _round_info_from_openspiel(
    game_state, *, street: Street, decision: Street
) -> ToolRoundInfo:
    state = game_state.openspiel_state
    struct = state.to_struct()
    num_players = len(struct.player_hands)

    round_index = _street_index(street)
    rounds = struct.betting_history.split("/") if struct.betting_history else [""]
    folded, contributions = _round_start_state(
        rounds=rounds,
        num_players=num_players,
        blinds=list(struct.blinds),
        round_index=round_index,
    )

    actions: list[ToolAction] = []
    if len(rounds) > round_index and rounds[round_index]:
        start_contrib = list(contributions)
        events, folded, contributions = parse_round_events(
            rounds[round_index],
            first_player=first_player_pos(num_players, round_index),
            num_players=num_players,
            folded_player_ids=folded,
            contributions=contributions,
        )
        actions = _events_to_tool_actions(
            events=events,
            round_start_contribs=start_contrib,
            round_index=round_index,
            num_players=num_players,
        )

    board = _board_text_from_struct(struct.board_cards or "", street=street)
    pot_size = (
        openspiel_chips_to_dataset_str(struct.pot_size, force_one_decimal=True)
        if street == decision
        else None
    )
    has_next, next_street = _next_street(street, decision=decision)
    return ToolRoundInfo(
        street=street,
        board=board,
        actions=actions,
        pot_size=pot_size,
        has_next_round=has_next,
        next_street=next_street,
    )


def _round_start_state(
    *,
    rounds: list[str],
    num_players: int,
    blinds: list[int],
    round_index: int,
) -> tuple[set[int], list[int]]:
    folded: set[int] = set()
    contributions = list(blinds)
    for r in range(min(round_index, len(rounds))):
        start_player = first_player_pos(num_players, r)
        _, folded, contributions = parse_round_events(
            rounds[r],
            first_player=start_player,
            num_players=num_players,
            folded_player_ids=folded,
            contributions=contributions,
        )
    return folded, contributions


def _events_to_tool_actions(
    *,
    events: list[RoundEvent],
    round_start_contribs: list[int],
    round_index: int,
    num_players: int,
) -> list[ToolAction]:
    out: list[ToolAction] = []
    saw_raise = False
    for ev in events:
        position = get_position_name(ev.player_id, num_players)
        if ev.token == "f":
            out.append(ToolAction(position, "fold", None))
            continue
        if ev.token == "c":
            action: ActionType = (
                "check" if ev.contrib_after == ev.contrib_before else "call"
            )
            out.append(ToolAction(position, action, None))
            continue
        if ev.token != "r" or ev.amount_total is None:
            continue

        if round_index == 0:
            action_type: ActionType = "raise"
            amount_units = ev.amount_total
            amount_str = format_preflop_amount(amount_units)
        else:
            action_type = "raise" if saw_raise else "bet"
            amount_units = ev.amount_total - round_start_contribs[ev.player_id]
            amount_str = format_postflop_amount(amount_units)
        saw_raise = True
        out.append(ToolAction(position, action_type, amount_str))
    return out


def _board_text_from_struct(board_cards: str, *, street: Street) -> str | None:
    if street == "preflop":
        return None
    if street == "flop":
        if len(board_cards) < 6:
            return None
        text = format_board_cards(board_cards[:6])
        return text or None
    if street == "turn":
        if len(board_cards) < 8:
            return None
        text = format_single_card(board_cards[6:8])
        return text or None
    if street == "river":
        if len(board_cards) < 10:
            return None
        text = format_single_card(board_cards[8:10])
        return text or None
    return None


def _next_street(street: Street, *, decision: Street) -> tuple[bool, Street | None]:
    if street == decision:
        return False, None
    i = _street_index(street)
    return True, _STREETS[i + 1]
