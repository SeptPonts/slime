"""PokerBench prompt parsing utilities.

This module parses the PokerBench dataset `instruction` text into structured
information suitable for tool-based agents.

Design constraints:
- Deterministic parsing (no LLM involvement).
- No future information leakage: only parse content before "Now it is your turn".
- Keep output stable for SFT/RL trajectory generation.

Expected instruction format:
    The small blind is {sb} chips and the big blind is {bb} chips.
    Everyone started with {stack} chips.
    The player positions involved in this game are {pos1}, {pos2}, ...
    In this hand, your position is {hero_pos}, your holding is [{cards}].

    Before the flop, {actions}
    [The flop comes {board}, then {actions}]
    [The turn comes {board}, then {actions}]
    [The river comes {board}, then {actions}]

    Now it is your turn. Current pot size is {pot} chips.

Breaking changes to PokerBench format will cause PokerBenchParseError.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

Street = Literal["preflop", "flop", "turn", "river"]
ActionType = Literal["fold", "call", "check", "bet", "raise"]

_DECISION_ANCHOR = "Now it is your turn"

_BLINDS_RE = re.compile(
    r"the small blind is\s+(?P<sb>[\d.]+)\s+chips?\s+and the big blind is\s+"
    r"(?P<bb>[\d.]+)\s+chips?\.\s+everyone started with\s+"
    r"(?P<stack>[\d.]+)\s+chips?\.",
    re.IGNORECASE,
)
_POSITIONS_RE = re.compile(
    r"the player positions involved in this game are\s+(?P<positions>[^.]+)\.",
    re.IGNORECASE,
)
_HERO_POS_RE = re.compile(
    r"in this hand,\s+your position is\s+(?P<pos>\w+)\s*,",
    re.IGNORECASE,
)
_HERO_HOLDING_RE = re.compile(
    r"your holding is\s+\[(?P<holding>[^\]]+)\]",
    re.IGNORECASE,
)
_POT_RE = re.compile(
    r"current pot size is\s+(?P<pot>[\d.]+)\s+chips?",
    re.IGNORECASE,
)

_STREET_HEADERS: dict[Street, str] = {
    "preflop": "Before the flop,",
    "flop": "The flop comes",
    "turn": "The turn comes",
    "river": "The river comes",
}


@dataclass(frozen=True, slots=True)
class ParsedHandInfo:
    small_blind: str
    big_blind: str
    starting_stack: str
    positions: list[str]
    hero_position: str
    hero_holding: str


@dataclass(frozen=True, slots=True)
class ParsedAction:
    actor: str
    action: ActionType
    amount_chips: str | None


@dataclass(frozen=True, slots=True)
class ParsedRoundInfo:
    street: Street
    board: str | None
    actions: list[ParsedAction]
    pot_size: str | None


class PokerBenchParseError(ValueError):
    """Raised when a PokerBench instruction cannot be parsed deterministically."""


def decision_street(instruction: str) -> Street:
    """Determine the decision street from a PokerBench instruction."""
    cut = _text_before_decision(instruction).lower()
    if "the river comes" in cut:
        return "river"
    if "the turn comes" in cut:
        return "turn"
    if "the flop comes" in cut:
        return "flop"
    return "preflop"


def parse_hand_info(instruction: str) -> ParsedHandInfo:
    """Parse static per-hand information from a PokerBench instruction."""
    blinds = _BLINDS_RE.search(instruction)
    if blinds is None:
        raise PokerBenchParseError("Failed to parse blinds/stack line")

    positions_match = _POSITIONS_RE.search(instruction)
    if positions_match is None:
        raise PokerBenchParseError("Failed to parse positions list")

    hero_pos_match = _HERO_POS_RE.search(instruction)
    if hero_pos_match is None:
        raise PokerBenchParseError("Failed to parse hero position")

    holding_match = _HERO_HOLDING_RE.search(instruction)
    if holding_match is None:
        raise PokerBenchParseError("Failed to parse hero holding")

    positions_raw = positions_match.group("positions")
    positions = [p.strip().strip(".") for p in positions_raw.split(",") if p.strip()]

    return ParsedHandInfo(
        small_blind=_normalize_decimal_str(blinds.group("sb")),
        big_blind=_normalize_decimal_str(blinds.group("bb")),
        starting_stack=_normalize_decimal_str(blinds.group("stack")),
        positions=positions,
        hero_position=hero_pos_match.group("pos").upper(),
        hero_holding=holding_match.group("holding").strip(),
    )


def parse_round_info(instruction: str, *, street: Street) -> ParsedRoundInfo:
    """Parse one street of history (board + actions) from a PokerBench instruction."""
    history = _text_before_decision(instruction)
    board = _parse_board(history, street=street)
    actions = _parse_actions(history, street=street)
    pot = _parse_pot_size(instruction) if street == _decision_street(history) else None
    return ParsedRoundInfo(street=street, board=board, actions=actions, pot_size=pot)


def _decision_street(history: str) -> Street:
    """Determine decision street from history text before decision anchor."""
    cut = history.lower()
    if "the river comes" in cut:
        return "river"
    if "the turn comes" in cut:
        return "turn"
    if "the flop comes" in cut:
        return "flop"
    return "preflop"


def _text_before_decision(instruction: str) -> str:
    idx = instruction.find(_DECISION_ANCHOR)
    if idx == -1:
        raise PokerBenchParseError("Missing decision anchor")
    return instruction[:idx]


def _parse_pot_size(instruction: str) -> str | None:
    match = _POT_RE.search(instruction)
    if match is None:
        return None
    return _normalize_decimal_str(match.group("pot"))


def _parse_board(history: str, *, street: Street) -> str | None:
    if street == "preflop":
        return None

    header = _STREET_HEADERS[street]
    idx = history.lower().find(header.lower())
    if idx == -1:
        return None

    after = history[idx + len(header) :].strip()
    if ", then" in after:
        board_text = after.split(", then", 1)[0].strip()
        return board_text.rstrip(".")
    if "." in after:
        board_text = after.split(".", 1)[0].strip()
        return board_text.rstrip(".")
    return after.rstrip(".")


def _street_section(history: str, *, street: Street) -> str:
    street_headers = _STREET_HEADERS
    start_header = street_headers[street]
    start_idx = history.lower().find(start_header.lower())
    if start_idx == -1:
        return ""

    # Section ends at next street header (if present) or end of history.
    next_headers: list[str] = []
    if street == "preflop":
        next_headers = [street_headers["flop"]]
    elif street == "flop":
        next_headers = [street_headers["turn"]]
    elif street == "turn":
        next_headers = [street_headers["river"]]

    end_idx = len(history)
    lower = history.lower()
    for h in next_headers:
        j = lower.find(h.lower(), start_idx + 1)
        if j != -1:
            end_idx = min(end_idx, j)

    return history[start_idx:end_idx]


def _parse_actions(history: str, *, street: Street) -> list[ParsedAction]:
    section = _street_section(history, street=street)
    if not section:
        return []

    if street == "preflop":
        # "Before the flop, <actions>."
        if "," in section:
            section = section.split(",", 1)[1]
    else:
        # "The flop comes <board>, then <actions>."
        if ", then" in section:
            section = section.split(", then", 1)[1]
        else:
            return []

    # Parse "{POS} {action} [amount] chips" in order.
    valid_positions = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
    pos_re = "|".join(valid_positions)
    action_re = r"(fold|call|check|bet|raise)"
    if street == "preflop":
        amount_re = r"(?:\s+(?P<amt>[\d.]+)(?:\s+chips?)?)?"
    else:
        amount_re = r"(?:\s+(?P<amt>[\d.]+)\s+chips?)?"
    token_re = re.compile(
        rf"\b(?P<pos>{pos_re})\s+(?P<act>{action_re}){amount_re}\b",
        re.IGNORECASE,
    )

    out: list[ParsedAction] = []
    for m in token_re.finditer(section):
        action = m.group("act").lower()
        if action not in {"fold", "call", "check", "bet", "raise"}:
            continue
        amount = m.groupdict().get("amt")
        out.append(
            ParsedAction(
                actor=m.group("pos").upper(),
                action=action,  # type: ignore[assignment]
                amount_chips=_normalize_action_amount(amount, street=street)
                if amount
                else None,
            )
        )

    return out


def _normalize_decimal_str(value: str) -> str:
    dec = Decimal(value)
    if dec == dec.to_integral_value():
        return f"{int(dec)}.0"
    normalized = format(dec.normalize(), "f")
    return normalized


def _normalize_action_amount(value: str, *, street: Street) -> str:
    dec = Decimal(value)
    if street == "preflop":
        if dec == dec.to_integral_value():
            return f"{int(dec)}.0"
        return format(dec.normalize(), "f")
    if dec == dec.to_integral_value():
        return str(int(dec))
    return format(dec.normalize(), "f")
