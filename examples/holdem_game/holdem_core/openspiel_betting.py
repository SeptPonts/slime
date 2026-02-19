"""OpenSpiel universal_poker betting history helpers.

OpenSpiel encodes betting as a compact string per round separated by `/`, e.g.
`cccccc/cr20c` where:
- `f` = fold
- `c` = check/call
- `rNN` = bet/raise to total contribution `NN` (integer chips, cumulative)

Crucially, `rNN` is *total player contribution* across the whole hand, not
"amount bet on this street".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoundEvent:
    player_id: int
    token: str  # "f", "c", "r"
    amount_total: int | None  # only for "r"
    contrib_before: int
    contrib_after: int


def split_rounds(betting_history: str) -> list[str]:
    if betting_history == "":
        return [""]
    return betting_history.split("/")


def current_round_index(betting_history: str) -> int:
    return len(split_rounds(betting_history)) - 1


def first_player_pos(num_players: int, round_index: int) -> int:
    """
    Betting order assumption used by this project for dataset alignment.

    For 6-max dataset mapping [SB, BB, UTG, HJ, CO, BTN]:
    - preflop starts at UTG (player 2)
    - postflop starts at SB (player 0)
    """
    if num_players == 6:
        return 2 if round_index == 0 else 0
    if num_players == 2:
        return 0
    return 0


def _next_active_player(
    player_id: int, *, num_players: int, folded_player_ids: set[int]
) -> int:
    while player_id in folded_player_ids:
        player_id = (player_id + 1) % num_players
    return player_id


def parse_round_events(
    round_str: str,
    *,
    first_player: int,
    num_players: int,
    folded_player_ids: set[int],
    contributions: list[int],
) -> tuple[list[RoundEvent], set[int], list[int]]:
    """
    Parse one betting round into player-indexed events, updating stateful totals.

    Args:
        round_str: Round substring (no `/`).
        first_player: Player to act first (0-based).
        num_players: Total players.
        folded_player_ids: Folded players (updated in place).
        contributions: Total contributions per player (updated in place).

    Returns:
        (events, updated_folded, updated_contributions)
    """
    events: list[RoundEvent] = []
    current_player = first_player
    i = 0

    # "Call" always matches the current max contribution among active players.
    current_max = max(
        contributions[p] for p in range(num_players) if p not in folded_player_ids
    )

    while i < len(round_str):
        current_player = _next_active_player(
            current_player, num_players=num_players, folded_player_ids=folded_player_ids
        )
        token = round_str[i]

        if token == "f":
            before = contributions[current_player]
            events.append(RoundEvent(current_player, "f", None, before, before))
            folded_player_ids.add(current_player)
            i += 1
        elif token == "c":
            # check/call: match the current max contribution
            before = contributions[current_player]
            contributions[current_player] = current_max
            after = contributions[current_player]
            events.append(RoundEvent(current_player, "c", None, before, after))
            i += 1
        elif token == "r":
            i += 1
            num_str = ""
            while i < len(round_str) and round_str[i].isdigit():
                num_str += round_str[i]
                i += 1
            amount_total = int(num_str) if num_str else 0
            before = contributions[current_player]
            contributions[current_player] = amount_total
            after = contributions[current_player]
            events.append(RoundEvent(current_player, "r", amount_total, before, after))
            if amount_total > current_max:
                current_max = amount_total
        else:
            i += 1

        current_player = (current_player + 1) % num_players

    return events, folded_player_ids, contributions


def round_start_contributions(
    *,
    blinds: list[int],
    betting_history: str,
    num_players: int,
    round_index: int,
) -> list[int]:
    """
    Compute total contributions at the start of `round_index` (0=preflop).
    """
    rounds = split_rounds(betting_history)
    contributions = list(blinds)
    folded: set[int] = set()

    for r in range(min(round_index, len(rounds))):
        start_player = first_player_pos(num_players, r)
        parse_round_events(
            rounds[r],
            first_player=start_player,
            num_players=num_players,
            folded_player_ids=folded,
            contributions=contributions,
        )

    return list(contributions)
