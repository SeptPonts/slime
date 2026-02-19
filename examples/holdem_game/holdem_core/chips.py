"""Chip scaling utilities for OpenSpiel <-> dataset boundary.

OpenSpiel universal_poker requires integer chip amounts.
PokerBench dataset prompts use SB=0.5, BB=1, and show amounts with 0.1 precision
in preflop histories and pot sizes.

We model 1 dataset chip == `OPENSPIEL_CHIP_SCALE` OpenSpiel chips.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Final

OPENSPIEL_CHIP_SCALE: Final[int] = 10


DATASET_SMALL_BLIND: Final[Decimal] = Decimal("0.5")
DATASET_BIG_BLIND: Final[Decimal] = Decimal("1")
DATASET_STARTING_STACK: Final[Decimal] = Decimal("100")


def dataset_chips_to_openspiel(amount_dataset: Decimal) -> int:
    """
    Convert dataset-standard chip amount to OpenSpiel integer chips.

    Args:
        amount_dataset: Chip amount in dataset units (supports 0.1 precision).

    Returns:
        Integer chip amount for OpenSpiel.

    Raises:
        ValueError: If `amount_dataset` cannot be represented exactly.
    """
    scaled = amount_dataset * OPENSPIEL_CHIP_SCALE
    if scaled != scaled.to_integral_value():
        raise ValueError(
            f"Amount {amount_dataset} not representable with scale "
            f"{OPENSPIEL_CHIP_SCALE}"
        )
    return int(scaled)


def openspiel_chips_to_dataset_str(
    amount_openspiel: int,
    *,
    force_one_decimal: bool,
) -> str:
    """
    Format OpenSpiel integer chips as dataset-standard string.

    This avoids float rounding issues by formatting from integers directly.

    Args:
        amount_openspiel: Chip amount in OpenSpiel units (integer).
        force_one_decimal: If True, always emit a single decimal place (e.g. 48.0).

    Returns:
        String representation in dataset units.
    """
    whole, frac = divmod(amount_openspiel, OPENSPIEL_CHIP_SCALE)
    if frac == 0:
        return f"{whole}.0" if force_one_decimal else str(whole)
    return f"{whole}.{frac}"


def openspiel_chips_to_dataset_decimal(amount_openspiel: int) -> Decimal:
    """Convert OpenSpiel integer chips to dataset-standard Decimal chips.

    Args:
        amount_openspiel: Chip amount in OpenSpiel units (integer).

    Returns:
        Decimal representation in dataset units.
    """
    whole, frac = divmod(amount_openspiel, OPENSPIEL_CHIP_SCALE)
    return Decimal(whole) + (Decimal(frac) / Decimal(OPENSPIEL_CHIP_SCALE))


def format_preflop_amount(amount_openspiel: int) -> str:
    """Dataset preflop histories typically use one decimal place."""
    return openspiel_chips_to_dataset_str(amount_openspiel, force_one_decimal=True)


def format_postflop_amount(amount_openspiel: int) -> str:
    """Dataset postflop histories typically omit trailing .0."""
    return openspiel_chips_to_dataset_str(amount_openspiel, force_one_decimal=False)
