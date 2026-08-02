"""Power-related configuration and calculations."""

from __future__ import annotations

import os


POSITIVE_POWER_OFFSET = float(os.getenv("POSITIVE_POWER_OFFSET", "10.0"))
NEGATIVE_POWER_OFFSET = float(os.getenv("NEGATIVE_POWER_OFFSET", "10.0"))


def apply_total_power_offset(total_power: float) -> float:
    """Subtract the configured offset selected by the total's direction."""
    if total_power > 0:
        return total_power - POSITIVE_POWER_OFFSET
    if total_power < 0:
        return total_power - NEGATIVE_POWER_OFFSET
    return total_power
