"""Core speed-order calculations for Pokemon Champions doubles.

The helpers in this module mirror the relevant integer rounding used by the
pinned Pokemon Showdown simulator for stat stages and chained speed modifiers.
The battle-state layer will later decide which modifiers are active.
"""

from dataclasses import dataclass
from enum import Enum


class TurnOrder(str, Enum):
    FIRST = "first"
    SECOND = "second"
    TIE = "tie"


@dataclass(frozen=True)
class SpeedState:
    """Known speed information for one Pokemon."""

    speed_stat: int
    stage: int = 0
    tailwind: bool = False
    unburden: bool = False
    choice_scarf: bool = False
    weather_speed_boost: bool = False
    paradox_speed_boost: bool = False
    paralyzed: bool = False


def calculate_speed_stat(
    base_stat: int,
    *,
    level: int = 50,
    iv: int = 31,
    ev: int = 0,
    nature_numerator: int = 1,
    nature_denominator: int = 1,
) -> int:
    """Calculate the visible Speed stat before in-battle modifiers.

    The default is level 50, 31 IVs, 0 EVs, neutral nature.
    Use 11/10 for a Speed-raising nature and 9/10 for a Speed-lowering nature.
    """

    if base_stat < 1:
        raise ValueError("base_stat must be positive")
    if not 1 <= level <= 100:
        raise ValueError("level must be between 1 and 100")
    if not 0 <= iv <= 31:
        raise ValueError("iv must be between 0 and 31")
    if not 0 <= ev <= 252:
        raise ValueError("ev must be between 0 and 252")
    if nature_numerator <= 0 or nature_denominator <= 0:
        raise ValueError("nature multiplier must be positive")

    stat = ((2 * base_stat + iv + ev // 4) * level) // 100 + 5
    return (stat * nature_numerator) // nature_denominator


def apply_speed_stage(speed: int, stage: int) -> int:
    """Apply a Pokemon stat stage using Showdown's stage table."""

    if speed < 1:
        raise ValueError("speed must be positive")

    stage = max(-6, min(6, stage))
    if stage >= 0:
        return (speed * (2 + stage)) // 2
    return (speed * 2) // (2 - stage)


def _chain_modifiers(modifiers: list[tuple[int, int]]) -> int:
    """Return a Showdown-style fixed-point chained modifier."""

    modifier = 4096
    for numerator, denominator in modifiers:
        next_modifier = (numerator * 4096) // denominator
        modifier = ((modifier * next_modifier + 2048) >> 12)
    return modifier


def _apply_chained_modifier(value: int, modifier: int) -> int:
    """Apply a Showdown-style fixed-point modifier to an integer stat."""

    return (value * modifier + 2048 - 1) // 4096


def effective_speed(state: SpeedState) -> int:
    """Return the supported in-battle effective Speed value."""

    speed = apply_speed_stage(state.speed_stat, state.stage)

    modifiers: list[tuple[int, int]] = []
    if state.tailwind:
        modifiers.append((2, 1))
    if state.unburden:
        modifiers.append((2, 1))
    if state.choice_scarf:
        modifiers.append((3, 2))
    if state.weather_speed_boost:
        modifiers.append((2, 1))
    if state.paradox_speed_boost:
        modifiers.append((3, 2))
    if state.paralyzed:
        modifiers.append((1, 2))

    if modifiers:
        speed = _apply_chained_modifier(speed, _chain_modifiers(modifiers))

    return min(speed, 10000)


def compare_actions(
    first: SpeedState,
    second: SpeedState,
    *,
    first_priority: int = 0,
    second_priority: int = 0,
    trick_room: bool = False,
) -> TurnOrder:
    """Determine whether the first or second action moves first.

    Priority is compared before Speed. Trick Room reverses Speed order only
    inside the same priority bracket. Exact Speed ties remain ties because the
    simulator resolves them randomly.
    """

    if first_priority > second_priority:
        return TurnOrder.FIRST
    if first_priority < second_priority:
        return TurnOrder.SECOND

    first_speed = effective_speed(first)
    second_speed = effective_speed(second)

    if first_speed == second_speed:
        return TurnOrder.TIE

    if trick_room:
        return TurnOrder.FIRST if first_speed < second_speed else TurnOrder.SECOND
    return TurnOrder.FIRST if first_speed > second_speed else TurnOrder.SECOND
