"""Exact core damage arithmetic for Champions AI.

This module mirrors the generation-9 integer ordering used by the pinned
Pokemon Showdown simulator for the standard damage formula. It expects the
caller to provide already-resolved battle inputs such as final Attack,
Defense, base power, STAB, weather, and final damage modifiers.

The later matchup layer will be responsible for deriving those inputs from
species, moves, abilities, items, terrain, weather, screens, and other effects.
"""

from dataclasses import dataclass
from itertools import product
from typing import TypeAlias


Rational: TypeAlias = tuple[int, int]


@dataclass(frozen=True)
class DamageContext:
    """Inputs needed by the generation-9 core damage formula."""

    level: int
    base_power: int
    attack: int
    defense: int
    spread: bool = False
    weather_modifier: Rational = (1, 1)
    critical: bool = False
    critical_modifier: Rational = (3, 2)
    stab_modifier: Rational = (1, 1)
    type_modifier: int = 0
    immune: bool = False
    burned: bool = False
    burn_modifier: Rational = (1, 2)
    final_modifiers: tuple[Rational, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("level", self.level),
            ("base_power", self.base_power),
            ("attack", self.attack),
            ("defense", self.defense),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        for modifier in (
            self.weather_modifier,
            self.critical_modifier,
            self.stab_modifier,
            self.burn_modifier,
            *self.final_modifiers,
        ):
            _validate_modifier(modifier)


@dataclass(frozen=True)
class DamageResult:
    """The 16 equally likely generation-9 random damage rolls."""

    rolls: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.rolls) != 16:
            raise ValueError("damage results must contain exactly 16 rolls")
        if any(damage < 0 for damage in self.rolls):
            raise ValueError("damage rolls cannot be negative")

    @property
    def minimum(self) -> int:
        return min(self.rolls)

    @property
    def maximum(self) -> int:
        return max(self.rolls)

    def percent_range(self, max_hp: int) -> tuple[float, float]:
        """Return minimum and maximum damage as percentages of max HP."""

        if max_hp <= 0:
            raise ValueError("max_hp must be positive")
        return (
            self.minimum * 100 / max_hp,
            self.maximum * 100 / max_hp,
        )

    def ko_probability(self, current_hp: int) -> float:
        """Return the exact probability of this hit KOing at current HP."""

        if current_hp <= 0:
            return 1.0
        kos = sum(damage >= current_hp for damage in self.rolls)
        return kos / len(self.rolls)


def _validate_modifier(modifier: Rational) -> None:
    numerator, denominator = modifier
    if numerator < 0:
        raise ValueError("modifier numerator cannot be negative")
    if denominator <= 0:
        raise ValueError("modifier denominator must be positive")


def showdown_modify(value: int, modifier: Rational) -> int:
    """Apply Pokemon Showdown's fixed-point modifier rounding."""

    _validate_modifier(modifier)
    numerator, denominator = modifier
    fixed_modifier = numerator * 4096 // denominator
    return (value * fixed_modifier + 2047) // 4096


def chain_modifiers(modifiers: tuple[Rational, ...]) -> int:
    """Combine final modifiers using Showdown's 4096-based chaining."""

    chained = 4096
    for numerator, denominator in modifiers:
        _validate_modifier((numerator, denominator))
        next_modifier = numerator * 4096 // denominator
        chained = (chained * next_modifier + 2048) >> 12
    return chained


def apply_chained_modifier(value: int, chained_modifier: int) -> int:
    """Apply a previously chained fixed-point modifier."""

    if chained_modifier < 0:
        raise ValueError("chained_modifier cannot be negative")
    return (value * chained_modifier + 2047) // 4096


def base_damage(context: DamageContext) -> int:
    """Return damage before spread, weather, random, STAB, and later modifiers."""

    level_factor = 2 * context.level // 5 + 2
    damage = level_factor * context.base_power * context.attack
    damage //= context.defense
    damage //= 50
    return damage + 2


def _apply_type_modifier(damage: int, type_modifier: int) -> int:
    if type_modifier > 0:
        for _ in range(type_modifier):
            damage *= 2
    elif type_modifier < 0:
        for _ in range(-type_modifier):
            damage //= 2
    return damage


def calculate_damage(context: DamageContext) -> DamageResult:
    """Calculate all 16 standard generation-9 damage rolls.

    The ordering mirrors Pokemon Showdown:
    base formula -> +2 -> spread -> weather -> critical -> random -> STAB ->
    type effectiveness -> burn -> final chained modifiers.
    """

    if context.immune:
        return DamageResult((0,) * 16)

    damage = base_damage(context)

    if context.spread:
        damage = showdown_modify(damage, (3, 4))

    if context.weather_modifier != (1, 1):
        damage = showdown_modify(damage, context.weather_modifier)

    if context.critical:
        numerator, denominator = context.critical_modifier
        damage = damage * numerator // denominator

    final_modifier = (
        chain_modifiers(context.final_modifiers)
        if context.final_modifiers
        else None
    )

    rolls: list[int] = []
    for random_factor in range(85, 101):
        roll = damage * random_factor // 100

        if context.stab_modifier != (1, 1):
            roll = showdown_modify(roll, context.stab_modifier)

        roll = _apply_type_modifier(roll, context.type_modifier)

        if context.burned:
            roll = showdown_modify(roll, context.burn_modifier)

        if final_modifier is not None:
            roll = apply_chained_modifier(roll, final_modifier)

        rolls.append(max(1, roll))

    return DamageResult(tuple(rolls))


def combined_ko_probability(
    current_hp: int,
    *results: DamageResult,
) -> float:
    """Return exact KO probability for multiple independent damage rolls."""

    if current_hp <= 0:
        return 1.0
    if not results:
        return 0.0

    total_outcomes = 16 ** len(results)
    ko_outcomes = sum(
        sum(outcome) >= current_hp
        for outcome in product(*(result.rolls for result in results))
    )
    return ko_outcomes / total_outcomes
