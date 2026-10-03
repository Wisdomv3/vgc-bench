"""Resolve real Pokemon/move matchups into exact damage inputs.

This layer bridges battle information to the core arithmetic in damage.py.
It intentionally refuses to invent unknown stats. Opponent uncertainty will be
handled later by hidden-set sampling.
"""

from dataclasses import dataclass, field

from poke_env.battle import Move, MoveCategory, Pokemon
from poke_env.data import GenData

from vgc_bench.src.champions_ai.damage import (
    DamageContext,
    DamageResult,
    Rational,
    calculate_damage,
)
from vgc_bench.src.champions_ai.spread import is_spread_move, normalize_move_id


@dataclass(frozen=True)
class CombatantProfile:
    """Exact information needed to calculate damage for one Pokemon."""

    name: str
    level: int
    current_hp: int
    max_hp: int
    types: tuple[str, ...]
    stats: dict[str, int]
    boosts: dict[str, int] = field(default_factory=dict)
    item: str | None = None
    ability: str | None = None
    status: str | None = None
    gender: str | None = None

    def __post_init__(self) -> None:
        if self.level <= 0:
            raise ValueError("level must be positive")
        if self.current_hp < 0:
            raise ValueError("current_hp cannot be negative")
        if self.max_hp <= 0:
            raise ValueError("max_hp must be positive")
        if self.current_hp > self.max_hp:
            raise ValueError("current_hp cannot exceed max_hp")

        required = {"atk", "def", "spa", "spd", "spe"}
        missing = required.difference(self.stats)
        if missing:
            raise ValueError(f"missing exact stats: {sorted(missing)}")
        if any(self.stats[stat] <= 0 for stat in required):
            raise ValueError("all exact stats must be positive")


@dataclass(frozen=True)
class MoveProfile:
    """Resolved move information used by the damage engine."""

    move_id: str
    base_power: int
    category: str
    move_type: str
    spread: bool = False

    def __post_init__(self) -> None:
        category = self.category.lower()
        if category not in {"physical", "special"}:
            raise ValueError("damage matchup requires a physical or special move")
        if self.base_power <= 0:
            raise ValueError("base_power must be positive")


def _apply_stat_stage(stat: int, stage: int) -> int:
    """Apply the standard Pokemon stat-stage multiplier."""

    stage = max(-6, min(6, stage))
    if stage >= 0:
        return stat * (2 + stage) // 2
    return stat * 2 // (2 - stage)


def _type_modifier(move_type: str, defender_types: tuple[str, ...], gen: int) -> int | None:
    """Return Showdown-style type stage, or None for immunity."""

    if normalize_move_id(move_type) in {"threequestionmarks", ""} or move_type == "???":
        return 0

    data = GenData.from_gen(gen)
    attack_type = move_type.upper()

    multiplier = 1.0
    for defender_type in defender_types:
        multiplier *= data.type_chart[defender_type.upper()][attack_type]

    if multiplier == 0:
        return None

    mapping = {
        0.25: -2,
        0.5: -1,
        1.0: 0,
        2.0: 1,
        4.0: 2,
    }
    if multiplier not in mapping:
        raise ValueError(f"unsupported type multiplier: {multiplier}")
    return mapping[multiplier]


def _weather_modifier(weather: str | None, move_type: str) -> Rational:
    if weather is None:
        return (1, 1)

    weather_id = normalize_move_id(weather)
    type_id = move_type.lower()

    if weather_id in {"raindance", "rain"}:
        if type_id == "water":
            return (3, 2)
        if type_id == "fire":
            return (1, 2)

    if weather_id in {"sunnyday", "sun", "harshsunlight"}:
        if type_id == "fire":
            return (3, 2)
        if type_id == "water":
            return (1, 2)

    return (1, 1)


def _stab_modifier(attacker: CombatantProfile, move_type: str) -> Rational:
    has_stab = move_type.lower() in {type_name.lower() for type_name in attacker.types}
    if not has_stab:
        return (1, 1)

    if normalize_move_id(attacker.ability or "") == "adaptability":
        return (2, 1)

    return (3, 2)


def _final_modifiers(attacker: CombatantProfile) -> tuple[Rational, ...]:
    modifiers: list[Rational] = []

    if normalize_move_id(attacker.item or "") == "lifeorb":
        # Pokemon Showdown's exact Life Orb modifier.
        modifiers.append((5324, 4096))

    return tuple(modifiers)


def profile_from_poke_env(pokemon: Pokemon) -> CombatantProfile:
    """Build an exact profile from poke-env, refusing unknown battle stats."""

    exact_stats: dict[str, int] = {}
    for stat in ("atk", "def", "spa", "spd", "spe"):
        value = pokemon.stats.get(stat)
        if value is None:
            raise ValueError(
                f"Exact {stat} is unknown for {pokemon.species}; "
                "use a sampled or manually supplied CombatantProfile instead"
            )
        exact_stats[stat] = int(value)

    if pokemon.max_hp <= 0:
        raise ValueError(f"Exact max HP is unknown for {pokemon.species}")

    status = pokemon.status.name.lower() if pokemon.status is not None else None

    return CombatantProfile(
        name=pokemon.species or pokemon.name,
        level=pokemon.level,
        current_hp=pokemon.current_hp,
        max_hp=pokemon.max_hp,
        types=tuple(pokemon_type.name.lower() for pokemon_type in pokemon.types),
        stats=exact_stats,
        boosts={stat: int(stage) for stat, stage in pokemon.boosts.items()},
        item=pokemon.item,
        ability=pokemon.ability,
        status=status,
        gender=(
            pokemon.gender.name.lower()
            if pokemon.gender is not None
            else None
        ),
    )


def move_profile_from_poke_env(
    move: Move,
    *,
    spread: bool | None = None,
    base_power: int | None = None,
) -> MoveProfile:
    """Build a resolved move profile from poke-env.

    Dynamic-power moves may pass an explicit base_power override later.
    """

    if move.category is MoveCategory.STATUS:
        raise ValueError("status moves do not use the standard damage formula")

    return MoveProfile(
        move_id=move.id,
        base_power=move.base_power if base_power is None else base_power,
        category=move.category.name.lower(),
        move_type=move.type.name.lower(),
        spread=is_spread_move(move.id) if spread is None else spread,
    )


def resolve_damage_context(
    attacker: CombatantProfile,
    defender: CombatantProfile,
    move: MoveProfile,
    *,
    weather: str | None = None,
    critical: bool = False,
) -> DamageContext:
    """Resolve profiles into an exact core DamageContext."""

    category = move.category.lower()
    attack_stat_name = "atk" if category == "physical" else "spa"
    defense_stat_name = "def" if category == "physical" else "spd"

    attack_stage = int(attacker.boosts.get(attack_stat_name, 0))
    defense_stage = int(defender.boosts.get(defense_stat_name, 0))

    # Generation 6+ critical hits ignore the attacker's negative offensive
    # stages and the defender's positive defensive stages.
    if critical and attack_stage < 0:
        attack_stage = 0
    if critical and defense_stage > 0:
        defense_stage = 0

    attack = _apply_stat_stage(attacker.stats[attack_stat_name], attack_stage)
    defense = _apply_stat_stage(defender.stats[defense_stat_name], defense_stage)

    type_modifier = _type_modifier(move.move_type, defender.types, gen=9)
    immune = type_modifier is None

    burned = (
        category == "physical"
        and attacker.status == "brn"
        and normalize_move_id(attacker.ability or "") != "guts"
        and normalize_move_id(move.move_id) != "facade"
    )

    return DamageContext(
        level=attacker.level,
        base_power=move.base_power,
        attack=attack,
        defense=defense,
        spread=move.spread,
        weather_modifier=_weather_modifier(weather, move.move_type),
        critical=critical,
        stab_modifier=_stab_modifier(attacker, move.move_type),
        type_modifier=0 if immune else type_modifier,
        immune=immune,
        burned=burned,
        final_modifiers=_final_modifiers(attacker),
    )


def calculate_matchup_damage(
    attacker: CombatantProfile,
    defender: CombatantProfile,
    move: MoveProfile,
    *,
    weather: str | None = None,
    critical: bool = False,
) -> DamageResult:
    """Resolve a matchup and return its exact 16-roll damage distribution."""

    return calculate_damage(
        resolve_damage_context(
            attacker,
            defender,
            move,
            weather=weather,
            critical=critical,
        )
    )
