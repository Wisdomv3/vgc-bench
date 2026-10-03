"""Doubles field mechanics used by the Champions AI turn simulator."""

from poke_env.battle import Move

from vgc_bench.src.champions_ai.matchup import CombatantProfile
from vgc_bench.src.champions_ai.spread import normalize_move_id


REDIRECTABLE_TARGETS = {
    "NORMAL",
    "ADJACENT_FOE",
    "RANDOM_NORMAL",
    "ANY",
}

WIDE_GUARD_TARGETS = {
    "ALL_ADJACENT",
    "ALL_ADJACENT_FOES",
}


def is_grounded(
    profile: CombatantProfile,
    *,
    field_conditions: set[str] | frozenset[str] = frozenset(),
) -> bool:
    """Return the supported grounded state used by terrain mechanics."""

    normalized_conditions = {
        normalize_move_id(condition)
        for condition in field_conditions
    }
    if "gravity" in normalized_conditions:
        return True

    item = normalize_move_id(profile.item or "")
    if item == "ironball":
        return True
    if item == "airballoon":
        return False

    ability = normalize_move_id(profile.ability or "")
    if ability == "levitate":
        return False

    types = {
        normalize_move_id(type_name)
        for type_name in profile.types
    }
    if "flying" in types:
        return False

    return True


def psychic_terrain_blocks_priority(
    target: CombatantProfile,
    *,
    priority: int,
    source_is_ally: bool,
    terrain: str | None,
    field_conditions: set[str] | frozenset[str] = frozenset(),
) -> bool:
    """Return whether Psychic Terrain blocks this positive-priority hit."""

    if normalize_move_id(terrain or "") != "psychicterrain":
        return False
    if priority <= 0 or source_is_ally:
        return False
    return is_grounded(target, field_conditions=field_conditions)


def wide_guard_blocks(move: Move) -> bool:
    target = move.target
    return target is not None and target.name in WIDE_GUARD_TARGETS


def move_can_be_redirected(move: Move) -> bool:
    target = move.target
    return target is not None and target.name in REDIRECTABLE_TARGETS


def bypasses_redirection(
    attacker: CombatantProfile,
    move: Move,
) -> bool:
    """Return whether the move/user ignores Follow Me and Rage Powder."""

    ability = normalize_move_id(attacker.ability or "")
    if ability in {"stalwart", "propellertail"}:
        return True

    return bool(move.entry.get("tracksTarget", False))


def powder_redirection_immune(attacker: CombatantProfile) -> bool:
    """Return whether Rage Powder fails to redirect this attacker."""

    ability = normalize_move_id(attacker.ability or "")
    item = normalize_move_id(attacker.item or "")
    types = {
        normalize_move_id(type_name)
        for type_name in attacker.types
    }

    return (
        "grass" in types
        or ability == "overcoat"
        or item == "safetygoggles"
    )
