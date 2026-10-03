import pytest

from vgc_bench.src.champions_ai.matchup import (
    CombatantProfile,
    MoveProfile,
    calculate_matchup_damage,
    resolve_damage_context,
)


def _mon(
    name: str,
    *,
    types: tuple[str, ...],
    stats: dict[str, int] | None = None,
    boosts: dict[str, int] | None = None,
    item: str | None = None,
    ability: str | None = None,
    status: str | None = None,
):
    return CombatantProfile(
        name=name,
        level=50,
        current_hp=150,
        max_hp=150,
        types=types,
        stats=stats
        or {
            "atk": 150,
            "def": 100,
            "spa": 150,
            "spd": 100,
            "spe": 100,
        },
        boosts=boosts or {},
        item=item,
        ability=ability,
        status=status,
    )


def test_resolver_uses_correct_physical_stats_and_stages() -> None:
    attacker = _mon(
        "attacker",
        types=("dragon",),
        boosts={"atk": 1},
    )
    defender = _mon(
        "defender",
        types=("normal",),
        boosts={"def": 1},
    )
    move = MoveProfile(
        move_id="dragonclaw",
        base_power=80,
        category="physical",
        move_type="dragon",
    )

    context = resolve_damage_context(attacker, defender, move)

    assert context.attack == 225
    assert context.defense == 150
    assert context.stab_modifier == (3, 2)


def test_type_effectiveness_is_resolved_from_gen9_chart() -> None:
    attacker = _mon("attacker", types=("fire",))
    grass = _mon("grass-target", types=("grass",))
    water = _mon("water-target", types=("water",))
    move = MoveProfile(
        move_id="flamethrower",
        base_power=90,
        category="special",
        move_type="fire",
    )

    vs_grass = resolve_damage_context(attacker, grass, move)
    vs_water = resolve_damage_context(attacker, water, move)

    assert vs_grass.type_modifier == 1
    assert vs_water.type_modifier == -1


def test_immunity_is_detected() -> None:
    attacker = _mon("attacker", types=("normal",))
    ghost = _mon("ghost-target", types=("ghost",))
    move = MoveProfile(
        move_id="tackle",
        base_power=40,
        category="physical",
        move_type="normal",
    )

    result = calculate_matchup_damage(attacker, ghost, move)

    assert result.rolls == (0,) * 16


def test_rain_changes_fire_and_water_damage() -> None:
    attacker = _mon("attacker", types=("fire", "water"))
    target = _mon("target", types=("normal",))

    fire = MoveProfile(
        move_id="flamethrower",
        base_power=90,
        category="special",
        move_type="fire",
    )
    water = MoveProfile(
        move_id="surf",
        base_power=90,
        category="special",
        move_type="water",
        spread=False,
    )

    fire_clear = calculate_matchup_damage(attacker, target, fire)
    fire_rain = calculate_matchup_damage(attacker, target, fire, weather="raindance")
    water_clear = calculate_matchup_damage(attacker, target, water)
    water_rain = calculate_matchup_damage(attacker, target, water, weather="raindance")

    assert fire_rain.maximum < fire_clear.maximum
    assert water_rain.minimum > water_clear.minimum


def test_burn_reduces_physical_damage_but_guts_ignores_penalty() -> None:
    burned = _mon("burned", types=("normal",), status="brn")
    guts = _mon(
        "guts",
        types=("normal",),
        status="brn",
        ability="guts",
    )
    target = _mon("target", types=("normal",))
    move = MoveProfile(
        move_id="bodyslam",
        base_power=85,
        category="physical",
        move_type="normal",
    )

    burned_result = calculate_matchup_damage(burned, target, move)
    guts_result = calculate_matchup_damage(guts, target, move)

    assert burned_result.maximum < guts_result.minimum


def test_life_orb_is_added_as_exact_final_modifier() -> None:
    attacker = _mon("attacker", types=("normal",), item="lifeorb")
    target = _mon("target", types=("normal",))
    move = MoveProfile(
        move_id="bodyslam",
        base_power=85,
        category="physical",
        move_type="normal",
    )

    context = resolve_damage_context(attacker, target, move)

    assert context.final_modifiers == ((5324, 4096),)


def test_critical_ignores_negative_attack_and_positive_defense_stages() -> None:
    attacker = _mon(
        "attacker",
        types=("normal",),
        boosts={"atk": -2},
    )
    defender = _mon(
        "defender",
        types=("normal",),
        boosts={"def": 2},
    )
    move = MoveProfile(
        move_id="bodyslam",
        base_power=85,
        category="physical",
        move_type="normal",
    )

    normal = resolve_damage_context(attacker, defender, move, critical=False)
    critical = resolve_damage_context(attacker, defender, move, critical=True)

    assert normal.attack == 75
    assert normal.defense == 200
    assert critical.attack == 150
    assert critical.defense == 100


def test_unknown_exact_stats_are_never_silently_invented() -> None:
    with pytest.raises(ValueError, match="missing exact stats"):
        CombatantProfile(
            name="unknown",
            level=50,
            current_hp=100,
            max_hp=100,
            types=("normal",),
            stats={"atk": 100},
        )
