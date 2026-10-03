import pytest

from vgc_bench.src.champions_ai.damage import (
    DamageContext,
    calculate_damage,
    combined_ko_probability,
)


def test_neutral_damage_rolls_match_showdown_integer_order() -> None:
    result = calculate_damage(
        DamageContext(
            level=50,
            base_power=100,
            attack=150,
            defense=100,
        )
    )

    assert result.rolls == (
        57,
        58,
        59,
        59,
        60,
        61,
        61,
        62,
        63,
        63,
        64,
        65,
        65,
        66,
        67,
        68,
    )


def test_spread_modifier_is_applied_before_random_roll() -> None:
    result = calculate_damage(
        DamageContext(
            level=50,
            base_power=100,
            attack=150,
            defense=100,
            spread=True,
        )
    )

    assert result.rolls == (
        43,
        43,
        44,
        44,
        45,
        45,
        46,
        46,
        47,
        47,
        48,
        48,
        49,
        49,
        50,
        51,
    )


def test_stab_and_super_effective_are_applied_in_correct_order() -> None:
    result = calculate_damage(
        DamageContext(
            level=50,
            base_power=100,
            attack=150,
            defense=100,
            stab_modifier=(3, 2),
            type_modifier=1,
        )
    )

    assert result.minimum == 170
    assert result.maximum == 204


def test_exact_single_hit_ko_probability() -> None:
    result = calculate_damage(
        DamageContext(
            level=50,
            base_power=100,
            attack=150,
            defense=100,
        )
    )

    assert result.ko_probability(60) == pytest.approx(0.75)


def test_combined_damage_ko_probability() -> None:
    result = calculate_damage(
        DamageContext(
            level=50,
            base_power=100,
            attack=150,
            defense=100,
        )
    )

    assert combined_ko_probability(120, result, result) == pytest.approx(
        222 / 256
    )


def test_immunity_returns_zero_damage() -> None:
    result = calculate_damage(
        DamageContext(
            level=50,
            base_power=100,
            attack=150,
            defense=100,
            immune=True,
        )
    )

    assert result.rolls == (0,) * 16
    assert result.ko_probability(1) == 0.0


def test_life_orb_style_final_modifier_matches_showdown_fixed_point() -> None:
    result = calculate_damage(
        DamageContext(
            level=50,
            base_power=100,
            attack=150,
            defense=100,
            final_modifiers=((5324, 4096),),
        )
    )

    assert result.rolls == (
        74,
        75,
        77,
        77,
        78,
        79,
        79,
        81,
        82,
        82,
        83,
        84,
        84,
        86,
        87,
        88,
    )
