from vgc_bench.src.champions_ai.speed import (
    SpeedState,
    TurnOrder,
    apply_speed_stage,
    calculate_speed_stat,
    compare_actions,
    effective_speed,
)


def test_level_50_speed_stat_calculation() -> None:
    # Garchomp base 102 Speed, 31 IV, 252 EV, neutral nature.
    assert calculate_speed_stat(102, ev=252) == 154
    assert calculate_speed_stat(
        102, ev=252, nature_numerator=11, nature_denominator=10
    ) == 169


def test_speed_stages() -> None:
    assert apply_speed_stage(100, 1) == 150
    assert apply_speed_stage(100, 2) == 200
    assert apply_speed_stage(100, -1) == 66
    assert apply_speed_stage(100, -2) == 50


def test_tailwind_and_unburden_stack() -> None:
    state = SpeedState(speed_stat=100, tailwind=True, unburden=True)
    assert effective_speed(state) == 400


def test_choice_scarf_and_paralysis() -> None:
    state = SpeedState(speed_stat=100, choice_scarf=True, paralyzed=True)
    assert effective_speed(state) == 75


def test_normal_turn_order() -> None:
    fast = SpeedState(speed_stat=150)
    slow = SpeedState(speed_stat=100)
    assert compare_actions(fast, slow) is TurnOrder.FIRST


def test_trick_room_reverses_speed_order() -> None:
    fast = SpeedState(speed_stat=150)
    slow = SpeedState(speed_stat=100)
    assert compare_actions(fast, slow, trick_room=True) is TurnOrder.SECOND


def test_priority_beats_speed_even_in_trick_room() -> None:
    slow = SpeedState(speed_stat=50)
    fast = SpeedState(speed_stat=200)
    assert (
        compare_actions(
            slow,
            fast,
            first_priority=1,
            second_priority=0,
            trick_room=True,
        )
        is TurnOrder.FIRST
    )


def test_exact_speed_tie_is_reported() -> None:
    left = SpeedState(speed_stat=123)
    right = SpeedState(speed_stat=123)
    assert compare_actions(left, right) is TurnOrder.TIE
