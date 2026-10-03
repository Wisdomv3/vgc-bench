from dataclasses import replace

from vgc_bench.src.champions_ai.matchup import CombatantProfile
from vgc_bench.src.champions_ai.position_value import evaluate_position
from vgc_bench.src.champions_ai.turn_order import TurnSide
from vgc_bench.src.champions_ai.turn_simulator import ExactTurnState


def _profile(name: str, hp: int = 100, status: str | None = None):
    return CombatantProfile(
        name=name,
        level=50,
        current_hp=hp,
        max_hp=100,
        types=("normal",),
        stats={
            "atk": 100,
            "def": 100,
            "spa": 100,
            "spd": 100,
            "spe": 100,
        },
        status=status,
    )


def _state() -> ExactTurnState:
    return ExactTurnState(
        profiles={
            (TurnSide.PLAYER, "a"): _profile("a"),
            (TurnSide.PLAYER, "b"): _profile("b"),
            (TurnSide.OPPONENT, "x"): _profile("x"),
            (TurnSide.OPPONENT, "y"): _profile("y"),
        },
        active_slots={
            (TurnSide.PLAYER, 0): "a",
            (TurnSide.PLAYER, 1): "b",
            (TurnSide.OPPONENT, 0): "x",
            (TurnSide.OPPONENT, 1): "y",
        },
    )


def test_symmetric_position_scores_zero() -> None:
    assert evaluate_position(_state()).score == 0


def test_opponent_faint_is_large_positive_swing() -> None:
    state = _state()
    state.profiles[(TurnSide.OPPONENT, "x")] = replace(
        state.profile(TurnSide.OPPONENT, "x"),
        current_hp=0,
    )

    assert evaluate_position(state).score > 100


def test_losing_our_pokemon_is_negative() -> None:
    state = _state()
    state.profiles[(TurnSide.PLAYER, "a")] = replace(
        state.profile(TurnSide.PLAYER, "a"),
        current_hp=0,
    )

    assert evaluate_position(state).score < -100


def test_status_on_opponent_improves_score() -> None:
    healthy = _state()
    statused = _state()
    statused.profiles[(TurnSide.OPPONENT, "x")] = replace(
        statused.profile(TurnSide.OPPONENT, "x"),
        status="par",
    )

    assert evaluate_position(statused).score > evaluate_position(healthy).score


def test_tailwind_control_changes_score() -> None:
    player_tailwind = _state()
    player_tailwind.tailwind_sides.add(TurnSide.PLAYER)

    opponent_tailwind = _state()
    opponent_tailwind.tailwind_sides.add(TurnSide.OPPONENT)

    assert evaluate_position(player_tailwind).score > 0
    assert evaluate_position(opponent_tailwind).score < 0
