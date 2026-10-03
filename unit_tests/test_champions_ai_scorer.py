from vgc_bench.src.champions_ai.actions import (
    ActionKind,
    JointAction,
    SlotAction,
)
from vgc_bench.src.champions_ai.events import BattleEvent
from vgc_bench.src.champions_ai.grades import GRADE_ORDER, DecisionGrade
from vgc_bench.src.champions_ai.scorer import evaluate_action, rank_actions
from vgc_bench.src.champions_ai.snapshot import DecisionSnapshot
from vgc_bench.src.champions_ai.state import BattleState


def _snapshot(*, spread_threat: bool = True, protect_streak: int = 0):
    state = BattleState()

    state.apply(
        BattleEvent(type="switch_in", side="player", slot=0, pokemon="garchomp")
    )
    state.apply(
        BattleEvent(type="switch_in", side="player", slot=1, pokemon="whimsicott")
    )
    state.apply(
        BattleEvent(type="switch_in", side="opponent", slot=0, pokemon="salamence")
    )
    state.apply(
        BattleEvent(type="switch_in", side="opponent", slot=1, pokemon="sneasler")
    )

    if spread_threat:
        state.apply(
            BattleEvent(
                type="move_revealed",
                side="opponent",
                pokemon="salamence",
                move="Hyper Voice",
            )
        )

    if protect_streak:
        state.apply(
            BattleEvent(
                type="protect_streak_changed",
                side="player",
                pokemon="garchomp",
                value=protect_streak,
            )
        )

    return DecisionSnapshot.from_state(state)


def _move(slot: int, actor: str, move: str, target: str | None = None):
    target_position = None
    if target == "salamence":
        target_position = 1
    elif target == "sneasler":
        target_position = 2

    return SlotAction(
        slot=slot,
        kind=ActionKind.MOVE,
        actor=actor,
        move=move,
        target=target,
        target_position=target_position,
    )


def test_grade_names_are_locked_in_requested_order() -> None:
    assert GRADE_ORDER == (
        DecisionGrade.STUPENDOUS,
        DecisionGrade.AMAZING,
        DecisionGrade.OUTSTANDING,
        DecisionGrade.AWESOME,
        DecisionGrade.GREAT,
        DecisionGrade.GOOD,
        DecisionGrade.OK,
        DecisionGrade.MISTAKE,
        DecisionGrade.MISS,
        DecisionGrade.THROWING,
    )


def test_protect_plus_pressure_beats_double_exposure_to_revealed_spread() -> None:
    snapshot = _snapshot(spread_threat=True)

    protect_pressure = JointAction(
        first=_move(0, "garchomp", "Protect"),
        second=_move(1, "whimsicott", "Moonblast", "salamence"),
    )
    double_attack = JointAction(
        first=_move(0, "garchomp", "Dragon Claw", "salamence"),
        second=_move(1, "whimsicott", "Moonblast", "salamence"),
    )

    ranked = rank_actions(snapshot, [double_attack, protect_pressure])

    assert ranked[0].action == protect_pressure


def test_repeated_protect_is_penalized() -> None:
    fresh = _snapshot(spread_threat=True, protect_streak=0)
    repeated = _snapshot(spread_threat=True, protect_streak=2)

    action = JointAction(
        first=_move(0, "garchomp", "Protect"),
        second=_move(1, "whimsicott", "Moonblast", "salamence"),
    )

    fresh_eval = evaluate_action(fresh, action)
    repeated_eval = evaluate_action(repeated, action)

    assert repeated_eval.heuristic_score < fresh_eval.heuristic_score
    assert repeated_eval.signals.repeated_protect_slots == 1


def test_baseline_does_not_pretend_to_have_win_probability_yet() -> None:
    snapshot = _snapshot(spread_threat=False)
    action = JointAction(
        first=_move(0, "garchomp", "Dragon Claw", "salamence"),
        second=_move(1, "whimsicott", "Moonblast", "sneasler"),
    )

    evaluation = evaluate_action(snapshot, action)

    assert evaluation.estimated_win_probability is None
    assert evaluation.model == "heuristic_v0"


def test_passive_actions_are_penalized() -> None:
    snapshot = _snapshot(spread_threat=False)

    attack = JointAction(
        first=_move(0, "garchomp", "Dragon Claw", "salamence"),
        second=_move(1, "whimsicott", "Moonblast", "sneasler"),
    )
    passive = JointAction(
        first=SlotAction(slot=0, kind=ActionKind.PASS, actor="garchomp"),
        second=SlotAction(slot=1, kind=ActionKind.DEFAULT, actor="whimsicott"),
    )

    ranked = rank_actions(snapshot, [passive, attack])

    assert ranked[0].action == attack
